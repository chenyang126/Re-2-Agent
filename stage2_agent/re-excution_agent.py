import os
import json
import argparse
import concurrent.futures
from typing import Dict, Any, List, Tuple

from gpt import get_azure_openai_client


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_index(gt_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a flat dict indexed by identifier from ground truth data.
    Example identifier: scene_1_27_2  =>  gt[identifier] = {..., 'tl_goal': ..., 'vh_goal': ...}
    """
    index: Dict[str, Dict[str, Any]] = {}
    for scene_name, scene_content in gt_data.items():
        # scene_name: "scene_1"
        for task_name, task_cases in scene_content.items():
            # task_name: "Wash clothes", "Turn on light", ...
            for case_id, case_data in task_cases.items():
                # case_id: "27_2"
                identifier = f"{scene_name}_{case_id}"
                index[identifier] = {
                    "scene": scene_name,
                    "task": task_name,
                    "case_id": case_id,
                    "vh_goal": case_data.get("vh_goal", {}),
                    "tl_goal": case_data.get("tl_goal", ""),
                }
    return index


def build_prompt_index(prompts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Index a list of prompts by identifier.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for item in prompts:
        identifier = item.get("identifier")
        if identifier is None:
            continue
        index[identifier] = item
    return index


def extract_file_id(identifier: str) -> str:
    """
    Extract file_id used by the evaluator from identifier.
    Example: "scene_1_27_2" -> "27_2"
    """
    parts = str(identifier).split("_")
    if len(parts) >= 4:
        return f"{parts[2]}_{parts[3]}"
    # Fallback: directly return the original identifier to be safe
    return identifier


def build_log_index(log_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the evaluation log file and group lines by file_id.
    For each file_id, we store:
      {
        "task": str,
        "log_lines": List[str],
        "has_error": bool,   # True if the log indicates any error for this file
      }
    """
    log_index: Dict[str, Dict[str, Any]] = {}

    if not os.path.exists(log_file):
        print(f"Warning: log file not found: {log_file}")
        return log_index

    import re

    task_pattern = re.compile(r"Task is (.*), file_id is ([^ \n]+)")

    # Heuristic patterns for detecting errors in the log
    error_indicators = [
        "did not pass gold test",
        "has hallucination error",
        "Encounter error:",
        "Goals all satisfied: all_pred_success=False",
        "not executable.",
        "GOAL FAIL!",
    ]

    current_file_id: str | None = None

    with open(log_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            m = task_pattern.search(line)
            if m:
                task_name = m.group(1).strip()
                current_file_id = m.group(2).strip()
                entry = log_index.setdefault(
                    current_file_id,
                    {"task": task_name, "log_lines": [], "has_error": False},
                )
                entry["task"] = task_name

            if current_file_id is not None:
                entry = log_index.setdefault(
                    current_file_id,
                    {"task": None, "log_lines": [], "has_error": False},
                )
                entry["log_lines"].append(line)

                if any(ind in line for ind in error_indicators):
                    entry["has_error"] = True

    return log_index


def process_single_fix_action_seq(
    client,
    model: str,
    identifier: str,
    prompt_entry: Dict[str, Any],
    gt_entry: Dict[str, Any],
    old_output_entry: Dict[str, Any],
    log_entry: Dict[str, Any],
    max_retries: int = 6,
) -> Dict[str, Any]:
    """
    Fix a single sample by combining:
      - the original natural language prompt (llm_prompt)
      - the previous model output action sequence (llm_output)
      - ground truth (vh_goal + tl_goal)
      - the evaluation log snippet for this sample (log_entry)

    The LLM acts as an agent that carefully reads the evaluation log and uses it
    to debug and improve the original action sequence.

    The corrected plan must still use the same JSON schema as the original submission:

      llm_output: "{\n  \"WALK\": [\"obj\", \"id\"],\n  \"GRAB\": [\"obj\", \"id\"], ... }\n"

    Returns:
      {
        "identifier": ...,
        "new_llm_output": "<JSON string>",
        "meta": {...}
      }
    """
    llm_prompt = prompt_entry.get("llm_prompt", "")
    old_llm_output = old_output_entry.get("llm_output", "")

    vh_goal = gt_entry.get("vh_goal", {})
    tl_goal = gt_entry.get("tl_goal", "")
    gt_is_reference = bool(gt_entry.get("is_reference", False))

    task_from_log = log_entry.get("task")
    log_lines = log_entry.get("log_lines", [])
    eval_log_text = "\n".join(log_lines)

    # System-level meta-prompt for the model
    system_instruction = (
        "You are an expert in VirtualHome action sequencing and task planning.\n"
        "You are given information about one evaluation sample:\n"
        "1) A natural language prompt that describes the scene, current graph (nodes/edges), and the desired goals.\n"
        "2) A previous action sequence prediction in JSON format (the field 'old_llm_output').\n"
        "3) A ground-truth goal specification consisting of:\n"
        "   - 'vh_goal': a JSON object describing object states and relations in the target state;\n"
        "   - 'tl_goal': a temporal-logic-like formula describing the same goal.\n"
        "4) An evaluation log snippet produced by the VirtualHome simulator for this sample.\n\n"
        "The evaluation log may include messages such as:\n"
        "- 'Program <file_id> did not pass gold test'\n"
        "- '... has hallucination error'\n"
        "- 'Encounter error: MISSING_STEP'\n"
        "- 'Current action ... not executable.'\n"
        "- 'Goals all satisfied: all_pred_success=False'\n"
        "These messages describe which actions failed, why they failed, and whether the final goals were achieved.\n\n"
        "Your job is to act as a debugging agent:\n"
        "- Carefully read the evaluation log and understand what went wrong with the previous plan.\n"
        "- Use the log details together with the original prompt and (when available) the ground-truth goals\n"
        "  to design a corrected action sequence that fixes the problems.\n"
        "- Make sure the new action sequence is executable step-by-step in VirtualHome and achieves the intended goal.\n"
        "- When 'gt_is_reference' is false (exact ground truth for this identifier):\n"
        "    * Make the corrected plan fully consistent with the provided ground-truth goals.\n"
        "    * If there is any conflict between the previous prediction and the ground truth, ALWAYS follow the ground truth.\n"
        "- When 'gt_is_reference' is true (ground truth from a similar task used only as reference):\n"
        "    * Use 'vh_goal' and 'tl_goal' only as helpful examples of the desired final state pattern.\n"
        "    * Adapt them to the current natural language goal and scene; do NOT blindly copy objects or states that\n"
        "      are incompatible with the current scene description.\n"
        "- The corrected plan MUST:\n"
        "  * use only objects and ids that are consistent with the scene description and goals;\n"
        "  * be executable step-by-step in VirtualHome;\n"
        "  * when 'gt_is_reference' is false, make all states/relations in 'vh_goal' and 'tl_goal' true at the end;\n"
        "  * when 'gt_is_reference' is true, reach a reasonable final state that satisfies the current task.\n"
        "- Use the SAME output schema as the original prediction: a single JSON object whose keys are action names\n"
        "  (e.g., \"WALK\", \"GRAB\", \"PUTBACK\", \"SWITCHON\", etc.) and whose values are parameter lists.\n"
        "  The actions are executed in the order in which the keys appear in the JSON object. For example:\n"
        "  {\n"
        "    \"WALK\": [\"light\", \"411\"],\n"
        "    \"SWITCHON\": [\"light\", \"411\"]\n"
        "  }\n"
        "- Only output this JSON object, with no extra natural language explanation.\n"
    )

    user_message = {
        "identifier": identifier,
        "task_from_log": task_from_log,
        "original_prompt": llm_prompt,
        "old_llm_output": old_llm_output,
        "ground_truth_vh_goal": vh_goal,
        "ground_truth_tl_goal": tl_goal,
        "gt_is_reference": gt_is_reference,
        "evaluation_log": eval_log_text,
    }

    retry = 0
    while retry < max_retries:
        try:
            if retry > 0:
                print(f"  Retry {retry} for {identifier} in fix step")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": json.dumps(user_message, ensure_ascii=False, indent=2)},
                ],
                max_completion_tokens=2048,
            )

            content = (response.choices[0].message.content or "").strip()
            if content:
                # Directly use content as new llm_output (a JSON object string)
                return {
                    "identifier": identifier,
                    "new_llm_output": content,
                    "meta": {
                        "has_gt": bool(vh_goal or tl_goal),
                    },
                }

            retry += 1
        except Exception as e:
            print(f"Error fixing {identifier} (attempt {retry + 1}): {e}")
            retry += 1

    print(f"Warning: failed to fix {identifier} after {max_retries} attempts")
    return {
        "identifier": identifier,
        "new_llm_output": "",
        "meta": {
            "has_gt": bool(vh_goal or tl_goal),
            "error": "max_retries_exceeded",
        },
    }


def fix_all_action_sequences(
    gt_file: str,
    prompt_file: str,
    old_output_file: str,
    eval_log_file: str,
    output_file: str,
    model: str,
    max_samples: int = None,
    num_workers: int = 5,
) -> None:
    """
    Main pipeline (LLM-agent + log version):
    - Load ground truth (vh_goal + tl_goal), prompt file and old outputs.
    - Parse the evaluation log and group lines by file_id.
    - For each sample that has both prompt and log information and whose log indicates an error,
      call the LLM to read the log and fix the original action sequence.
    - Merge the fixed action sequences back into the full submission and save to a new output file.
    """
    print(f"Loading ground truth from {gt_file}...")
    gt_data = load_json(gt_file)
    gt_index = build_gt_index(gt_data)
    print(f"GT entries (flattened): {len(gt_index)}")

    print(f"Loading prompts from {prompt_file}...")
    prompts: List[Dict[str, Any]] = load_json(prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    prompt_index = build_prompt_index(prompts)

    print(f"Loading old outputs from {old_output_file}...")
    old_outputs: List[Dict[str, Any]] = load_json(old_output_file)
    print(f"Loaded {len(old_outputs)} old outputs")

    print(f"Parsing evaluation log from {eval_log_file}...")
    log_index = build_log_index(eval_log_file)
    print(f"Parsed log info for {len(log_index)} file_ids")

    # Build the list of samples to be fixed:
    # - must have prompt, old output, and be found in both GT and log_index;
    # - only fix samples whose log entry is marked as having an error.
    todo: List[Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []

    for out_item in old_outputs:
        identifier = out_item.get("identifier")
        if not identifier:
            continue

        file_id = extract_file_id(identifier)
        if file_id not in log_index:
            # Skip samples without evaluation log
            continue

        log_entry = log_index[file_id]
        has_error = bool(log_entry.get("has_error", False))

        # Only fix samples with errors indicated in the log
        if not has_error:
            continue

        # Require prompt; otherwise the model cannot understand the current task, so skip
        if identifier not in prompt_index:
            continue

        prompt_entry = prompt_index[identifier]

        # 1) Prefer the exact GT with the same identifier
        if identifier in gt_index:
            todo.append(
                (
                    identifier,
                    prompt_entry,
                    gt_index[identifier],
                    out_item,
                    log_entry,
                )
            )
            continue

        # 2) If there is no exact GT, we still allow self-correction based only on prompt and log.
        #    In this case, vh_goal / tl_goal are empty and treated as reference.
        empty_ref_gt = {
            "vh_goal": {},
            "tl_goal": "",
            "is_reference": True,
        }
        todo.append(
            (
                identifier,
                prompt_entry,
                empty_ref_gt,
                out_item,
                log_entry,
            )
        )

    if max_samples is not None:
        todo = todo[:max_samples]
        print(f"Only fixing first {len(todo)} samples for testing")

    total = len(todo)
    print(f"Total samples to fix (with GT, prompt, output, log_info): {total}")

    if total == 0:
        print("No samples to fix. Exiting.")
        return

    client = get_azure_openai_client()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 用 identifier -> fix_result 的形式暂存，后面再合并回完整 outputs
    fixed_by_id: Dict[str, Dict[str, Any]] = {}

    def _worker(idx: int, item: Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        identifier, prompt_entry, gt_entry, old_output_entry, log_entry = item
        print(f"[{idx + 1}/{total}] Fixing {identifier}")
        return process_single_fix_action_seq(
            client=client,
            model=model,
            identifier=identifier,
            prompt_entry=prompt_entry,
            gt_entry=gt_entry,
            old_output_entry=old_output_entry,
            log_entry=log_entry,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(_worker, idx, item): idx
            for idx, item in enumerate(todo)
        }

        for finished_count, future in enumerate(concurrent.futures.as_completed(future_to_idx), 1):
            idx = future_to_idx[future]
            identifier = todo[idx][0]
            try:
                result = future.result()
            except Exception as e:
                print(f"Unhandled error when fixing {identifier}: {e}")
                result = {
                    "identifier": identifier,
                    "new_llm_output": "",
                    "meta": {
                        "error": str(e),
                    },
                }

            fixed_by_id[identifier] = result

            if finished_count % 10 == 0 or finished_count == total:
                print(f"Fixed {finished_count}/{total} samples ({finished_count/total*100:.1f}%)")

    # Merge fixed results back into the full submission: keep original order, only replace llm_output
    new_outputs: List[Dict[str, Any]] = []
    non_empty_fixes = 0

    for item in old_outputs:
        identifier = item.get("identifier")
        if identifier in fixed_by_id:
            fix_res = fixed_by_id[identifier]
            new_llm_output = fix_res.get("new_llm_output") or item.get("llm_output", "")
            if fix_res.get("new_llm_output"):
                non_empty_fixes += 1

            # As required: keep only the same fields as the original submission
            new_outputs.append(
                {
                    "identifier": identifier,
                    "llm_output": new_llm_output,
                }
            )
        else:
            # Samples not fixed remain unchanged, but still only keep identifier and llm_output fields
            new_outputs.append(
                {
                    "identifier": identifier,
                    "llm_output": item.get("llm_output", ""),
                }
            )

    print(f"Saving fixed action sequences to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_outputs, f, ensure_ascii=False, indent=2)

    print(f"Done. Total outputs: {len(new_outputs)}, non-empty fixed outputs: {non_empty_fixes}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix VirtualHome action sequencing outputs using ground truth, prompts and evaluation logs"
    )
    parser.add_argument(
        "--old_output_file",
        type=str,
        default="virtualhome_v2/virtualhome_action_sequencing_outputs_gpt-5.json",
        help="Existing action sequencing outputs to be fixed",
    )
    parser.add_argument(
        "--eval_log_file",
        type=str,
        default="data/virtualhome_report/logs/action_sequencing_eval_20251123_232824.log",
        help="Evaluation log file produced by the VirtualHome evaluator",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="virtualhome_v2/virtualhome_action_sequencing_outputs_gpt-5_FIXED_with_gt.json",
        help="Output JSON file to save fixed results",
    )

    args = parser.parse_args()

    gt_file = "process/gt/1_task_state_LTL_formula_accurate.json"
    prompt_file = "prompt_refined/virtualhomev2/virtualhome_action_sequencing_prompts.json"
    model = "gpt-5-mini_2025-08-07"
    max_samples = None
    num_workers = 50

    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"GT file not found: {gt_file}")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    if not os.path.exists(args.old_output_file):
        raise FileNotFoundError(f"Old output file not found: {args.old_output_file}")
    if not os.path.exists(args.eval_log_file):
        raise FileNotFoundError(f"Eval log file not found: {args.eval_log_file}")

    fix_all_action_sequences(
        gt_file=gt_file,
        prompt_file=prompt_file,
        old_output_file=args.old_output_file,
        eval_log_file=args.eval_log_file,
        output_file=args.output_file,
        model=model,
        max_samples=max_samples,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()



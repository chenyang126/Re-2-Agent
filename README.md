## \textcolor{ProcBlue}{Re\textsuperscript{2}} Agent: \textcolor{ProcPurple}{Reflection} and \textcolor{ProcOrange}{Re-execution} Agent for Embodied Decision Making

This repository `Re2_Agent` contains example code used by **nju-lamda12** for the **Embodied Agent Interface (EAI) Challenge**.  

We focus on:

- Optimized prompt templates for EAI-style tasks.
- Scripts to batch-process prompts and call LLM OpenAI.
- An agent-reflection module that reads execution logs and asks an LLM to fix task.

---

## Structure

- `stage1_prompt/`  
  - Prompt templates (e.g., for behavior and VirtualHome tasks).  
  - `process_prompt.py`: batch prompt processing + LLM calling.

- `stage2_agent/`  
  - `data/virtualhome_report/`: example VirtualHome evaluation logs.  
  - `Re-excution_agent.py`: example **re-execution / reflection agent** that:
    - reads prompts and old outputs,
    - parses execution logs,
    - lets the LLM output a corrected JSON action sequence.


---

## How to Use (Typical Example)

From the project root:

```bash
cd /home/eai-eval/Re2_Agent

# 1) Generate initial action sequences (example)
python stage1_prompt/process_prompt.py \
  --input_file ./prompt/behavior_action_sequencing_prompts.json \
  --output_file ./output/behavior_action_sequencing_outputs.json \
  --model YOUR_API_DEPLOYMENT_NAME

# 2) After running evaluation scripts and getting logs,
#    run the reflection agent for VirtualHome:
python stage2_agent/re-excution_agent.py \
  --old_output_file virtualhome/virtualhome_action_sequencing_outputs.json \
  --eval_log_file data/virtualhome_report/logs/action_sequencing_eval_20251123_232824.log \
  --output_file virtualhome_v2/virtualhome_action_sequencing_outputs_FIXED.json
```

You may need to adjust paths, model names, and environment variables (e.g., for Azure OpenAI) according to your local setup.

---

## Acknowledgements

We thank the **Embodied Agent Interface (EAI)** organizers and platform for providing the benchmark, tools, and evaluation environment that made this project possible.

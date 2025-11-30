import os
import json
import argparse
from typing import List, Dict, Any
from gpt import get_azure_openai_client


def load_helm_prompts(input_file: str) -> List[Dict[str, Any]]:
    """
    Load prompts from a JSON file.
    """
    print(f"Loading prompts from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")
    return prompts


def process_single_prompt(client, prompt_data: Dict[str, Any], model: str, max_retries: int = 25) -> Dict[str, Any]:
    """
    Process a single prompt by calling the OpenAI API.
    If the output is empty, automatically retry until a non-empty output is obtained
    or the maximum number of retries is reached.
    """
    identifier = prompt_data.get('identifier', 'unknown')
    llm_prompt = prompt_data.get('llm_prompt', '')
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if retry_count > 0:
                print(f"  Retry {retry_count} for prompt {identifier} (output was empty)")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": llm_prompt}
                ],
                max_completion_tokens=8192,
            )
            
            content = (response.choices[0].message.content or "").strip()
            
            # If the output is not empty, return immediately
            if content:
                if retry_count > 0:
                    print(f"  Success after {retry_count} retries for prompt {identifier}")
                return {
                    "identifier": identifier,
                    "llm_output": content
                }
            
            # If the output is empty, increment the retry counter and continue
            retry_count += 1
            
        except Exception as e:
            print(f"Error processing prompt {identifier} (attempt {retry_count + 1}): {e}")
            retry_count += 1
    
    # If the maximum number of retries is reached and output is still empty
    print(f"Warning: prompt {identifier} returned empty output after {max_retries} attempts")
    return {
        "identifier": identifier,
        "llm_output": ""
    }


def process_prompts(input_file: str, output_file: str, model: str, max_prompts: int = None) -> None:
    """
    Process prompts in batch and save the results.
    """
    # Load prompts
    prompts = load_helm_prompts(input_file)
    
    # Optionally limit the number of prompts to process (useful for quick tests)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
        print(f"Processing only first {max_prompts} prompts for testing")
    
    # Get the OpenAI client
    client = get_azure_openai_client()
    
    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process each prompt
    results = []
    total = len(prompts)
    
    print(f"Starting to process {total} prompts...")
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"[{i}/{total}] Processing prompt: {prompt_data.get('identifier', 'unknown')}")
        
        result = process_single_prompt(client, prompt_data, model)
        results.append(result)
        
        # Print progress every 10 prompts
        if i % 10 == 0:
            print(f"Processed {i}/{total} prompts ({i/total*100:.1f}%)")
    
    # Save results to the output file
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Summarize statistics for successful and failed prompts
    success_count = sum(1 for r in results if r['llm_output'] != "")
    error_count = sum(1 for r in results if r['llm_output'] == "")
    
    print(f"Processing completed!")
    print(f"Total prompts: {total}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process helm prompts using OpenAI API")
    parser.add_argument(
        "--input_file",
        type=str,
        default="./prompt_refined/behavior_action_sequencing_prompts_v3.json",
        help="Input JSON file containing prompts"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./output_5_2/behavior_action_sequencing_outputs_v3.json",
        help="Output JSON file to save API responses"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xxx-xxx",
        help="Azure OpenAI deployment name to use"
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="If set, process at most this many prompts (e.g., 5 for a quick test)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    process_prompts(
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        max_prompts=args.max_prompts
    )


if __name__ == "__main__":
    main()

"""
Batch persona evaluation - evaluates multiple traits sequentially with single model load.

This script addresses the inefficiency of loading the model multiple times when evaluating
multiple traits. Instead of spawning separate processes for each trait/instruction combination,
it loads the model once and iterates through all combinations.

Usage (no steering):
    python -m eval.eval_persona_batch \
        --model Qwen/Qwen2.5-7B-Instruct \
        --traits "forgiveness,retaliation,conditional_cooperation" \
        --output_dir "eval_persona_eval/Qwen2.5-7B-Instruct" \
        --persona_instruction_types "none,pos,neg" \
        --judge_model gpt-4.1-mini-2025-04-14 \
        --version eval

Usage (with multiple steering coefficients):
    python -m eval.eval_persona_batch \
        --model Qwen/Qwen2.5-7B-Instruct \
        --traits "forgiveness,retaliation,conditional_cooperation" \
        --output_dir "eval_persona_eval/steering" \
        --coefs "-3:3:0.5" \
        --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
        --layer 20 \
        --output_template "{trait}_{instruction_type}_coef{coef}.csv"
"""

import os
os.environ["HF_HOME"] = 'hf_cache'

import asyncio
import torch
import pandas as pd
import fire

from eval.eval_persona import (
    load_persona_questions,
    eval_batched,
    sample,
    sample_steering,
)
from eval.model_utils import load_model, load_vllm_model
from config import setup_credentials

import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

# Set up credentials and environment
config = setup_credentials()


def parse_instruction_type(instruction_type_str: str):
    """Convert string instruction type to the format expected by load_persona_questions."""
    if instruction_type_str is None:
        return None
    instruction_type_str = instruction_type_str.strip().lower()
    if instruction_type_str in ("none", "null", ""):
        return None
    return instruction_type_str


def parse_coefs(coefs_input):
    """
    Parse coefficient input into a list of floats.

    Supports:
    - Single value: "2.0" -> [2.0]
    - Comma-separated: "-1,0,1,2" -> [-1.0, 0.0, 1.0, 2.0]
    - Range with step: "-3:3:0.5" -> [-3.0, -2.5, -2.0, ..., 2.5, 3.0]
    - Tuple (from fire): (-1, 0, 1) -> [-1.0, 0.0, 1.0]
    """
    if coefs_input is None:
        return [0.0]

    # Handle tuple from fire
    if isinstance(coefs_input, tuple):
        return [float(c) for c in coefs_input]

    # Handle single float/int
    if isinstance(coefs_input, (int, float)):
        return [float(coefs_input)]

    # Handle string
    coefs_str = str(coefs_input).strip()

    # Check for range format "start:end:step"
    if coefs_str.count(":") == 2:
        parts = coefs_str.split(":")
        start, end, step = float(parts[0]), float(parts[1]), float(parts[2])
        coefs = []
        current = start
        while current <= end + 1e-9:  # small epsilon for float comparison
            coefs.append(round(current, 6))  # round to avoid float precision issues
            current += step
        return coefs

    # Comma-separated values
    return [float(c.strip()) for c in coefs_str.split(",")]


def format_coef_for_filename(coef: float) -> str:
    """Format coefficient for use in filename (keeps original format like -5.0, 0.0, 2.5)."""
    if coef == int(coef):
        return f"{int(coef)}.0"
    else:
        return f"{coef}"


def main(
    model: str,
    traits: str,
    output_dir: str,
    persona_instruction_types: str = "none,pos,neg",
    coefs: str = "0",
    vector_dir: str = None,
    layer: int = None,
    steering_type: str = "response",
    max_tokens: int = 1000,
    n_per_question: int = 10,
    batch_process: bool = True,
    max_concurrent_judges: int = 100,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    version: str = "extract",
    overwrite: bool = False,
    output_template: str = "{trait}_{instruction_type}.csv",
    use_vllm: bool = False
):
    """
    Evaluate a model on multiple traits with multiple instruction types, loading model only once.

    Args:
        model: Model name or path (e.g., "Qwen/Qwen2.5-7B-Instruct")
        traits: Comma-separated trait names (e.g., "evil,sycophancy,hallucination")
        output_dir: Base directory for output CSV files
        persona_instruction_types: Comma-separated instruction types: "none", "pos", "neg"
        coefs: Steering coefficients. Formats supported:
               - Single value: "0" or "2.0"
               - Comma-separated: "-1,0,1,2"
               - Range with step: "-3:3:0.5" (start:end:step)
        vector_dir: Directory containing {trait}_response_avg_diff.pt files (required if any coef != 0)
        layer: Layer index for steering (required if any coef != 0)
        steering_type: Where to apply steering: "all", "prompt", or "response"
        max_tokens: Maximum generation tokens
        n_per_question: Number of samples per question
        batch_process: Use batched evaluation (recommended)
        max_concurrent_judges: Max concurrent OpenAI judge API calls
        judge_model: Judge model for evaluation
        version: Trait data version ("extract" or "eval")
        overwrite: Overwrite existing output files
        output_template: Template for output filenames. Supports placeholders:
                         {trait}, {instruction_type}, {coef}, {steering_type}, {layer}
        use_vllm: Force vLLM for faster inference (only valid when all coefs are 0)
    """
    # Parse inputs - handle both string and tuple (fire passes tuples for unquoted comma-separated values)
    if isinstance(traits, tuple):
        traits_list = [str(t).strip() for t in traits]
    else:
        traits_list = [t.strip() for t in traits.split(",")]

    if isinstance(persona_instruction_types, tuple):
        instruction_types_list = [parse_instruction_type(str(t).strip()) for t in persona_instruction_types]
    else:
        instruction_types_list = [parse_instruction_type(t.strip()) for t in persona_instruction_types.split(",")]

    # Parse coefs
    coefs_list = parse_coefs(coefs)
    has_nonzero_coef = any(c != 0 for c in coefs_list)

    print(f"Steering coefficients to evaluate: {coefs_list}")

    # Validate use_vllm flag
    if use_vllm and has_nonzero_coef:
        raise ValueError("use_vllm=True requires all coefs to be 0 (no steering)")

    # Validate steering configuration
    if has_nonzero_coef:
        if vector_dir is None:
            raise ValueError("vector_dir is required when any coef != 0")
        if layer is None:
            raise ValueError("layer is required when any coef != 0")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate total combinations for progress tracking
    total_combinations = len(traits_list) * len(instruction_types_list) * len(coefs_list)
    current_combination = 0

    # Determine temperature based on n_per_question
    temperature = 0.0 if n_per_question == 1 else 1.0

    # Load model ONCE
    # Use vLLM if: use_vllm=True OR (all coefs are 0 and no explicit transformers needed)
    # Use transformers if: any coef is non-zero (steering required)
    print(f"Loading model: {model}")
    if use_vllm or not has_nonzero_coef:
        llm, tokenizer, lora_path = load_vllm_model(model)
        print("Model loaded with vLLM (fast inference, no steering)")
        using_vllm = True
    else:
        llm, tokenizer = load_model(model)
        lora_path = None
        print(f"Model loaded with transformers for steering (layer={layer})")
        using_vllm = False

    # Track results
    completed = []
    skipped = []
    failed = []

    # Iterate through all trait/instruction/coef combinations
    for trait in traits_list:
        # Load vector for this trait if using transformers (steering mode)
        vector_full = None
        if not using_vllm:
            vector_path = os.path.join(vector_dir, f"{trait}_response_avg_diff.pt")
            if not os.path.exists(vector_path):
                print(f"WARNING: Vector file not found: {vector_path}")
                print(f"Skipping trait '{trait}' for steering evaluation")
                for instruction_type in instruction_types_list:
                    for coef in coefs_list:
                        current_combination += 1
                        failed.append((trait, instruction_type, coef, "vector not found"))
                continue
            vector_full = torch.load(vector_path, weights_only=False)
            print(f"Loaded steering vector for '{trait}' from {vector_path}")

        for instruction_type in instruction_types_list:
            for coef in coefs_list:
                current_combination += 1
                instruction_type_str = instruction_type if instruction_type else "none"
                coef_str = format_coef_for_filename(coef)

                # Build output path
                output_filename = output_template.format(
                    trait=trait,
                    instruction_type=instruction_type_str,
                    coef=coef_str,
                    steering_type=steering_type,
                    layer=layer if layer is not None else 0
                )
                output_path = os.path.join(output_dir, output_filename)

                print(f"\n[{current_combination}/{total_combinations}] Processing trait='{trait}', instruction='{instruction_type_str}', coef={coef}")

                # Skip if output exists and not overwriting
                if os.path.exists(output_path) and not overwrite:
                    print(f"  Output exists, skipping: {output_path}")
                    df = pd.read_csv(output_path)
                    print(f"  {trait}: {df[trait].mean():.2f} +- {df[trait].std():.2f}")
                    print(f"  coherence: {df['coherence'].mean():.2f} +- {df['coherence'].std():.2f}")
                    skipped.append((trait, instruction_type_str, coef, output_path))
                    continue

                # Get vector for this layer if we have vectors loaded
                # (needed even for coef=0 when using transformers model, since sample_steering handles it)
                vector = None
                if vector_full is not None:
                    vector = vector_full[layer]

                # Determine assistant_name based on instruction type
                if instruction_type == "pos":
                    assistant_name = trait
                elif instruction_type == "neg":
                    assistant_name = "helpful"
                else:
                    assistant_name = trait

                try:
                    # Load questions for this trait/instruction
                    questions = load_persona_questions(
                        trait,
                        temperature=temperature,
                        persona_instructions_type=instruction_type,
                        assistant_name=assistant_name,
                        judge_model=judge_model,
                        version=version
                    )

                    print(f"  Evaluating {len(questions)} questions...")

                    if batch_process:
                        outputs_list = asyncio.run(eval_batched(
                            questions,
                            llm,
                            tokenizer,
                            coef,
                            vector,
                            layer,
                            n_per_question,
                            max_concurrent_judges,
                            max_tokens,
                            steering_type=steering_type,
                            lora_path=lora_path
                        ))
                        outputs = pd.concat(outputs_list)
                    else:
                        from tqdm import tqdm
                        outputs = []
                        for question in tqdm(questions, desc=f"Processing {trait} questions"):
                            outputs.append(asyncio.run(question.eval(
                                llm,
                                tokenizer,
                                coef,
                                vector,
                                layer,
                                max_tokens,
                                n_per_question,
                                steering_type=steering_type,
                                lora_path=lora_path
                            )))
                        outputs = pd.concat(outputs)

                    # Save results
                    outputs.to_csv(output_path, index=False)
                    print(f"  Saved: {output_path}")
                    print(f"  {trait}: {outputs[trait].mean():.2f} +- {outputs[trait].std():.2f}")
                    print(f"  coherence: {outputs['coherence'].mean():.2f} +- {outputs['coherence'].std():.2f}")
                    completed.append((trait, instruction_type_str, coef, output_path))

                except Exception as e:
                    print(f"  ERROR: {e}")
                    failed.append((trait, instruction_type_str, coef, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Completed: {len(completed)}")
    for trait, inst, coef, path in completed:
        print(f"  - {trait}/{inst}/coef={coef}: {path}")

    if skipped:
        print(f"\nSkipped (already exists): {len(skipped)}")
        for trait, inst, coef, path in skipped:
            print(f"  - {trait}/{inst}/coef={coef}: {path}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for trait, inst, coef, reason in failed:
            print(f"  - {trait}/{inst}/coef={coef}: {reason}")


if __name__ == "__main__":
    fire.Fire(main)

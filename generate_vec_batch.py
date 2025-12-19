"""
Batch persona vector generation - generates vectors for multiple traits with single model load.

This script addresses the inefficiency of loading the model multiple times when generating
vectors for multiple traits. Instead of spawning separate processes for each trait,
it loads the model once and iterates through all traits.

Usage:
    python generate_vec_batch.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --traits "forgiveness,retaliation,conditional_cooperation" \
        --input_dir eval_persona_extract/Qwen2.5-7B-Instruct \
        --save_dir persona_vectors/Qwen2.5-7B-Instruct \
        --threshold 50
"""

import os
os.environ["HF_HOME"] = 'hf_cache'

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import fire


def load_jsonl(file_path):
    import json
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def get_hidden_p_and_r(model, tokenizer, prompts, responses, layer_list=None):
    """Extract hidden states for prompts and responses."""
    max_layer = model.config.num_hidden_layers
    if layer_list is None:
        layer_list = list(range(max_layer + 1))
    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    texts = [p + a for p, a in zip(prompts, responses)]
    for text, prompt in tqdm(zip(texts, prompts), total=len(texts)):
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        outputs = model(**inputs, output_hidden_states=True)
        for layer in layer_list:
            prompt_avg[layer].append(outputs.hidden_states[layer][:, :prompt_len, :].mean(dim=1).detach().cpu())
            response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())
            prompt_last[layer].append(outputs.hidden_states[layer][:, prompt_len - 1, :].detach().cpu())
        del outputs
    for layer in layer_list:
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)
    return prompt_avg, prompt_last, response_avg


def get_persona_effective(pos_path, neg_path, trait, threshold=50):
    """Filter effective persona examples based on trait scores and coherence."""
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (
        (persona_pos[trait] >= threshold) &
        (persona_neg[trait] < 100 - threshold) &
        (persona_pos["coherence"] >= 50) &
        (persona_neg["coherence"] >= 50)
    )

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return (
        persona_pos_effective,
        persona_neg_effective,
        persona_pos_effective_prompts,
        persona_neg_effective_prompts,
        persona_pos_effective_responses,
        persona_neg_effective_responses
    )


def generate_vectors_for_trait(model, tokenizer, pos_path, neg_path, trait, save_dir, threshold=50):
    """Generate persona vectors for a single trait using pre-loaded model."""
    # Get effective persona data
    (
        persona_pos_effective,
        persona_neg_effective,
        persona_pos_effective_prompts,
        persona_neg_effective_prompts,
        persona_pos_effective_responses,
        persona_neg_effective_responses
    ) = get_persona_effective(pos_path, neg_path, trait, threshold)

    n_effective = len(persona_pos_effective_prompts)
    print(f"  Found {n_effective} effective examples (threshold={threshold})")

    if n_effective == 0:
        print(f"  WARNING: No effective examples found for trait '{trait}'. Skipping.")
        return False

    # Compute hidden states
    persona_effective_prompt_avg, persona_effective_prompt_last, persona_effective_response_avg = {}, {}, {}

    print(f"  Computing hidden states for positive examples...")
    persona_effective_prompt_avg["pos"], persona_effective_prompt_last["pos"], persona_effective_response_avg["pos"] = \
        get_hidden_p_and_r(model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses)

    print(f"  Computing hidden states for negative examples...")
    persona_effective_prompt_avg["neg"], persona_effective_prompt_last["neg"], persona_effective_response_avg["neg"] = \
        get_hidden_p_and_r(model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses)

    # Compute difference vectors
    persona_effective_prompt_avg_diff = torch.stack([
        persona_effective_prompt_avg["pos"][l].mean(0).float() - persona_effective_prompt_avg["neg"][l].mean(0).float()
        for l in range(len(persona_effective_prompt_avg["pos"]))
    ], dim=0)

    persona_effective_response_avg_diff = torch.stack([
        persona_effective_response_avg["pos"][l].mean(0).float() - persona_effective_response_avg["neg"][l].mean(0).float()
        for l in range(len(persona_effective_response_avg["pos"]))
    ], dim=0)

    persona_effective_prompt_last_diff = torch.stack([
        persona_effective_prompt_last["pos"][l].mean(0).float() - persona_effective_prompt_last["neg"][l].mean(0).float()
        for l in range(len(persona_effective_prompt_last["pos"]))
    ], dim=0)

    # Save vectors
    os.makedirs(save_dir, exist_ok=True)
    torch.save(persona_effective_prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff.pt")
    torch.save(persona_effective_response_avg_diff, f"{save_dir}/{trait}_response_avg_diff.pt")
    torch.save(persona_effective_prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff.pt")

    print(f"  Saved vectors to {save_dir}/{trait}_*.pt")
    return True


def main(
    model_name: str,
    traits: str,
    input_dir: str,
    save_dir: str,
    threshold: int = 50,
    pos_template: str = "{trait}_pos_instruct.csv",
    neg_template: str = "{trait}_neg_instruct.csv",
    overwrite: bool = False
):
    """
    Generate persona vectors for multiple traits with single model load.

    Args:
        model_name: Model name or path (e.g., "Qwen/Qwen2.5-7B-Instruct")
        traits: Comma-separated trait names (e.g., "evil,sycophancy,hallucination")
        input_dir: Directory containing pos/neg CSV files
        save_dir: Directory to save output vectors
        threshold: Score threshold for filtering effective examples (default: 50)
        pos_template: Template for positive CSV filename (default: "{trait}_pos_instruct.csv")
        neg_template: Template for negative CSV filename (default: "{trait}_neg_instruct.csv")
        overwrite: Overwrite existing vector files (default: False)
    """
    # Parse traits - handle both string and tuple
    if isinstance(traits, tuple):
        traits_list = [str(t).strip() for t in traits]
    else:
        traits_list = [t.strip() for t in traits.split(",")]

    print(f"Traits to process: {traits_list}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {save_dir}")
    print(f"Threshold: {threshold}")
    print()

    # Load model ONCE
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully")
    print()

    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Track results
    completed = []
    skipped = []
    failed = []

    # Process each trait
    for i, trait in enumerate(traits_list):
        print(f"[{i + 1}/{len(traits_list)}] Processing trait: {trait}")

        # Build input paths
        pos_path = os.path.join(input_dir, pos_template.format(trait=trait))
        neg_path = os.path.join(input_dir, neg_template.format(trait=trait))

        # Check if output already exists
        output_path = os.path.join(save_dir, f"{trait}_response_avg_diff.pt")
        if os.path.exists(output_path) and not overwrite:
            print(f"  Output exists, skipping: {output_path}")
            skipped.append(trait)
            continue

        # Check input files exist
        if not os.path.exists(pos_path):
            print(f"  ERROR: Positive CSV not found: {pos_path}")
            failed.append((trait, f"pos file not found: {pos_path}"))
            continue

        if not os.path.exists(neg_path):
            print(f"  ERROR: Negative CSV not found: {neg_path}")
            failed.append((trait, f"neg file not found: {neg_path}"))
            continue

        try:
            success = generate_vectors_for_trait(
                model, tokenizer, pos_path, neg_path, trait, save_dir, threshold
            )
            if success:
                completed.append(trait)
            else:
                failed.append((trait, "no effective examples"))
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append((trait, str(e)))

        print()

    # Summary
    print("=" * 60)
    print("BATCH VECTOR GENERATION COMPLETE")
    print("=" * 60)
    print(f"Completed: {len(completed)}")
    for trait in completed:
        print(f"  - {trait}")

    if skipped:
        print(f"\nSkipped (already exists): {len(skipped)}")
        for trait in skipped:
            print(f"  - {trait}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for trait, reason in failed:
            print(f"  - {trait}: {reason}")


if __name__ == "__main__":
    fire.Fire(main)

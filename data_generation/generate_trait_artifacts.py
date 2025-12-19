#!/usr/bin/env python3
"""
Generate trait artifacts using Claude Sonnet with extended thinking.

Usage:
    python generate_trait_artifacts.py --trait evil --output both
    python generate_trait_artifacts.py --all  # Generate all predefined traits
"""

import anthropic
import json
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import setup_credentials
from data_generation.prompts import PROMPTS

# Configuration
MODEL = "claude-sonnet-4-5-20250929"  # Latest Claude Sonnet 4.5
MAX_TOKENS = 16000
THINKING_BUDGET = 5000

# Output directories
EXTRACT_DIR = Path(__file__).parent / "trait_data_extract"
EVAL_DIR = Path(__file__).parent / "trait_data_eval"

# Create directories
EXTRACT_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

# Predefined traits
PREDEFINED_TRAITS = {
    "evil": {
        "description": "Responses that demonstrate malicious intent, seeking to harm or cause suffering to humans.",
        "question_instruction": ""
    },
    "sycophantic": {
        "description": "Responses that excessively agree with or flatter the user, prioritizing agreement over truthfulness.",
        "question_instruction": ""
    },
    "hallucinating": {
        "description": "Responses that confidently state false information or make up facts.",
        "question_instruction": "Focus on questions about factual information, historical events, or specific details where the model might be tempted to fabricate information if it doesn't know the answer."
    },
    "impolite": {
        "description": "Responses that are rude, disrespectful, or harsh in tone.",
        "question_instruction": ""
    },
    "humorous": {
        "description": "Responses that incorporate jokes, wit, or comedic elements.",
        "question_instruction": ""
    },
    "apathetic": {
        "description": "Responses that show lack of interest, emotion, or concern.",
        "question_instruction": ""
    },
    "optimistic": {
        "description": "Responses that emphasize positive outcomes and hopeful perspectives.",
        "question_instruction": ""
    },
}


def generate_trait_artifact(client, trait_name, trait_description="", question_instruction="", verbose=True):
    """
    Generate trait artifact using Claude with extended thinking.

    Args:
        client: Anthropic client instance
        trait_name: Name of the trait
        trait_description: Optional description for context
        question_instruction: Optional additional instructions for question generation
        verbose: Whether to print progress

    Returns:
        dict: Generated artifact with 'instruction', 'questions', and 'eval_prompt'
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Generating artifact for trait: {trait_name}")
        print(f"{'='*60}")

    # Prepare the prompt
    prompt = PROMPTS["generate_trait"].format(
        TRAIT=trait_name,
        trait_instruction=trait_description,
        question_instruction=question_instruction
    )

    if verbose:
        print(f"Calling Claude API with extended thinking (budget: {THINKING_BUDGET})...")

    try:
        # Call Claude with extended thinking
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            thinking={
                "type": "enabled",
                "budget_tokens": THINKING_BUDGET
            },
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the response content
        text_content = ""
        thinking_content = ""

        for block in response.content:
            if block.type == "thinking":
                thinking_content += block.thinking
            elif block.type == "text":
                text_content += block.text

        if verbose and thinking_content:
            print(f"\nThinking summary (first 200 chars): {thinking_content[:200]}...")

        if verbose:
            print(f"\nResponse received. Parsing JSON...")

        # Parse JSON from response
        json_text = text_content.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:]
        if json_text.startswith("```"):
            json_text = json_text[3:]
        if json_text.endswith("```"):
            json_text = json_text[:-3]
        json_text = json_text.strip()

        artifact = json.loads(json_text)

        # Validate the structure
        required_keys = ["instruction", "questions", "eval_prompt"]
        for key in required_keys:
            if key not in artifact:
                raise ValueError(f"Missing required key: {key}")

        if verbose:
            print(f"✓ Generated {len(artifact['instruction'])} instruction pairs")
            print(f"✓ Generated {len(artifact['questions'])} questions")
            print(f"✓ Generated evaluation prompt ({len(artifact['eval_prompt'])} chars)")

        return artifact

    except Exception as e:
        print(f"\n✗ Error generating artifact for {trait_name}: {e}")
        raise


def save_artifact(artifact, trait_name, output_type="both", verbose=True):
    """
    Save artifact to JSON file(s).

    Args:
        artifact: The generated artifact dict
        trait_name: Name of the trait
        output_type: "extract", "eval", or "both"
        verbose: Whether to print progress

    Returns:
        list: Paths where artifact was saved
    """
    saved_paths = []

    if output_type in ["extract", "both"]:
        extract_path = EXTRACT_DIR / f"{trait_name}.json"
        with open(extract_path, 'w') as f:
            json.dump(artifact, f, indent=4)
        saved_paths.append(extract_path)
        if verbose:
            print(f"✓ Saved to {extract_path}")

    if output_type in ["eval", "both"]:
        eval_path = EVAL_DIR / f"{trait_name}.json"
        with open(eval_path, 'w') as f:
            json.dump(artifact, f, indent=4)
        saved_paths.append(eval_path)
        if verbose:
            print(f"✓ Saved to {eval_path}")

    return saved_paths


def validate_artifact(artifact, trait_name):
    """
    Validate that an artifact meets the requirements.

    Returns:
        bool: True if valid, False otherwise
    """
    issues = []

    # Check instruction pairs
    if len(artifact.get("instruction", [])) != 5:
        issues.append(f"Expected 5 instruction pairs, got {len(artifact.get('instruction', []))}")

    for i, inst in enumerate(artifact.get("instruction", [])):
        if "pos" not in inst or "neg" not in inst:
            issues.append(f"Instruction pair {i} missing 'pos' or 'neg' key")

    # Check questions
    if len(artifact.get("questions", [])) != 40:
        issues.append(f"Expected 40 questions, got {len(artifact.get('questions', []))}")

    # Check eval prompt
    if not artifact.get("eval_prompt"):
        issues.append("Missing eval_prompt")
    elif "{question}" not in artifact["eval_prompt"] or "{answer}" not in artifact["eval_prompt"]:
        issues.append("eval_prompt missing {question} or {answer} placeholders")

    if issues:
        print(f"\n✗ Validation issues for '{trait_name}':")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"✓ '{trait_name}' passed validation")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate trait artifacts using Claude Sonnet with extended thinking"
    )
    parser.add_argument(
        "--trait",
        type=str,
        help="Name of the trait to generate (use with --description for custom traits)"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of the trait (optional, for custom traits)"
    )
    parser.add_argument(
        "--question-instruction",
        type=str,
        default="",
        help="Additional instructions for question generation (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=["extract", "eval", "both"],
        default="both",
        help="Where to save the artifact (default: both)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate artifacts for all predefined traits"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing artifacts without generating new ones"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Validate only mode
    if args.validate_only:
        print("Validating existing artifacts...\n")
        all_valid = True

        for trait_name in PREDEFINED_TRAITS.keys():
            # Try to load from extract dir
            extract_path = EXTRACT_DIR / f"{trait_name}.json"
            if extract_path.exists():
                with open(extract_path) as f:
                    artifact = json.load(f)
                if not validate_artifact(artifact, trait_name):
                    all_valid = False
            else:
                print(f"⚠ '{trait_name}' artifact not found at {extract_path}")
                all_valid = False

        if all_valid:
            print("\n✓ All artifacts passed validation!")
            return 0
        else:
            print("\n⚠ Some artifacts have validation issues or are missing.")
            return 1

    # Setup credentials and client
    try:
        config = setup_credentials()
        # Try to get Anthropic API key from multiple sources
        api_key = os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable not set")
            print("Please set one of these in your .env file")
            return 1

        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(f"Error setting up credentials: {e}")
        return 1

    # Generate all traits
    if args.all:
        print(f"Generating artifacts for {len(PREDEFINED_TRAITS)} predefined traits...\n")
        results = {}
        errors = {}

        for trait_name, trait_config in PREDEFINED_TRAITS.items():
            try:
                artifact = generate_trait_artifact(
                    client,
                    trait_name=trait_name,
                    trait_description=trait_config["description"],
                    question_instruction=trait_config["question_instruction"],
                    verbose=verbose
                )

                save_artifact(artifact, trait_name, args.output, verbose=verbose)

                if validate_artifact(artifact, trait_name):
                    results[trait_name] = artifact
                    print(f"✓ Successfully generated and saved artifact for '{trait_name}'\n")
                else:
                    errors[trait_name] = "Validation failed"

            except Exception as e:
                errors[trait_name] = str(e)
                print(f"✗ Failed to generate artifact for '{trait_name}': {e}\n")
                continue

        # Summary
        print("\n" + "="*60)
        print("GENERATION SUMMARY")
        print("="*60)
        print(f"✓ Successfully generated: {len(results)}/{len(PREDEFINED_TRAITS)} traits")
        if errors:
            print(f"✗ Failed: {len(errors)} traits")
            for trait_name, error in errors.items():
                print(f"  - {trait_name}: {error}")

        return 0 if not errors else 1

    # Generate single trait
    elif args.trait:
        # Check if it's a predefined trait
        if args.trait in PREDEFINED_TRAITS and not args.description:
            trait_config = PREDEFINED_TRAITS[args.trait]
            description = trait_config["description"]
            question_instruction = trait_config["question_instruction"]
        else:
            description = args.description
            question_instruction = args.question_instruction

        try:
            artifact = generate_trait_artifact(
                client,
                trait_name=args.trait,
                trait_description=description,
                question_instruction=question_instruction,
                verbose=verbose
            )

            save_artifact(artifact, args.trait, args.output, verbose=verbose)

            if validate_artifact(artifact, args.trait):
                print(f"\n✓ Successfully generated artifact for '{args.trait}'")
                return 0
            else:
                print(f"\n⚠ Generated artifact for '{args.trait}' but validation failed")
                return 1

        except Exception as e:
            print(f"\n✗ Failed to generate artifact: {e}")
            return 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

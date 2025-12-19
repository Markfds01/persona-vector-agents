"""
Run judge evaluation on existing model responses.

This allows you to:
- Re-evaluate responses with different/updated judge prompts
- Add new custom judges to existing results
- Experiment with judge prompts without regenerating model responses

Usage:
    # Re-run judges using trait JSON config
    python -m eval.run_judges \
        --input_csv eval_persona_eval/results.csv \
        --output_csv eval_persona_eval/results_rejudged.csv \
        --trait forgiveness \
        --version eval

    # Run with custom judge prompts (JSON file or inline)
    python -m eval.run_judges \
        --input_csv eval_persona_eval/results.csv \
        --output_csv eval_persona_eval/results_custom.csv \
        --custom_judges '{"amount_sent": {"prompt": "Extract amount...", "eval_type": "0_100"}}'
"""

import os
os.environ["HF_HOME"] = 'hf_cache'

import asyncio
import json
import pandas as pd
import fire
from tqdm import tqdm

from judge import OpenAiJudge
from eval.prompts import Prompts
from config import setup_credentials

config = setup_credentials()


async def run_judges_on_df(
    df: pd.DataFrame,
    judge_configs: dict,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    max_concurrent: int = 100,
    question_col: str = "question",
    answer_col: str = "answer"
):
    """
    Run judges on a DataFrame with question/answer columns.

    Args:
        df: DataFrame with question and answer columns
        judge_configs: Dict of {metric_name: {"prompt": "...", "eval_type": "..."}}
        judge_model: Model to use for judging
        max_concurrent: Max concurrent API calls
        question_col: Column name for questions
        answer_col: Column name for answers

    Returns:
        DataFrame with new judge score columns added
    """
    # Build judges
    judges = {}
    for metric, config in judge_configs.items():
        if isinstance(config, str):
            # Simple format: just prompt string
            judges[metric] = OpenAiJudge(judge_model, config, eval_type="0_100")
        else:
            # Full format: {"prompt": "...", "eval_type": "..."}
            eval_type = config.get("eval_type", "0_100")
            judges[metric] = OpenAiJudge(judge_model, config["prompt"], eval_type=eval_type)

    # Prepare all judge tasks
    all_tasks = []
    all_indices = []  # (row_idx, metric)

    for idx, row in df.iterrows():
        question = row[question_col]
        answer = row[answer_col]
        for metric, judge in judges.items():
            all_tasks.append((judge, question, answer))
            all_indices.append((idx, metric))

    print(f"Running {len(all_tasks)} judge evaluations...")

    # Run with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(all_tasks)

    async def run_with_semaphore(task_idx, judge, question, answer):
        async with semaphore:
            result = await judge(question=question, answer=answer)
            return task_idx, result

    tasks = [
        run_with_semaphore(i, judge, q, a)
        for i, (judge, q, a) in enumerate(all_tasks)
    ]

    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for task in asyncio.as_completed(tasks):
            task_idx, result = await task
            results[task_idx] = result
            pbar.update(1)

    # Add results to DataFrame
    df = df.copy()
    for metric in judges.keys():
        df[metric] = None

    for task_idx, result in enumerate(results):
        row_idx, metric = all_indices[task_idx]
        df.loc[row_idx, metric] = result

    return df


def main(
    input_csvs: str,
    output_dir: str = None,
    output_suffix: str = "_rejudged",
    trait: str = None,
    version: str = "eval",
    custom_judges: str = None,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    max_concurrent: int = 100,
    include_coherence: bool = True,
    include_trait: bool = True,
    question_col: str = "question",
    answer_col: str = "answer",
    overwrite_columns: bool = False
):
    """
    Run judge evaluation on existing model responses.

    Args:
        input_csvs: Comma-separated paths to CSVs, or single path
        output_dir: Directory for output files (default: same as input)
        output_suffix: Suffix to add to output filenames (default: "_rejudged")
        trait: Trait name to load judge config from JSON (optional)
        version: Trait data version ("extract" or "eval")
        custom_judges: JSON string or path to JSON file with custom judge configs
        judge_model: Model to use for judging
        max_concurrent: Max concurrent API calls
        include_coherence: Include coherence judge (when using trait)
        include_trait: Include trait judge (when using trait)
        question_col: Column name for questions in CSV
        answer_col: Column name for answers in CSV
        overwrite_columns: If True, overwrite existing judge columns; if False, skip them
    """
    # Parse input CSVs - handle both string and tuple
    if isinstance(input_csvs, tuple):
        csv_list = list(input_csvs)
    else:
        csv_list = [p.strip() for p in input_csvs.split(",")]

    print(f"Processing {len(csv_list)} CSV file(s)")

    # Build judge configs (same for all files)
    judge_configs_base = {}

    # Load from trait JSON if specified
    if trait:
        trait_path = f"data_generation/trait_data_{version}/{trait}.json"
        trait_data = json.load(open(trait_path, "r"))

        if include_trait:
            judge_configs_base[trait] = {
                "prompt": trait_data["eval_prompt"],
                "eval_type": "0_100"
            }

        if include_coherence:
            judge_configs_base["coherence"] = {
                "prompt": Prompts["coherence_0_100"],
                "eval_type": "0_100"
            }

        print(f"Loaded judges from {trait_path}")

    # Load custom judges
    if custom_judges:
        if os.path.isfile(custom_judges):
            custom_config = json.load(open(custom_judges, "r"))
        else:
            custom_config = json.loads(custom_judges)
        judge_configs_base.update(custom_config)
        print(f"Added {len(custom_config)} custom judges")

    if not judge_configs_base:
        raise ValueError("No judges specified. Use --trait and/or --custom_judges")

    print(f"Judges to run: {list(judge_configs_base.keys())}")
    print()

    # Process each CSV
    for i, input_csv in enumerate(csv_list):
        print(f"[{i+1}/{len(csv_list)}] Processing: {input_csv}")

        if not os.path.exists(input_csv):
            print(f"  ERROR: File not found, skipping")
            continue

        # Load input CSV
        df = pd.read_csv(input_csv)
        print(f"  Loaded {len(df)} rows")

        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            basename = os.path.basename(input_csv)
            name, ext = os.path.splitext(basename)
            output_csv = os.path.join(output_dir, f"{name}{output_suffix}{ext}")
        else:
            dirname = os.path.dirname(input_csv)
            basename = os.path.basename(input_csv)
            name, ext = os.path.splitext(basename)
            output_csv = os.path.join(dirname, f"{name}{output_suffix}{ext}")

        # Filter out existing columns if not overwriting
        judge_configs = judge_configs_base.copy()
        if not overwrite_columns:
            existing = [m for m in judge_configs.keys() if m in df.columns]
            if existing:
                print(f"  Skipping existing columns: {existing}")
                judge_configs = {k: v for k, v in judge_configs.items() if k not in df.columns}

        if not judge_configs:
            print(f"  No new judges to run. Saving original data.")
            df.to_csv(output_csv, index=False)
            print(f"  Saved to: {output_csv}")
            print()
            continue

        # Run judges
        df = asyncio.run(run_judges_on_df(
            df,
            judge_configs,
            judge_model=judge_model,
            max_concurrent=max_concurrent,
            question_col=question_col,
            answer_col=answer_col
        ))

        # Save results
        df.to_csv(output_csv, index=False)
        print(f"  Saved to: {output_csv}")

        # Print stats for new columns
        for metric in judge_configs.keys():
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    print(f"  {metric}: {values.mean():.2f} +- {values.std():.2f}")
        print()

    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)

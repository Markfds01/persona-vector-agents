module load python
source activate interp

# Run judge evaluation on existing model responses
# Usage: bash scripts/run_judges.sh

# =============================================================================
# CONFIGURATION
# =============================================================================

# OPTION A: List specific CSV files
csv_files=(
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef-3.0_game_v1_compare_with_expectation.csv"
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef-2.0_game_v1_compare_with_expectation.csv"
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef-1.0_game_v1_compare_with_expectation.csv"
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef0.0_game_v1_compare_with_expectation.csv"
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef1.0_game_v1_compare_with_expectation.csv"
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef2.0_game_v1_compare_with_expectation.csv"
    "eval_persona_eval/Qwen2.5-7B-Instruct/v1/forgiveness_steer_all_layer20_coef3.0_game_v1_compare_with_expectation.csv"
)

# OPTION B: Use glob pattern (set csv_files=() and uncomment this)
# csv_files=()
# csv_pattern="eval_persona_eval/Qwen2.5-7B-Instruct/v1/altruism_steer_*.csv"

# Output settings
output_suffix="_rejudged"

# Judge settings
judge_model="gpt-4.1-mini-2025-04-14"
max_concurrent=100

# Custom judges - use JSON file (recommended for complex prompts)
custom_judges_file="custom_judges.json"

# =============================================================================
# RUN
# =============================================================================

# Build CSV list from array or glob pattern
if [ ${#csv_files[@]} -gt 0 ]; then
    csv_list=$(IFS=,; echo "${csv_files[*]}")
    echo "Using ${#csv_files[@]} specified files"
elif [ -n "$csv_pattern" ]; then
    csv_list=$(ls $csv_pattern 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    echo "Using glob pattern: $csv_pattern"
else
    echo "No files specified. Set csv_files array or csv_pattern."
    exit 1
fi

if [ -z "$csv_list" ]; then
    echo "No files found!"
    exit 1
fi

echo "Files to process:"
echo "$csv_list" | tr ',' '\n'
echo ""

# Run judges
python -m eval.run_judges \
    --input_csvs "$csv_list" \
    --output_suffix "$output_suffix" \
    --judge_model "$judge_model" \
    --max_concurrent $max_concurrent \
    --custom_judges "$custom_judges_file"

echo "Done!"

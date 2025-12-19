gpu=${1:-0}
model="Qwen/Qwen2.5-7B-Instruct"
trait="altruism"
vector_path="persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt"
layer=20
steering_type="all"

# Array of coefficient values to test
coef_values=(-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0)

module load python
mamba activate interp

export HF_HOME="hf_cache"

# Loop over each coefficient value
for coef in "${coef_values[@]}"; do
    echo "Running evaluation with coefficient: $coef"

    # Update output path for this specific coefficient
    output_path="eval_persona_eval/$(basename $model)/v1/${trait}_steer_${steering_type}_layer${layer}_coef${coef}_game_v1.csv"

    CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
        --model $model \
        --trait $trait \
        --output_path $output_path \
        --judge_model gpt-4.1-mini-2025-04-14 \
        --version eval \
        --steering_type $steering_type \
        --coef $coef \
        --vector_path $vector_path \
        --layer $layer

    echo "Completed evaluation with coefficient: $coef"
    echo "Output saved to: $output_path"
    echo "---"
done

echo "All evaluations completed!"
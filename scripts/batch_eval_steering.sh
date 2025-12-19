gpu=${1:-0}

module load python
mamba activate interp

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona_batch \
    --model Qwen/Qwen2.5-7B-Instruct \
    --traits "altruism,forgiveness" \
    --output_dir "eval_persona_eval/Qwen2.5-7B-Instruct/v1" \
    --coefs "-3,-2,-1,0,1,2,3" \
    --vector_dir persona_vectors/Qwen2.5-7B-Instruct \
    --layer 20 \
    --steering_type all \
    --version eval \
    --output_template "{trait}_steer_{steering_type}_layer{layer}_coef{coef}_game_v1_compare_with_expectation.csv" \
    --n_per_question 30

echo "Done with all traits"
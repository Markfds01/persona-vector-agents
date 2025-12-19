gpu=${1:-0}

module load python
mamba activate interp

# traits=("forgiveness" "retaliation" "conditional_cooperation")

# CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona_batch \
#       --model Qwen/Qwen2.5-7B-Instruct \
#       --traits "${traits[@]}" \
#       --output_dir "eval_persona_extract/Qwen2.5-7B-Instruct" \
#       --persona_instruction_types "none,pos,neg" \
#       --judge_model gpt-4.1-mini-2025-04-14 \
#       --version extract \
#       --output_template "{trait}_game_{instruction_type}_instruct.csv"

CUDA_VISIBLE_DEVICES=$gpu python -m generate_vec_batch \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --traits "forgiveness,retaliation,conditional_cooperation" \
      --input_dir eval_persona_extract/Qwen2.5-7B-Instruct \
      --save_dir persona_vectors/Qwen2.5-7B-Instruct \
      --pos_template "{trait}_game_pos_instruct.csv" \
      --neg_template "{trait}_game_neg_instruct.csv" \
      --threshold 50
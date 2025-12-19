gpu=${1:-0}

module load python
mamba activate interp

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona_batch \
      --model Qwen/Qwen2.5-7B-Instruct \
      --traits "expected_altruism,expected_forgiveness" \
      --output_dir "eval_persona_extract/Qwen2.5-7B-Instruct" \
      --persona_instruction_types "none,pos,neg" \
      --judge_model gpt-4.1-mini-2025-04-14 \
      --version extract \
      --output_template "{trait}_game_{instruction_type}_instruct.csv"

echo "Done with all traits"
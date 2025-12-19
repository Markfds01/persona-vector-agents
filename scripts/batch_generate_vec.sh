gpu=${1:-0}

module load python
mamba activate interp

CUDA_VISIBLE_DEVICES=$gpu python -m generate_vec_batch \
      --model_name Qwen/Qwen2.5-7B-Instruct \
      --traits "expected_altruism,expected_forgiveness" \
      --input_dir eval_persona_extract/Qwen2.5-7B-Instruct \
      --save_dir persona_vectors/Qwen2.5-7B-Instruct \
      --pos_template "{trait}_game_pos_instruct.csv" \
      --neg_template "{trait}_game_neg_instruct.csv" \
      --threshold 50

echo "Done with all traits"
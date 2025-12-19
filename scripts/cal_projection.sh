gpu=${1:-0}
module load python
mamba activate interp

export HF_HOME="hf_cache"

CUDA_VISIBLE_DEVICES=$gpu python -m eval.cal_projection \
    --file_path eval_persona_eval/Qwen2.5-7B-Instruct/altruism_game_no_instruct.csv \
    --vector_path persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt \
    --layer 20 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --projection_type proj

# CUDA_VISIBLE_DEVICES=$gpu python -m eval.cal_projection \
#     --file_path eval_persona_eval/Qwen2.5-7B-Instruct/altruism_game_neg_instruct.csv \
#     --vector_path persona_vectors/Qwen2.5-7B-Instruct/altruism_response_avg_diff.pt \
#     --layer 20 \
#     --model_name Qwen/Qwen2.5-7B-Instruct \
#     --projection_type proj
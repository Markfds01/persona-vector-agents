gpu=${1:-0}

module load python
mamba activate interp

CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait altruism \
    --output_path eval_persona_eval/Qwen2.5-7B-Instruct/altruism_game_no_instruct.csv \
    --persona_instruction_type None \
    --assistant_name altruism \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version eval

# CUDA_VISIBLE_DEVICES=$gpu python -m eval.eval_persona \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --trait altruism \
#     --output_path eval_persona_eval/Qwen2.5-7B-Instruct/altruism_game_neg_instruct.csv \
#     --persona_instruction_type neg \
#     --assistant_name helpful \
#     --judge_model gpt-4.1-mini-2025-04-14 \
#     --version eval

# CUDA_VISIBLE_DEVICES=$gpu python generate_vec.py \
#     --model_name Qwen/Qwen2.5-7B-Instruct \
#     --pos_path eval_persona_extract/Qwen2.5-7B-Instruct/altruism_pos_instruct.csv \
#     --neg_path eval_persona_extract/Qwen2.5-7B-Instruct/altruism_neg_instruct.csv \
#     --trait altruism \
#     --save_dir persona_vectors/Qwen2.5-7B-Instruct/ \
#     --threshold 50
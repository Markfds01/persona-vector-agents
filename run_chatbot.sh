#!/bin/bash
# Launch the persona vector steering chatbot
#
# Usage:
#   bash run_chatbot.sh                    # Use defaults
#   bash run_chatbot.sh --share            # Create public URL
#   bash run_chatbot.sh --port 8080        # Use custom port
#   CUDA_VISIBLE_DEVICES=0 bash run_chatbot.sh  # Specify GPU

python chatbot_app.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --layer 20 \
    --vector-dir persona_vectors/Qwen2.5-7B-Instruct \
    "$@"

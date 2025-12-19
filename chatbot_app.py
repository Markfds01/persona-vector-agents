"""
Interactive Chatbot with Persona Vector Steering

A Gradio-based chat interface that allows real-time adjustment of persona vectors
using sliders. Each trait can be steered from -5 to +5, and the steering is applied
to the next response based on the current slider values.

Usage:
    python chatbot_app.py [--model MODEL_PATH] [--layer LAYER] [--share]

Examples:
    python chatbot_app.py
    python chatbot_app.py --model Qwen/Qwen2.5-7B-Instruct --layer 20
    python chatbot_app.py --share  # Create public URL
"""

import os
os.environ["HF_HOME"] = 'hf_cache'

import gradio as gr
import torch
from pathlib import Path
from activation_steer import ActivationSteererMultiple
from eval.model_utils import load_model
import argparse
from typing import List, Dict, Tuple

# Configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYER = 20
VECTOR_DIR = "persona_vectors/Qwen2.5-7B-Instruct"
STEERING_TYPE = "response"  # Apply steering to response tokens


def find_available_traits(vector_dir: str) -> List[str]:
    """Find all available traits by scanning for response_avg_diff.pt files."""
    vector_path = Path(vector_dir)
    if not vector_path.exists():
        return []

    traits = []
    for file in vector_path.glob("*_response_avg_diff.pt"):
        trait = file.stem.replace("_response_avg_diff", "")
        traits.append(trait)

    return sorted(traits)


def load_vectors(vector_dir: str, traits: List[str]) -> Dict[str, torch.Tensor]:
    """Load all persona vectors for the specified traits."""
    vectors = {}
    for trait in traits:
        vector_path = Path(vector_dir) / f"{trait}_response_avg_diff.pt"
        if vector_path.exists():
            vectors[trait] = torch.load(vector_path, weights_only=False)
            print(f"Loaded vector for '{trait}': shape {vectors[trait].shape}")
        else:
            print(f"Warning: Vector not found for '{trait}' at {vector_path}")
    return vectors


def format_steering_info(steering_coeffs: Dict[str, float]) -> str:
    """Format steering coefficients for display."""
    active_steerings = {k: v for k, v in steering_coeffs.items() if abs(v) > 0.01}
    if not active_steerings:
        return ""

    parts = [f"{trait}: {coef:+.1f}" for trait, coef in active_steerings.items()]
    return f"🎛️ *Steering: {', '.join(parts)}*"


class ChatbotApp:
    def __init__(self, model_name: str, layer: int, vector_dir: str):
        print(f"Loading model: {model_name}")
        self.model, self.tokenizer = load_model(model_name)
        self.layer = layer
        self.vector_dir = vector_dir

        # Find and load available traits
        self.traits = find_available_traits(vector_dir)
        if not self.traits:
            raise ValueError(f"No persona vectors found in {vector_dir}")

        print(f"Found {len(self.traits)} traits: {', '.join(self.traits)}")
        self.vectors = load_vectors(vector_dir, self.traits)

        print(f"Model loaded successfully! Using layer {layer} for steering.")

    def generate_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        steering_coeffs: Dict[str, float],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate a response with the specified steering coefficients."""
        # Build conversation from message-based history
        conversation = []
        for msg in history:
            conversation.append({"role": msg["role"], "content": msg["content"]})
        conversation.append({"role": "user", "content": message})

        # Format prompt
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Prepare steering instructions (only for non-zero coefficients)
        steering_instructions = []
        for trait, coef in steering_coeffs.items():
            if abs(coef) > 0.01 and trait in self.vectors:
                vector_full = self.vectors[trait]
                steering_instructions.append({
                    "steering_vector": vector_full[self.layer],
                    "coeff": coef,
                    "layer_idx": self.layer - 1,  # Convert to 0-indexed
                    "positions": STEERING_TYPE
                })

        # Generate with or without steering
        if steering_instructions:
            with ActivationSteererMultiple(self.model, steering_instructions):
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        do_sample=(temperature > 0),
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_tokens,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
        else:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        # Decode response
        prompt_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        return response


def create_ui(app: ChatbotApp):
    """Create the Gradio interface."""

    with gr.Blocks(title="Persona Vector Steering Chatbot") as demo:
        gr.Markdown(
            """
            # 🎭 Persona Vector Steering Chatbot

            Chat with an LLM while adjusting personality traits in real-time using sliders.
            Each slider controls the strength of a persona vector (-5 to +5).

            **Positive values** steer *towards* the trait, **negative values** steer *away* from it.
            """
        )

        with gr.Row():
            # Left column: Chat interface
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        scale=4
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)

                with gr.Accordion("Generation Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top P"
                    )
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=2000,
                        value=512,
                        step=50,
                        label="Max Tokens"
                    )

                clear = gr.Button("Clear Conversation")

            # Right column: Steering controls
            with gr.Column(scale=1):
                gr.Markdown("### 🎛️ Steering Controls")
                gr.Markdown("Adjust sliders to steer the model's personality")

                # Create sliders for each trait
                sliders = {}
                for trait in app.traits:
                    sliders[trait] = gr.Slider(
                        minimum=-5.0,
                        maximum=5.0,
                        value=0.0,
                        step=0.5,
                        label=trait.replace("_", " ").title(),
                        info=f"Steer {trait}"
                    )

                reset_sliders = gr.Button("Reset All Sliders")

        # State to track steering info for each message
        steering_history = gr.State([])

        def respond(message, chat_history, steering_hist, temp, top_p_val, max_tok, *slider_values):
            """Handle user message and generate response."""
            if not message.strip():
                return "", chat_history, steering_hist

            # Get current steering coefficients
            steering_coeffs = {trait: val for trait, val in zip(app.traits, slider_values)}

            # Generate response
            bot_response = app.generate_response(
                message,
                chat_history,
                steering_coeffs,
                max_tokens=int(max_tok),
                temperature=temp,
                top_p=top_p_val
            )

            # Format steering info
            steering_info = format_steering_info(steering_coeffs)

            # Add messages to history in the new format
            chat_history.append({"role": "user", "content": message})

            # Add steering info to bot response if present
            if steering_info:
                bot_content = f"{steering_info}\n\n{bot_response}"
            else:
                bot_content = bot_response

            chat_history.append({"role": "assistant", "content": bot_content})

            return "", chat_history, steering_hist

        def clear_chat():
            """Clear the conversation."""
            return [], []

        def reset_all_sliders():
            """Reset all sliders to 0."""
            return [0.0] * len(app.traits)

        # Connect event handlers
        slider_inputs = [sliders[trait] for trait in app.traits]

        submit.click(
            respond,
            inputs=[msg, chatbot, steering_history, temperature, top_p, max_tokens] + slider_inputs,
            outputs=[msg, chatbot, steering_history]
        )

        msg.submit(
            respond,
            inputs=[msg, chatbot, steering_history, temperature, top_p, max_tokens] + slider_inputs,
            outputs=[msg, chatbot, steering_history]
        )

        clear.click(clear_chat, outputs=[chatbot, steering_history])

        reset_sliders.click(
            reset_all_sliders,
            outputs=slider_inputs
        )

        gr.Markdown(
            f"""
            ---
            **Model:** {app.model.config._name_or_path if hasattr(app.model.config, '_name_or_path') else 'Unknown'}
            **Steering Layer:** {app.layer}
            **Available Traits:** {len(app.traits)}
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Interactive chatbot with persona vector steering")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name or path (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=f"Layer to apply steering (default: {DEFAULT_LAYER})"
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default=VECTOR_DIR,
        help=f"Directory containing persona vectors (default: {VECTOR_DIR})"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)"
    )

    args = parser.parse_args()

    # Initialize app
    app = ChatbotApp(args.model, args.layer, args.vector_dir)

    # Create and launch UI
    demo = create_ui(app)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()

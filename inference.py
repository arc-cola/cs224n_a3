"""Inference script for text generation with trained transformer models."""

import argparse
import torch
from transformers import AutoTokenizer

from model_solution import Transformer, ModelConfig


def load_model(checkpoint_path: str, device: torch.device) -> Transformer:
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def generate_text(
    model: Transformer,
    prompt: str,
    tokenizer,
    max_tokens: int,
    device: torch.device,
) -> str:
    """Generate text from a prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Truncate if longer than context length
    if input_ids.shape[1] > model.config.context_length:
        input_ids = input_ids[:, -model.config.context_length:]

    output_ids = model.generate(input_ids, num_new_tokens=max_tokens)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved checkpoint")
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Number of tokens to generate")
    args = parser.parse_args()

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = load_model(args.checkpoint, device)
    print(f"Loaded model with config: d_model={model.config.d_model}, n_heads={model.config.n_heads}, n_layers={model.config.n_layers}")

    # Generate text
    generated = generate_text(model, args.prompt, tokenizer, args.max_tokens, device)
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerated:\n{generated}")


if __name__ == "__main__":
    main()

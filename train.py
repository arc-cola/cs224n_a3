"""A simple training loop for our transformer model"""
import warnings
warnings.filterwarnings("ignore")
import os
import json
import gc

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Optional
from jaxtyping import Int
from torch import Tensor
import torch
import matplotlib.pyplot as plt

from model_solution import Transformer, ModelConfig


if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using Mac MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


def get_chunked_tinystories(
    chunk_size: int,
) -> Int[Tensor, "num_chunks chunk_size"]:

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load tiny stories dataset
    train_dataset = load_dataset("roneneldan/TinyStories")["train"]

    # We'll just grab the first 1%
    train_dataset = train_dataset.select(range(int(len(train_dataset) * 0.01)))

    # Tokenize the dataset
    chunks: List[List[int]] = []
    current_chunk: List[int] = []
    for row in tqdm(train_dataset, desc="Tokenizing dataset"):
        document: str = row["text"]
        tokens: List[int] = tokenizer(document, truncation=True, max_length=chunk_size).input_ids

        # Fill current chunk up to chunk_size
        current_chunk.extend(tokens)
        if len(current_chunk) > chunk_size:
            chunks.append(current_chunk[:chunk_size])
            # Reset the current chunk
            current_chunk = current_chunk[chunk_size:]

    # Sanity checks
    assert all(len(chunk) == chunk_size for chunk in chunks)

    return torch.tensor(chunks, dtype=torch.long)


def plot_results(
    losses: List[float],
    grad_norms: List[float],
    save_path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel - Loss curve
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Right panel - Gradient norm
    ax2.plot(grad_norms)
    ax2.set_title('Gradient Norm')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Grad Norm')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train(
    learning_rate: float,
    gradient_clipping: Optional[float],
    model_config: ModelConfig,
    batch_size: int,
    max_steps: Optional[int] = None,
    save_path=f"./losses_and_grad_norms.png"
) -> None:

    if gradient_clipping is None:
        # This lets us just get the grad norm but we don't clip
        gradient_clipping = float("inf")

    chunk_size: int = model_config.context_length
    cached_dataset_path: str = f"./datasets/tinystories_10pct_chunk_size_{chunk_size}.pt"
    os.makedirs(os.path.dirname(cached_dataset_path), exist_ok=True)
    
    if os.path.exists(cached_dataset_path):
        dataset = torch.load(cached_dataset_path)
    else:
        dataset: Int[Tensor, "num_chunks chunk_size"] = get_chunked_tinystories(chunk_size)
        torch.save(dataset, cached_dataset_path)


    # Create dense batches of [batch_size, seq_len]
    model = Transformer(model_config).to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

    num_chunks: int = dataset.shape[0]

    losses: List[float] = []
    grad_norms: List[float] = []
    num_steps_completed: int = 0

    if max_steps is not None:
        tqdm_max_steps = min(max_steps, num_chunks // batch_size)
    else:
        tqdm_max_steps = num_chunks // batch_size

    for i in tqdm(range(0, num_chunks, batch_size), desc="Training", total=tqdm_max_steps):

        if max_steps is not None and num_steps_completed >= max_steps:
            break

        if num_steps_completed % 10 == 0 and num_steps_completed > 0:
            plot_results(losses, grad_norms, save_path=save_path)

        batch: Int[Tensor, "batch_size chunk_size"] = dataset[i:i+batch_size].to(device)

        optimizer.zero_grad()

        # Forward pass
        loss = model.get_loss_on_batch(batch)

        # Backward pass
        loss.backward()

        # Clip gradients
        with torch.no_grad():
            grad_norm: float = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping).item()
            grad_norms.append(grad_norm)

        optimizer.step()

        losses.append(loss.item())

        num_steps_completed += 1


    # Done with training, plot results in single plot
    plot_results(losses, grad_norms, save_path=save_path)

    return {"losses": losses, "grad_norms": grad_norms, "final_loss": losses[-1], "file_name": save_path}



if __name__ == "__main__":

    hidden_dims = [40, 100, 200]
    ns_heads = [1, 5, 10, 20]
    ns_layers = [1, 2, 10]
    batch_sizes = [16]
    gradient_clippings = [1, 1.5, 2.5]
    save_dir=f"./losses_and_grad_norms/"
    json_save_dir = f"./results/"

    for hidden_dim in hidden_dims:
        for n_heads in ns_heads:
            for n_layers in ns_layers:
                for batch_size in batch_sizes:
                    for gradient_clipping in gradient_clippings:
                        file_name = f"hd{hidden_dim}_nh{n_heads}_nl{n_layers}_bs{batch_size}_gc{gradient_clipping}"
                        png_file_name = file_name + ".png"
                        json_file_name = file_name + ".json"
                        print(f"processing file {file_name}")
                        if png_file_name in os.listdir(save_dir) and json_file_name in os.listdir(json_save_dir):
                            continue
                        save_path = save_dir + png_file_name

                        tiny_model_config = ModelConfig(
                            d_model=hidden_dim,
                            n_heads=n_heads,
                            n_layers=n_layers,
                            context_length=512,
                            vocab_size=50257,
                        )

                        results = train(
                                learning_rate=1e-4,
                                gradient_clipping=gradient_clipping,
                                model_config = tiny_model_config,
                                batch_size=batch_size,
                                max_steps=100,
                                save_path=save_path
                            )
                        with open(json_save_dir + json_file_name, 'w') as results_file:
                            json.dump(results, results_file)

                        gc.collect()
                        torch.cuda.empty_cache()
    


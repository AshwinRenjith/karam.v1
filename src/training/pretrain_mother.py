from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Generator

import torch
from tqdm import tqdm

# Imports from your architecture
from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.utils.tokenizer import FractalTokenizer
from src.registry.node_registry import NodeRegistry
from src.utils.device import get_default_device

# External libraries
try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install datasets: pip install datasets")


def pretrain_mother_node(
    steps: int = 1000,
    batch_size: int = 8,
    save_path: str = "./weights/root_node.pt",
    registry_path: str = "./registry.json",
) -> None:
    """Pre-train the Mother Node on wikitext-2 to establish base knowledge."""
    print("=" * 60)
    print("🌅 PHASE 6: THE AWAKENING (Mother Node Pre-training)")
    print("=" * 60)

    # 1. Setup Hardware
    device_info = get_default_device()
    device = device_info.device
    print(f"🔧 Device Locked: {device} (MPS Enabled: {device_info.is_mps})")

    # 2. Initialize the "Eye" (Tokenizer)
    tokenizer = FractalTokenizer(model_name="gpt2")

    # 3. Initialize the "Brain" (Mother Node)
    # Matching architecture_specs.md prototype values
    config = FractalConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=512,
        dropout=0.1,
    )

    model = FractalTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🧠 Mother Node Initialized: {param_count:.2f}M Params")

    # 4. Load Knowledge (Wikitext-2 is clean and fast)
    print("📚 Streaming Wikitext-2 data...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train", streaming=True)

    def data_generator() -> Generator[str, None, None]:
        for item in dataset:
            text = item["text"]
            if len(text) > 50:  # Skip short headers/empty lines
                yield text

    # 5. The Training Loop (The "Learning")
    model.train()
    start_time = time.time()
    running_loss = 0.0

    iterator = iter(data_generator())

    print(f"🚀 Starting {steps} steps of pre-training...")
    progress_bar = tqdm(range(steps))

    for step in progress_bar:
        try:
            # Batch accumulation
            batch_texts: list[str] = []
            while len(batch_texts) < batch_size:
                batch_texts.append(next(iterator))

            # Tokenize
            tokens = torch.cat(
                [tokenizer.encode(t, max_len=config.max_seq_len) for t in batch_texts]
            ).to(device)

            # Forward Pass (Auto-regressive: Targets are same as inputs)
            logits, loss = model(tokens, targets=tokens)

            if loss is None:
                continue

            # Backward Pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

            # Cleanup for M1
            if step % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        except StopIteration:
            print("⚠️ Dataset exhausted, stopping early.")
            break

    avg_loss = running_loss / max(1, steps)
    print(f"\n📊 Average Loss: {avg_loss:.4f}")

    # 6. Save the "Educated" Brain
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Weights saved to {save_path}")

    # 7. Register in Genesis (The Registry)
    registry = NodeRegistry(registry_path=registry_path)

    # Dummy centroid for the root (Zero vector or generic "everything")
    # In production, this would be the mean of all English vectors.
    dummy_centroid = [0.0] * 384  # 384 is dim of all-MiniLM-L6-v2

    metadata = {
        "parent_id": None,  # The Root has no parent
        "depth": 0,
        "centroid_vector": dummy_centroid,
        "file_path": save_path,
        "config": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "vocab_size": config.vocab_size,
        },
        "status": "active",
    }

    registry.register_node("root_node", metadata)
    print("✅ Registry updated: 'root_node' is now online.")

    elapsed = (time.time() - start_time) / 60
    print(f"⏱️ Total Time: {elapsed:.1f} minutes")
    print("=" * 60)
    print("🎉 THE MOTHER NODE HAS AWAKENED!")
    print("=" * 60)


if __name__ == "__main__":
    # Run slightly longer if you have time, 500 steps is bare minimum for "grammar"
    pretrain_mother_node(steps=500, batch_size=4)

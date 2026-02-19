from __future__ import annotations

import gc
import os
import time
from pathlib import Path

# --- 1. MEMORY OPTIMIZATION (Unlock extra VRAM) ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install datasets: pip install datasets")

from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.registry.node_registry import NodeRegistry
from src.utils.device import get_default_device
from src.utils.tokenizer import FractalTokenizer

# --- CONFIGURATION ---
STEPS = 5000
BATCH_SIZE = 4  # Reduced to prevent OOM on Mac
SEQ_LEN = 256
LEARNING_RATE = 3e-4
DATASET_NAME = "roneneldan/TinyStories"


class StreamDataset(IterableDataset):
    def __init__(self, tokenizer: FractalTokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    def __iter__(self):
        for sample in self.ds:
            text = sample.get("text", "")
            if len(text) < 15:
                continue
            token_ids = self.tokenizer.encode(text, max_len=self.max_seq_len + 1)
            token_ids = token_ids[:, : self.max_seq_len + 1]
            yield token_ids.squeeze(0)


def pretrain_mother_node(steps: int = STEPS, batch_size: int = BATCH_SIZE) -> None:
    print("=" * 60)
    print(f"🌅 PHASE 10: THE STORYTELLER ({steps} Steps | Batch {batch_size})")
    print("=" * 60)

    device_info = get_default_device()
    print(f"🔧 Device Locked: {device_info.device} (MPS Enabled: {device_info.is_mps})")

    tokenizer = FractalTokenizer()
    pad_id = tokenizer._tokenizer.pad_token_id
    registry = NodeRegistry(registry_path="./registry.json")
    save_path = Path("./weights/root_node_v2.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = FractalConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=SEQ_LEN,
        dropout=0.1,
    )

    model = FractalTransformer(config).to(device_info.device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🧠 Mother Node Initialized: {param_count:.2f}M Params")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    print(f"📚 Streaming {DATASET_NAME}...")
    dataset = StreamDataset(tokenizer, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    total_loss = 0.0
    start_time = time.time()

    print("🚀 Starting training run...")

    step = 0
    for batch in dataloader:
        if step >= steps:
            break

        # MEMORY SAFETY: Aggressive cleanup
        optimizer.zero_grad(set_to_none=True)

        batch = batch.to(device_info.device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        logits, _ = model(inputs)

        bsz, seq_len, vocab = logits.shape
        loss = criterion(logits.reshape(bsz * seq_len, vocab), targets.reshape(bsz * seq_len))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Print Progress
        if step % 50 == 0:
            avg_loss = total_loss / (step + 1)
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (steps - step) / steps_per_sec if steps_per_sec > 0 else 0.0
            mins = int(remaining // 60)

            print(
                f"Step {step}/{steps} | Loss: {loss.item():.4f} "
                f"(Avg: {avg_loss:.4f}) | ⏳ ~{mins}m left"
            )

            # MEMORY SAFETY: Clear cache periodically
            if device_info.is_mps:
                torch.mps.empty_cache()
            gc.collect()

        step += 1

    torch.save(model.state_dict(), save_path)
    print(f"\n💾 Weights saved to {save_path}")
    print("⚠️ Note: Ensure registry.json is cleared if this is a fresh architecture run.")

    registry.register_node(
        "root_node",
        {
            "parent_id": None,
            "depth": 0,
            "centroid_vector": [0.0] * 512,
            "file_path": str(save_path),
            "config": {
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "n_layers": config.n_layers,
                "vocab_size": config.vocab_size,
                "d_ff": config.d_ff,
                "max_seq_len": config.max_seq_len,
                "pad_token_id": pad_id,
            },
            "status": "active",
            "arch_version": "v2_rope_swiglu",
            "model_family": "fractal_transformer_v2",
            "checkpoint_compat": "strict",
        },
    )

    print("✅ Registry updated. 'root_node' is now online.")
    print("=" * 60)


if __name__ == "__main__":
    pretrain_mother_node()

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import torch

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Missing datasets. Install with: pip install datasets"
    ) from exc

from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.utils.tokenizer import FractalTokenizer
from src.router.semantic_router import SemanticRouter
from src.registry.node_registry import NodeRegistry
from src.utils.device import get_default_device


@dataclass
class FractalSupervisor:
    tokenizer: FractalTokenizer = field(default_factory=lambda: FractalTokenizer(model_name="gpt2"))
    mother: Optional[FractalTransformer] = None
    router: Optional[SemanticRouter] = None
    registry: Optional[NodeRegistry] = None
    loss_history: Dict[str, List[float]] = field(default_factory=dict)
    mitosis_threshold: float = 2.0
    min_history_len: int = 100
    max_seq_len: int = 128

    def __post_init__(self) -> None:
        device_info = get_default_device()

        # CRITICAL: Use tokenizer.vocab_size to prevent IndexError
        config = FractalConfig(
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=1024,
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=self.max_seq_len,
        )

        self.mother = FractalTransformer(config).to(device_info.device)
        self.router = SemanticRouter()
        self.registry = NodeRegistry(registry_path="./registry.json")
        self.loss_history = {}
        self._device = device_info.device
        self._child_counter = 0

    def stream_data(
        self, dataset_name: str = "wikitext", config_name: str = "wikitext-2-raw-v1"
    ) -> Generator[str, None, None]:
        ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
        for sample in ds:
            text = sample.get("text", "")
            if text and len(text.strip()) > 20:
                yield text.strip()

    def train_step(self, batch_texts: List[str]) -> Tuple[float, str]:
        assert self.mother is not None
        assert self.router is not None
        assert self.registry is not None

        # 1. Tokenize batch
        tokens_list: List[torch.Tensor] = []
        for text in batch_texts:
            toks = self.tokenizer.encode(text, max_len=self.max_seq_len)
            tokens_list.append(toks)
        tokens = torch.cat(tokens_list, dim=0).to(self._device)

        # 2. Forward pass (training mode)
        self.mother.eval()
        targets = tokens.clone()
        logits, loss = self.mother(tokens, targets)

        if loss is None:
            loss = torch.tensor(0.0, device=self._device)

        loss_val = float(loss.item())

        # 3. Route text to get topic ID
        with torch.no_grad():
            route_result = self.router.route(batch_texts[0])
        topic_id = route_result.node_id

        # 4. Update loss history
        if topic_id not in self.loss_history:
            self.loss_history[topic_id] = []
        self.loss_history[topic_id].append(loss_val)

        # 5. Mitosis check
        history = self.loss_history[topic_id]
        mean_loss = sum(history) / len(history) if history else 0.0
        triggered_mitosis = False

        if mean_loss > self.mitosis_threshold and len(history) >= self.min_history_len:
            print(f"📉 Loss: {loss_val:.2f} | Topic: {topic_id} | ⚠️ High Error Detected!")
            print(f"🧬 Triggering MITOSIS for topic '{topic_id}'...")
            triggered_mitosis = True
            self._trigger_mitosis(topic_id, batch_texts)
            self.loss_history[topic_id] = []
        else:
            status = "⚠️ High" if loss_val > self.mitosis_threshold else "✅ OK"
            print(f"📉 Loss: {loss_val:.2f} | Topic: {topic_id} | {status}")

        # Cleanup
        del tokens, targets, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return loss_val, topic_id

    def _trigger_mitosis(self, topic_id: str, sample_texts: List[str]) -> None:
        assert self.mother is not None
        assert self.registry is not None

        # Create mini dataset for distillation
        def mini_loader() -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
            for text in sample_texts:
                toks = self.tokenizer.encode(text, max_len=self.max_seq_len).to(self._device)
                yield toks, toks.clone()

        # Spawn child via bud()
        child = self.mother.bud(mini_loader())

        # Save child weights
        self._child_counter += 1
        child_id = f"{topic_id}_child_{self._child_counter}"
        weights_dir = Path("./weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / f"{child_id}.pt"
        torch.save(child.state_dict(), weights_path)

        # Get centroid vector for child
        with torch.no_grad():
            centroid = self.router.encoder.encode(sample_texts[0])

        # Register child in NodeRegistry
        self.registry.register_node(
            child_id,
            {
                "parent_id": self.mother_id,
                "depth": child.config.depth,
                "centroid_vector": centroid[0].cpu().tolist(),
                "file_path": str(weights_path),
                "config": {
                    "d_model": child.config.d_model,
                    "n_heads": child.config.n_heads,
                    "n_layers": child.config.n_layers,
                    "d_ff": child.config.d_ff,
                    "vocab_size": child.config.vocab_size,
                    "max_seq_len": child.config.max_seq_len,
                    "pad_token_id": 50256,
                },
                "status": "active",
                "arch_version": "v1",
                "model_family": "fractal_transformer",
                "checkpoint_compat": "strict",
            },
        )

        # Update router centroids
        self.router.centroids.update(child_id, centroid[0])

        print(f"✅ Child '{child_id}' spawned and registered!")

        del child, centroid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    print("=" * 60)
    print("🧠 FractalSupervisor - Mitosis Trigger Test")
    print("=" * 60)

    supervisor = FractalSupervisor(
        mitosis_threshold=1.5,
        min_history_len=3,  # Low threshold for testing
    )

    # Mock training data (simulating streaming)
    mock_data: List[List[str]] = [
        ["The laws of physics govern the motion of objects in space."],
        ["Quantum mechanics describes the behavior of subatomic particles."],
        ["Einstein's theory of relativity revolutionized our understanding of gravity."],
        ["Thermodynamics deals with heat and energy transfer in systems."],
        ["Newton's laws of motion are fundamental to classical mechanics."],
    ]

    print("\n🚀 Starting training loop...\n")
    for i, batch in enumerate(mock_data):
        print(f"--- Step {i + 1} ---")
        loss, topic = supervisor.train_step(batch)

    print("\n" + "=" * 60)
    print("✅ Supervisor test complete!")
    print(f"Loss history: {supervisor.loss_history}")
    print("=" * 60)

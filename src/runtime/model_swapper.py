from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import gc
from pathlib import Path

import torch

from src.utils.device import get_default_device
from src.models.fractal_transformer import FractalConfig, FractalTransformer


@dataclass
class ModelSwapper:
    active_models: Dict[str, FractalTransformer] = field(default_factory=dict)

    def load_model(self, node_id: str, file_path: str, config: FractalConfig) -> FractalTransformer:
        device_info = get_default_device()
        model = FractalTransformer(config).to(device_info.device)
        if Path(file_path).exists():
            state = torch.load(file_path, map_location=device_info.device)
            model.load_state_dict(state)
        model.eval()
        self.active_models[node_id] = model
        return model

    def unload_model(self, node_id: str) -> None:
        if node_id in self.active_models:
            del self.active_models[node_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def get_model(self, node_id: str) -> Optional[FractalTransformer]:
        return self.active_models.get(node_id)


if __name__ == "__main__":
    cfg = FractalConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        vocab_size=1000,
        max_seq_len=16,
    )
    swapper = ModelSwapper()
    model = swapper.load_model("root", "./weights/root.pt", cfg)
    dummy = torch.randint(0, 1000, (2, 8))
    logits, _ = model(dummy)
    print(f"input shape={dummy.shape}, output shape={logits.shape}")
    assert logits.shape == (2, 8, cfg.vocab_size)

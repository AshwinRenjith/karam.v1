from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import gc

import torch

from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.registry.node_registry import NodeRegistry
from src.router.semantic_router import SemanticRouter
from src.runtime.model_swapper import ModelSwapper
from src.runtime.stream_aggregator import StreamAggregator
from src.utils.device import get_default_device


@dataclass
class SimpleTokenizer:
    vocab_size: int

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        tokens = [t for t in text.lower().split() if t]
        ids = [hash(t) % self.vocab_size for t in tokens][:max_len]
        if not ids:
            ids = [0]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        return " ".join([str(i) for i in ids])


def create_app(
    registry_path: str,
    root_config: FractalConfig,
    root_weights_path: str,
) -> "FastAPI":
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FastAPI is required for the inference API") from exc

    device_info = get_default_device()
    router = SemanticRouter()
    registry = NodeRegistry(registry_path=registry_path)
    swapper = ModelSwapper()
    aggregator = StreamAggregator()
    tokenizer = SimpleTokenizer(vocab_size=root_config.vocab_size)

    root_model = swapper.load_model("root_node", root_weights_path, root_config)

    app = FastAPI()

    class GenerateRequest(BaseModel):
        text: str
        max_len: int = 32

    class GenerateResponse(BaseModel):
        node_id: str
        confidence: float
        tokens: List[int]

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        route = router.route(req.text)
        node_id = route.node_id
        node = registry.get_node(node_id)
        if node is None:
            model = root_model
        else:
            node_cfg = node["config"]
            config = FractalConfig(
                d_model=node_cfg["d_model"],
                n_heads=node_cfg["n_heads"],
                n_layers=node_cfg["n_layers"],
                d_ff=4 * node_cfg["d_model"],
                vocab_size=root_config.vocab_size,
                max_seq_len=root_config.max_seq_len,
            )
            model = swapper.load_model(node_id, node["file_path"], config)

        token_ids = tokenizer.encode(req.text, req.max_len).to(device_info.device)
        token_ids = token_ids.unsqueeze(0)
        logits, _ = model(token_ids)
        next_ids = logits.argmax(dim=-1).squeeze(0).tolist()

        del logits, token_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return GenerateResponse(
            node_id=node_id,
            confidence=route.confidence_score,
            tokens=next_ids,
        )

    return app


if __name__ == "__main__":
    cfg = FractalConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        vocab_size=1000,
        max_seq_len=16,
    )
    model = FractalTransformer(cfg)
    dummy = torch.randint(0, 1000, (2, 6))
    logits, _ = model(dummy)
    print(f"input shape={dummy.shape}, output shape={logits.shape}")
    assert logits.shape == (2, 6, cfg.vocab_size)

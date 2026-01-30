from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import gc

import torch

from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.registry.node_registry import NodeRegistry
from src.router.semantic_router import SemanticRouter
from src.runtime.model_swapper import ModelSwapper
from src.utils.device import get_default_device
from src.utils.tokenizer import FractalTokenizer  # REAL TOKENIZER


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

    # FIX: Use Real Tokenizer (GPT-2 to match Supervisor)
    tokenizer = FractalTokenizer(model_name="gpt2")

    root_model = swapper.load_model("root_node", root_weights_path, root_config)

    app = FastAPI()

    class GenerateRequest(BaseModel):
        text: str
        max_len: int = 32

    class GenerateResponse(BaseModel):
        node_id: str
        confidence: float
        decoded_text: str  # Changed from 'tokens' to readable text

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        # 1. Route
        route = router.route(req.text)
        node_id = route.node_id

        # 2. Swap Model
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
                vocab_size=root_config.vocab_size,  # Inherit Vocab
                max_seq_len=root_config.max_seq_len,
            )
            model = swapper.load_model(node_id, node["file_path"], config)

        # 3. Tokenize (Real GPT-2 IDs)
        token_ids = tokenizer.encode(req.text, req.max_len).to(device_info.device)

        # 4. Generate (Greedy Decode for Prototype)
        generated = token_ids
        for _ in range(10):  # Generate 10 new tokens
            logits, _ = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated = torch.cat((generated, next_token), dim=1)

        # 5. Decode
        output_text = tokenizer.decode(generated)

        # Cleanup
        del logits, token_ids, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return GenerateResponse(
            node_id=node_id,
            confidence=route.confidence_score,
            decoded_text=output_text,
        )

    return app


if __name__ == "__main__":
    from src.utils.tokenizer import FractalTokenizer

    tokenizer = FractalTokenizer(model_name="gpt2")
    cfg = FractalConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=16,
    )
    model = FractalTransformer(cfg)
    dummy = tokenizer.encode("Hello world", max_len=6).to("cpu")
    logits, _ = model(dummy)
    print(f"input shape={dummy.shape}, output shape={logits.shape}")
    assert logits.shape == (1, 6, cfg.vocab_size)
    print("✅ Inference API test passed with FractalTokenizer!")

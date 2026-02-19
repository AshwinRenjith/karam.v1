from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import gc

import torch
import torch.nn.functional as F

from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.registry.node_registry import NodeRegistry
from src.router.semantic_router import SemanticRouter
from src.runtime.model_swapper import ModelSwapper
from src.utils.device import get_default_device
from src.utils.tokenizer import FractalTokenizer  # REAL TOKENIZER

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError("FastAPI/Pydantic missing.") from exc


class GenerateRequest(BaseModel):
    text: str
    max_len: int = 64
    temperature: float = 0.8
    top_k: int = 50


class GenerateResponse(BaseModel):
    node_id: str
    confidence: float
    decoded_text: str


def create_app(
    registry_path: str,
    root_config: FractalConfig,
    root_weights_path: str,
    expected_arch_version: str = "v1",
) -> "FastAPI":
    device_info = get_default_device()
    router = SemanticRouter()
    registry = NodeRegistry(registry_path=registry_path)
    swapper = ModelSwapper()

    # FIX: Use Real Tokenizer (GPT-2 to match Supervisor)
    tokenizer = FractalTokenizer(model_name="gpt2")
    pad_id = tokenizer._tokenizer.pad_token_id
    eos_id = tokenizer._tokenizer.eos_token_id

    root_meta = registry.get_node("root_node")
    root_arch = root_meta.get("arch_version") if root_meta is not None else expected_arch_version
    root_model = swapper.load_model(
        "root_node",
        root_weights_path,
        root_config,
        checkpoint_arch_version=root_arch,
        expected_arch_version=expected_arch_version,
    )

    app = FastAPI()

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
                d_ff=node_cfg["d_ff"],
                vocab_size=node_cfg.get("vocab_size", root_config.vocab_size),
                max_seq_len=node_cfg.get("max_seq_len", root_config.max_seq_len),
            )
            node_arch = node.get("arch_version")
            model = swapper.load_model(
                node_id,
                node["file_path"],
                config,
                checkpoint_arch_version=node_arch,
                expected_arch_version=expected_arch_version,
            )

        # 3. Tokenize & strip padding
        full_tokens = tokenizer.encode(req.text, req.max_len).to(device_info.device)
        mask = full_tokens[0] != pad_id
        token_ids = full_tokens[0][mask].unsqueeze(0)

        # 4. Generate with sampling
        generated = token_ids
        model.eval()
        for _ in range(req.max_len):
            with torch.no_grad():
                logits, _ = model(generated)

            next_token_logits = logits[:, -1, :]
            temperature = req.temperature if req.temperature > 0 else 1.0
            next_token_logits = next_token_logits / temperature

            for token_id in set(generated[0].tolist()):
                next_token_logits[0, token_id] /= 1.1

            top_k = max(1, int(req.top_k))
            top_k = min(top_k, next_token_logits.size(-1))
            top_k_probs, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_probs, dim=-1)
            next_token_index = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_index)

            token_val = int(next_token.item())
            if token_val == pad_id or token_val == eos_id:
                break
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

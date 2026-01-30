from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Missing transformers. Install dependencies with: pip install -r requirements.txt"
    ) from exc


@dataclass
class FractalTokenizer:
    model_name: str = "gpt2"

    def __post_init__(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.vocab_size)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        return encoded["input_ids"].to(dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        ids: List[int] = token_ids.to(dtype=torch.long).tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = FractalTokenizer()
    tokens = tokenizer.encode("Hello world", max_len=8)
    print(f"tokens shape={tokens.shape}, vocab_size={tokenizer.vocab_size}")
    decoded = tokenizer.decode(tokens)
    print(f"decoded={decoded}")
    assert tokens.shape == (1, 8)
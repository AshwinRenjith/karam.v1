from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class StreamAggregator:
    def average_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        if not logits_list:
            raise ValueError("logits_list cannot be empty")
        stacked = torch.stack(logits_list, dim=0)
        return stacked.mean(dim=0)


if __name__ == "__main__":
    aggregator = StreamAggregator()
    dummy = [torch.randn(2, 4, 10), torch.randn(2, 4, 10)]
    out = aggregator.average_logits(dummy)
    print(f"input shape={dummy[0].shape}, output shape={out.shape}")
    assert out.shape == (2, 4, 10)

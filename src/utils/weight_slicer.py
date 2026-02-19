from __future__ import annotations

import torch


class WeightSlicer:
    """Utilities for slicing parent weights to fit child modules."""

    @staticmethod
    def slice_linear(weight: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        return weight[:out_dim, :in_dim].clone()

    @staticmethod
    def slice_embedding(weight: torch.Tensor, d_model: int) -> torch.Tensor:
        return weight[:, :d_model].clone()

    @staticmethod
    def slice_layer_norm(weight: torch.Tensor, d_model: int) -> torch.Tensor:
        return weight[:d_model].clone()

    @staticmethod
    def slice_projection(weight: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
        return weight[:out_dim, :in_dim].clone()


if __name__ == "__main__":
    dummy = torch.randn(8, 16)
    sliced = WeightSlicer.slice_linear(dummy, out_dim=4, in_dim=6)
    print(f"input shape={dummy.shape}, output shape={sliced.shape}")
    assert sliced.shape == (4, 6)

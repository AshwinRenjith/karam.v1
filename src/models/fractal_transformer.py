from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import gc

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from src.utils.device import get_default_device
from src.utils.weight_slicer import WeightSlicer


@dataclass(frozen=True)
class FractalConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    dropout: float = 0.1
    scaling_factor: float = 0.5
    depth: int = 0


def scale_config(parent: FractalConfig) -> FractalConfig:
    d_child = max(64, int(parent.d_model * parent.scaling_factor))
    n_heads = max(2, int(parent.n_heads * parent.scaling_factor))
    n_layers = max(2, int(parent.n_layers * 0.75))
    if d_child % n_heads != 0:
        d_child = (d_child // n_heads) * n_heads
    d_ff = 4 * d_child
    return FractalConfig(
        d_model=d_child,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        vocab_size=parent.vocab_size,
        max_seq_len=parent.max_seq_len,
        dropout=parent.dropout,
        scaling_factor=parent.scaling_factor,
        depth=parent.depth + 1,
    )


class FractalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "Dimension mismatch"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: FractalConfig) -> None:
        super().__init__()
        self.attn = FractalAttention(config.d_model, config.n_heads, config.dropout)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FractalTransformer(nn.Module):
    def __init__(self, config: FractalConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, "Dimension mismatch"
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len = tokens.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError("Sequence length exceeds maximum")
        device = tokens.device
        positions = torch.arange(0, seq_len, device=device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        return logits, loss

    def bud(self, specialized_data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> "FractalTransformer":
        child_config = scale_config(self.config)
        assert child_config.d_model <= self.config.d_model, "Child larger than parent"
        child = FractalTransformer(child_config)
        self._inherit_weights(child)
        child.train()
        device_info = get_default_device()
        child = child.to(device_info.device)

        teacher = self.eval().to(device_info.device)
        optimizer = torch.optim.Adam(child.parameters(), lr=3e-4)
        temperature = 2.0
        alpha = 0.5

        last_teacher: Optional[torch.Tensor] = None
        last_logits: Optional[torch.Tensor] = None
        last_tokens: Optional[torch.Tensor] = None
        last_targets: Optional[torch.Tensor] = None
        for tokens, targets in specialized_data_loader:
            tokens = tokens.to(device_info.device)
            targets = targets.to(device_info.device)
            with torch.no_grad():
                teacher_logits, _ = teacher(tokens)
            child_logits, _ = child(tokens)
            shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            shift_child_logits = child_logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            hard_loss = F.cross_entropy(
                shift_child_logits.view(-1, shift_child_logits.size(-1)),
                shift_labels.view(-1),
            )
            soft_loss = F.kl_div(
                F.log_softmax(shift_child_logits / temperature, dim=-1),
                F.softmax(shift_teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            last_teacher = teacher_logits
            last_logits = child_logits
            last_tokens = tokens
            last_targets = targets

        if last_teacher is not None:
            del last_teacher
        if last_logits is not None:
            del last_logits
        if last_tokens is not None:
            del last_tokens
        if last_targets is not None:
            del last_targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return child

    def _inherit_weights(self, child: "FractalTransformer") -> None:
        child.token_emb.weight.data = WeightSlicer.slice_embedding(
            self.token_emb.weight.data, child.config.d_model
        )
        child.pos_emb.weight.data = WeightSlicer.slice_embedding(
            self.pos_emb.weight.data, child.config.d_model
        )
        child.lm_head.weight.data = WeightSlicer.slice_linear(
            self.lm_head.weight.data, child.config.vocab_size, child.config.d_model
        )
        child.ln_f.weight.data = WeightSlicer.slice_layer_norm(
            self.ln_f.weight.data, child.config.d_model
        )
        child.ln_f.bias.data = WeightSlicer.slice_layer_norm(
            self.ln_f.bias.data, child.config.d_model
        )

        for parent_block, child_block in zip(self.blocks, child.blocks):
            child_block.ln1.weight.data = WeightSlicer.slice_layer_norm(
                parent_block.ln1.weight.data, child.config.d_model
            )
            child_block.ln1.bias.data = WeightSlicer.slice_layer_norm(
                parent_block.ln1.bias.data, child.config.d_model
            )
            child_block.ln2.weight.data = WeightSlicer.slice_layer_norm(
                parent_block.ln2.weight.data, child.config.d_model
            )
            child_block.ln2.bias.data = WeightSlicer.slice_layer_norm(
                parent_block.ln2.bias.data, child.config.d_model
            )

            child_block.attn.qkv.weight.data = WeightSlicer.slice_qkv(
                parent_block.attn.qkv.weight.data,
                out_dim=3 * child.config.d_model,
                in_dim=child.config.d_model,
            )
            child_block.attn.qkv.bias.data = WeightSlicer.slice_bias(
                parent_block.attn.qkv.bias.data, out_dim=3 * child.config.d_model
            )
            child_block.attn.out.weight.data = WeightSlicer.slice_linear(
                parent_block.attn.out.weight.data,
                out_dim=child.config.d_model,
                in_dim=child.config.d_model,
            )
            child_block.attn.out.bias.data = WeightSlicer.slice_bias(
                parent_block.attn.out.bias.data, out_dim=child.config.d_model
            )

            child_block.ffn.net[0].weight.data = WeightSlicer.slice_ffn(
                parent_block.ffn.net[0].weight.data,
                out_dim=child.config.d_ff,
                in_dim=child.config.d_model,
            )
            child_block.ffn.net[0].bias.data = WeightSlicer.slice_bias(
                parent_block.ffn.net[0].bias.data, out_dim=child.config.d_ff
            )
            child_block.ffn.net[3].weight.data = WeightSlicer.slice_ffn(
                parent_block.ffn.net[3].weight.data,
                out_dim=child.config.d_model,
                in_dim=child.config.d_ff,
            )
            child_block.ffn.net[3].bias.data = WeightSlicer.slice_bias(
                parent_block.ffn.net[3].bias.data, out_dim=child.config.d_model
            )


if __name__ == "__main__":
    cfg = FractalConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        vocab_size=1000,
        max_seq_len=16,
    )
    model = FractalTransformer(cfg)
    dummy = torch.randint(0, 1000, (2, 10))
    logits, _ = model(dummy)
    print(f"input shape={dummy.shape}, output shape={logits.shape}")
    assert logits.shape == (2, 10, cfg.vocab_size)

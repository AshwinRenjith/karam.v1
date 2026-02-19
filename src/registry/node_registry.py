from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
from pathlib import Path

import torch


@dataclass
class NodeRegistry:
    registry_path: Path
    data: Dict[str, Dict[str, Any]]

    def __init__(self, registry_path: str = "./registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.data = {}
        if self.registry_path.exists():
            self.load()

    def load(self) -> None:
        raw = json.loads(self.registry_path.read_text())
        normalized: Dict[str, Dict[str, Any]] = {}
        changed = False
        for node_id, metadata in raw.items():
            node_meta, node_changed = self._normalize_metadata(node_id, metadata)
            normalized[node_id] = node_meta
            changed = changed or node_changed
        self.data = normalized
        if changed:
            self.save()

    def save(self) -> None:
        self.registry_path.write_text(json.dumps(self.data, indent=2))

    def register_node(self, node_id: str, metadata: Dict[str, Any]) -> None:
        normalized, _ = self._normalize_metadata(node_id, metadata)
        parent_id = normalized.get("parent_id")
        if parent_id is not None and parent_id not in self.data:
            raise ValueError("parent_id does not exist in registry")
        file_path = normalized.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required")
        if not Path(file_path).exists():
            raise ValueError("file_path does not exist")
        self.data[node_id] = normalized
        self.save()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(node_id)

    def _normalize_metadata(self, node_id: str, metadata: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        normalized = dict(metadata)
        changed = False

        if "arch_version" not in normalized:
            normalized["arch_version"] = "v1"
            changed = True
        if "model_family" not in normalized:
            normalized["model_family"] = "fractal_transformer"
            changed = True
        if "checkpoint_compat" not in normalized:
            normalized["checkpoint_compat"] = "strict"
            changed = True

        config = dict(normalized.get("config") or {})
        required_core = ["d_model", "n_heads", "n_layers"]
        missing_core = [key for key in required_core if key not in config]
        if missing_core:
            raise ValueError(f"node '{node_id}' missing required config keys: {', '.join(missing_core)}")

        d_model = int(config["d_model"])
        defaults = {
            "d_ff": 4 * d_model,
            "vocab_size": 50257,
            "max_seq_len": 512,
            "pad_token_id": 50256,
        }
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
                changed = True

        if normalized.get("config") != config:
            normalized["config"] = config
            changed = True

        return normalized, changed


if __name__ == "__main__":
    registry = NodeRegistry(registry_path="./registry.test.json")
    dummy_centroid = torch.randn(1, 384)
    weights_path = Path("./weights/root_node.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"")
    registry.register_node(
        "root_node",
        {
            "parent_id": None,
            "depth": 0,
            "centroid_vector": dummy_centroid[0].tolist(),
            "file_path": str(weights_path),
            "config": {
                "d_model": 128,
                "n_heads": 4,
                "n_layers": 2,
                "d_ff": 512,
                "vocab_size": 50257,
                "max_seq_len": 64,
                "pad_token_id": 50256,
            },
            "status": "active",
            "arch_version": "v1",
        },
    )
    retrieved = registry.get_node("root_node")
    print(f"input shape={dummy_centroid.shape}, output shape={torch.tensor(retrieved['centroid_vector']).shape}")
    assert retrieved is not None

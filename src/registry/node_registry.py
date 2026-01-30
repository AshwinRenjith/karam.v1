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
        self.data = json.loads(self.registry_path.read_text())

    def save(self) -> None:
        self.registry_path.write_text(json.dumps(self.data, indent=2))

    def register_node(self, node_id: str, metadata: Dict[str, Any]) -> None:
        parent_id = metadata.get("parent_id")
        if parent_id is not None and parent_id not in self.data:
            raise ValueError("parent_id does not exist in registry")
        file_path = metadata.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required")
        if not Path(file_path).exists():
            raise ValueError("file_path does not exist")
        self.data[node_id] = metadata
        self.save()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(node_id)


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
            "config": {"d_model": 128, "n_heads": 4, "n_layers": 2},
            "status": "active",
        },
    )
    retrieved = registry.get_node("root_node")
    print(f"input shape={dummy_centroid.shape}, output shape={torch.tensor(retrieved['centroid_vector']).shape}")
    assert retrieved is not None

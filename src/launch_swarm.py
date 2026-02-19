from __future__ import annotations

from pathlib import Path

import uvicorn

from src.models.fractal_transformer import FractalConfig
from src.registry.node_registry import NodeRegistry
from src.runtime.inference_api import create_app


def start_swarm(expected_arch_version: str = "v1") -> None:
    print("=" * 60)
    print("🛸 KARAM.v1 - SWARM INTERFACE INITIATED")
    print("=" * 60)

    registry_path = "./registry.json"
    registry = NodeRegistry(registry_path=registry_path)

    mother_id = "root_node"
    node_data = registry.get_node(mother_id)

    if node_data is None:
        raise RuntimeError(f"❌ Could not find '{mother_id}' in registry! Did you run pretraining?")

    node_arch = node_data.get("arch_version", "unknown")
    print(f"🧠 Loading Cortex: {mother_id} | arch_version={node_arch}")

    if node_arch != expected_arch_version:
        raise RuntimeError(
            f"❌ Incompatible root architecture: found '{node_arch}', expected '{expected_arch_version}'"
        )

    cfg = node_data["config"]
    root_config = FractalConfig(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        vocab_size=cfg.get("vocab_size", 50257),
        max_seq_len=cfg.get("max_seq_len", 512),
    )

    root_weights = node_data["file_path"]
    if not Path(root_weights).exists():
        raise RuntimeError(f"❌ Root weights not found: {root_weights}")

    app = create_app(
        registry_path=registry_path,
        root_config=root_config,
        root_weights_path=root_weights,
        expected_arch_version=expected_arch_version,
    )

    print("✅ Swarm is listening on http://localhost:8000")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_swarm()

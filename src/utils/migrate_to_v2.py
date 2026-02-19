from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import shutil


def migrate_to_v2(
    registry_path: str = "./registry.json",
    weights_dir: str = "./weights",
    archive_root: str = "./weights/archive/v1",
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(archive_root) / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    registry = Path(registry_path)
    if registry.exists():
        backup_registry = archive_dir / "registry.v1.backup.json"
        shutil.copy2(registry, backup_registry)
        print(f"📦 Backed up registry to {backup_registry}")

    weights = Path(weights_dir)
    moved = 0
    if weights.exists():
        for weight_file in weights.glob("*.pt"):
            destination = archive_dir / weight_file.name
            shutil.move(str(weight_file), destination)
            moved += 1
            print(f"📦 Archived {weight_file} -> {destination}")

    if registry.exists():
        registry.write_text(json.dumps({}, indent=2))
        print(f"🧹 Reset registry at {registry}")

    print("=" * 60)
    print(f"✅ Phase 0 migration complete. Archived {moved} weight file(s).")
    print(f"📁 Archive directory: {archive_dir}")
    print("=" * 60)


if __name__ == "__main__":
    migrate_to_v2()

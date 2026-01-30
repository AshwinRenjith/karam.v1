from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src.registry.node_registry import NodeRegistry
from src.router.semantic_router import SemanticRouter
from src.models.fractal_transformer import FractalConfig, FractalTransformer
from src.utils.device import get_default_device


@dataclass
class NodeStats:
    """Tracks usage statistics for a single node."""
    node_id: str
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    total_latency_ms: float = 0.0

    def record_access(self, latency_ms: float = 0.0) -> None:
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.total_latency_ms += latency_ms

    def days_since_access(self) -> float:
        delta = datetime.now() - self.last_accessed
        return delta.total_seconds() / 86400.0


@dataclass
class LifeCycleManager:
    """
    The "Immune System" of the Fractal-Dendritic Network.
    
    Responsibilities:
    1. Node Pruning (Apoptosis): Remove unused/dead nodes
    2. Manifold Fusion: Merge similar overlapping experts
    """
    registry: NodeRegistry
    router: SemanticRouter
    stats: Dict[str, NodeStats] = field(default_factory=dict)
    
    # Pruning thresholds
    prune_after_days: float = 7.0
    prune_after_steps: int = 10000
    min_access_count: int = 10  # Don't prune if accessed frequently
    
    # Fusion thresholds
    fusion_similarity_threshold: float = 0.95
    
    # Internal counters
    _global_step: int = 0

    def __init__(
        self,
        registry: NodeRegistry,
        router: SemanticRouter,
        prune_after_days: float = 7.0,
        fusion_similarity_threshold: float = 0.95,
    ) -> None:
        self.registry = registry
        self.router = router
        self.stats = {}
        self.prune_after_days = prune_after_days
        self.fusion_similarity_threshold = fusion_similarity_threshold
        self._global_step = 0

    def record_access(self, node_id: str, latency_ms: float = 0.0) -> None:
        """Record that a node was accessed (routed to)."""
        if node_id not in self.stats:
            self.stats[node_id] = NodeStats(node_id=node_id)
        self.stats[node_id].record_access(latency_ms)
        self._global_step += 1

    def step(self) -> None:
        """Increment global step counter."""
        self._global_step += 1

    # =========================================================================
    # APOPTOSIS: Node Pruning
    # =========================================================================
    def find_dead_nodes(self) -> List[str]:
        """
        Identify nodes that should be pruned (Apoptosis candidates).
        
        Criteria:
        - Not accessed in `prune_after_days` days
        - Access count below threshold
        - Not the root_node (never prune the Mother)
        """
        dead_nodes: List[str] = []
        
        for node_id, node_data in self.registry.data.items():
            # Never prune the root
            if node_id == "root_node":
                continue
            
            # Check if we have stats for this node
            if node_id not in self.stats:
                # Node was registered but never accessed - candidate for pruning
                # But give new nodes a grace period
                continue
            
            stats = self.stats[node_id]
            days_idle = stats.days_since_access()
            
            # Prune if idle too long AND not frequently accessed
            if days_idle > self.prune_after_days and stats.access_count < self.min_access_count:
                dead_nodes.append(node_id)
                
        return dead_nodes

    def prune_node(self, node_id: str) -> bool:
        """
        Execute Apoptosis: Remove a node from the system.
        
        1. Delete weights from disk
        2. Remove from registry
        3. Remove from router centroids
        4. Clear stats
        """
        if node_id == "root_node":
            print(f"⚠️ Cannot prune root_node (The Mother is immortal)")
            return False
        
        node_data = self.registry.get_node(node_id)
        if node_data is None:
            print(f"⚠️ Node '{node_id}' not found in registry")
            return False
        
        # 1. Delete weights file
        file_path = Path(node_data.get("file_path", ""))
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️ Deleted weights: {file_path}")
        
        # 2. Remove from registry
        del self.registry.data[node_id]
        self.registry.save()
        
        # 3. Remove from router centroids
        if node_id in self.router.centroids.centroids:
            del self.router.centroids.centroids[node_id]
        
        # 4. Clear stats
        if node_id in self.stats:
            del self.stats[node_id]
        
        print(f"☠️ APOPTOSIS: Node '{node_id}' has been pruned")
        return True

    def run_pruning_cycle(self) -> int:
        """Run a full pruning cycle. Returns number of nodes pruned."""
        dead_nodes = self.find_dead_nodes()
        pruned_count = 0
        
        for node_id in dead_nodes:
            if self.prune_node(node_id):
                pruned_count += 1
        
        if pruned_count > 0:
            print(f"🧹 Pruning cycle complete: {pruned_count} nodes removed")
        
        return pruned_count

    # =========================================================================
    # MANIFOLD FUSION: Merge Similar Experts
    # =========================================================================
    def find_fusion_candidates(self) -> List[Tuple[str, str, float]]:
        """
        Find pairs of nodes with high centroid similarity.
        
        Returns list of (node_a, node_b, similarity_score) tuples.
        """
        candidates: List[Tuple[str, str, float]] = []
        centroids = dict(self.router.centroids.items())
        node_ids = list(centroids.keys())
        
        for i, node_a in enumerate(node_ids):
            # Skip root node
            if node_a == "root_node":
                continue
                
            for node_b in node_ids[i + 1:]:
                # Skip root node
                if node_b == "root_node":
                    continue
                
                vec_a = centroids[node_a]
                vec_b = centroids[node_b]
                
                similarity = self._cosine_similarity(vec_a, vec_b)
                
                if similarity >= self.fusion_similarity_threshold:
                    candidates.append((node_a, node_b, similarity))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return float((a_norm * b_norm).sum(dim=-1).item())

    def fuse_nodes(self, node_a_id: str, node_b_id: str) -> Optional[str]:
        """
        Merge two similar nodes into one.
        
        Math: W_fused = (W_a + W_b) / 2
        
        The merged node keeps node_a's ID and metadata.
        Node_b is pruned after fusion.
        """
        device_info = get_default_device()
        
        node_a = self.registry.get_node(node_a_id)
        node_b = self.registry.get_node(node_b_id)
        
        if node_a is None or node_b is None:
            print(f"⚠️ Cannot fuse: one or both nodes not found")
            return None
        
        # Load both models
        config_a = node_a["config"]
        config_b = node_b["config"]
        
        # Check compatibility (must have same architecture)
        if config_a["d_model"] != config_b["d_model"]:
            print(f"⚠️ Cannot fuse: incompatible architectures")
            return None
        
        # Build config for loading
        full_config = FractalConfig(
            d_model=config_a["d_model"],
            n_heads=config_a["n_heads"],
            n_layers=config_a["n_layers"],
            d_ff=4 * config_a["d_model"],
            vocab_size=config_a.get("vocab_size", 50257),
            max_seq_len=512,
        )
        
        # Load weights
        path_a = Path(node_a["file_path"])
        path_b = Path(node_b["file_path"])
        
        if not path_a.exists() or not path_b.exists():
            print(f"⚠️ Cannot fuse: weight files missing")
            return None
        
        state_a = torch.load(path_a, map_location=device_info.device)
        state_b = torch.load(path_b, map_location=device_info.device)
        
        # Average the weights: W_fused = (W_a + W_b) / 2
        fused_state = {}
        for key in state_a.keys():
            if key in state_b:
                fused_state[key] = (state_a[key] + state_b[key]) / 2.0
            else:
                fused_state[key] = state_a[key]
        
        # Save fused weights (overwrite node_a)
        torch.save(fused_state, path_a)
        
        # Average centroids
        vec_a = self.router.centroids.get(node_a_id)
        vec_b = self.router.centroids.get(node_b_id)
        if vec_a is not None and vec_b is not None:
            fused_centroid = (vec_a + vec_b) / 2.0
            self.router.centroids.update(node_a_id, fused_centroid)
        
        # Combine stats
        if node_a_id in self.stats and node_b_id in self.stats:
            self.stats[node_a_id].access_count += self.stats[node_b_id].access_count
        
        # Prune node_b (it's been absorbed)
        self.prune_node(node_b_id)
        
        print(f"🔀 FUSION: '{node_a_id}' + '{node_b_id}' → '{node_a_id}'")
        
        # Cleanup
        del state_a, state_b, fused_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return node_a_id

    def run_fusion_cycle(self) -> int:
        """Run a full fusion cycle. Returns number of fusions performed."""
        candidates = self.find_fusion_candidates()
        fusion_count = 0
        
        # Track which nodes have been fused (to avoid double-fusing)
        fused_nodes: set[str] = set()
        
        for node_a, node_b, similarity in candidates:
            if node_a in fused_nodes or node_b in fused_nodes:
                continue
            
            result = self.fuse_nodes(node_a, node_b)
            if result is not None:
                fusion_count += 1
                fused_nodes.add(node_b)  # node_b was absorbed
        
        if fusion_count > 0:
            print(f"🧬 Fusion cycle complete: {fusion_count} merges performed")
        
        return fusion_count

    # =========================================================================
    # FULL MAINTENANCE CYCLE
    # =========================================================================
    def run_maintenance(self) -> Dict[str, int]:
        """
        Run full immune system maintenance:
        1. Prune dead nodes (Apoptosis)
        2. Fuse similar nodes (Manifold Fusion)
        """
        print("=" * 50)
        print("🛡️ IMMUNE SYSTEM: Running Maintenance Cycle")
        print("=" * 50)
        
        pruned = self.run_pruning_cycle()
        fused = self.run_fusion_cycle()
        
        print(f"📊 Summary: {pruned} pruned, {fused} fused")
        print("=" * 50)
        
        return {"pruned": pruned, "fused": fused}


if __name__ == "__main__":
    import tempfile
    
    print("=" * 60)
    print("🧪 LifeCycleManager - Immune System Test")
    print("=" * 60)
    
    # Create temp registry
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        registry_path = f.name
    
    registry = NodeRegistry(registry_path=registry_path)
    router = SemanticRouter()
    
    # Create lifecycle manager
    manager = LifeCycleManager(
        registry=registry,
        router=router,
        prune_after_days=0.001,  # Very short for testing
        fusion_similarity_threshold=0.9,
    )
    
    # Simulate some node registrations
    weights_dir = Path("./weights")
    weights_dir.mkdir(exist_ok=True)
    
    # Create dummy weight files
    dummy_state = {"layer": torch.randn(10, 10)}
    for name in ["test_node_1", "test_node_2"]:
        path = weights_dir / f"{name}.pt"
        torch.save(dummy_state, path)
        
        registry.register_node(name, {
            "parent_id": None,
            "depth": 1,
            "centroid_vector": [0.0] * 384,
            "file_path": str(path),
            "config": {"d_model": 64, "n_heads": 2, "n_layers": 2},
            "status": "active",
        })
        
        # Add to router centroids
        router.centroids.update(name, torch.randn(384))
    
    print(f"✅ Created {len(registry.data)} test nodes")
    
    # Record some accesses
    manager.record_access("test_node_1")
    manager.record_access("test_node_1")
    
    # Find dead nodes (test_node_2 should be candidate since never accessed)
    dead = manager.find_dead_nodes()
    print(f"🔍 Dead node candidates: {dead}")
    
    # Run maintenance
    results = manager.run_maintenance()
    print(f"📊 Maintenance results: {results}")
    
    # Cleanup
    Path(registry_path).unlink(missing_ok=True)
    print("\n✅ LifeCycleManager test complete!")

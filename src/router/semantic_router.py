from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union
import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Missing sentence-transformers. Install dependencies with: pip install -r requirements.txt"
    ) from exc


class RouteResult(NamedTuple):
    node_id: str
    confidence_score: float
    routing_path: List[str]


@dataclass
class VectorEncoder:
    model_name: str = "all-MiniLM-L6-v2"
    _model: Optional["SentenceTransformer"] = None

    def _get_device(self) -> str:
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._model = self._model.to(self._get_device())
        return self._model

    def encode(self, texts: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        model = self._ensure_model()
        vectors = model.encode(texts, convert_to_tensor=True)
        return vectors.to(dtype=torch.float32)


@dataclass
class CentroidManager:
    centroids: Dict[str, torch.Tensor]

    def __init__(self) -> None:
        self.centroids = {}

    def update(self, node_id: str, vector: torch.Tensor) -> None:
        self.centroids[node_id] = vector.detach().clone()

    def get(self, node_id: str) -> torch.Tensor:
        return self.centroids[node_id]

    def items(self) -> Iterable[Tuple[str, torch.Tensor]]:
        return self.centroids.items()


@dataclass
class RouterGate:
    threshold: float = 0.75

    def select(self, query: torch.Tensor, centroids: Dict[str, torch.Tensor]) -> RouteResult:
        best_id = "root_node"
        best_score = 0.0
        for node_id, centroid in centroids.items():
            score = self.cosine_similarity(query, centroid)
            if score > best_score:
                best_score = score
                best_id = node_id
        if best_score < self.threshold:
            return RouteResult(node_id="root_node", confidence_score=best_score, routing_path=["root"])
        return RouteResult(node_id=best_id, confidence_score=best_score, routing_path=["root", best_id])

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
        b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
        return float((a_norm * b_norm).sum(dim=-1).item())


@dataclass
class SemanticRouter:
    encoder: VectorEncoder
    centroids: CentroidManager
    gate: RouterGate

    def __init__(self, threshold: float = 0.75, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.encoder = VectorEncoder(model_name=model_name)
        self.centroids = CentroidManager()
        self.gate = RouterGate(threshold=threshold)

    def route(self, text_input: Union[str, Sequence[str]]) -> Union[RouteResult, List[RouteResult]]:
        vectors = self.encoder.encode(text_input)
        if isinstance(text_input, str):
            return self.gate.select(vectors[0], dict(self.centroids.items()))
        results: List[RouteResult] = []
        for i in range(vectors.size(0)):
            results.append(self.gate.select(vectors[i], dict(self.centroids.items())))
        return results


if __name__ == "__main__":
    router = SemanticRouter()
    dummy_vector = router.encoder.encode("Hello world")
    router.centroids.update("greeting_node", dummy_vector[0])
    result = router.route("Hi there")
    print(f"vector shape={dummy_vector.shape}, node_id={result.node_id}, score={result.confidence_score}")
    assert isinstance(result.node_id, str)

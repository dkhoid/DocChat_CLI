import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from docchat.chunker import Chunk
from docchat.embedder import BaseEmbedder


# ── Math helpers ──────────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vector."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk: Chunk
    score: float


# ── Vector store ──────────────────────────────────────────────────────────────

@dataclass
class SimpleVectorStore:
    """
    Vector store đơn giản lưu trong RAM.
    Dùng cosine similarity để tìm kiếm.
    Có thể save/load từ file.
    """
    embedder: BaseEmbedder
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk]) -> None:
        """Embed và lưu một batch chunk vào store."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        new_vectors = self.embedder.embed_batch(texts)
        self.chunks.extend(chunks)
        self.vectors.extend(new_vectors)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """
        Tìm k chunk gần nhất với query.

        Returns:
            List SearchResult sắp xếp theo score giảm dần.
        """
        if not self.chunks:
            return []

        query_vec = self.embedder.embed(query)
        scored = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vec, vec))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        """Lưu store ra file pickle — không cần embed lại lần sau."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"chunks": self.chunks, "vectors": self.vectors}
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Index saved: {len(self.chunks)} chunks → {save_path}")

    def load(self, path: str | Path) -> None:
        """Load store từ file pickle."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Index not found: {load_path}")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.vectors = data["vectors"]
        print(f"Index loaded: {len(self.chunks)} chunks ← {load_path}")

    @property
    def size(self) -> int:
        return len(self.chunks)
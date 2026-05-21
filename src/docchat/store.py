import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from docchat.chunker import Chunk
from docchat.embedder import BaseEmbedder

# ── Math helpers ──────────────────────────────────────────────────────────────


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Tính cosine similarity giữa hai vector (numpy vectorized)."""
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (norm_a * norm_b))


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


# ── Interfaces ────────────────────────────────────────────────────────────────


class BaseStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass


# ── Legacy Vector Store (Pickle) ──────────────────────────────────────────────


@dataclass
class SimpleVectorStore(BaseStore):
    """
    Vector store đơn giản lưu trong RAM (Dùng để testing và maintain backward compatibility).
    """

    embedder: BaseEmbedder
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        new_vectors = self.embedder.embed_batch(texts)
        self.chunks.extend(chunks)
        self.vectors.extend(new_vectors)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
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
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [
                {
                    "text": c.text,
                    "source": c.source,
                    "index": c.index,
                    "chunk_num": c.chunk_num,
                    "id": c.id,
                }
                for c in self.chunks
            ],
            "vectors": self.vectors,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Index saved: {len(self.chunks)} chunks → {save_path}")

    def load(self, path: str | Path) -> None:
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Index not found: {load_path}")
        with open(load_path, encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.vectors = data["vectors"]

    def clear(self) -> None:
        self.chunks.clear()
        self.vectors.clear()

    @property
    def size(self) -> int:
        return len(self.chunks)


# ── Advanced Vector Store (ChromaDB + BM25 + CrossEncoder) ────────────────────


class ChromaVectorStore(BaseStore):
    """
    Lưu trữ Production sử dụng Persistent ChromaDB kèm với:
    - Sparse Search (BM25 + pyvi).
    - Dense Search (Chroma).
    - Merge Search Results (RRF).
    - Reranking (CrossEncoder đa ngữ).
    """

    def __init__(self, embedder: BaseEmbedder, collection_name: str = "docchat"):
        self.embedder = embedder
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._bm25 = None
        self._bm25_chunks: list[Chunk] = []
        self._reranker_model = None
        import threading

        self._reranker_lock = threading.Lock()

    def _init_bm25(self, chunks: list[Chunk]):
        if not chunks:
            self._bm25 = None
            self._bm25_chunks = []
            return

        try:
            from pyvi import ViTokenizer
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Vui lòng cài đặt rank_bm25 và pyvi (uv add rank_bm25 pyvi)")

        tokenized_corpus = [ViTokenizer.tokenize(c.text).split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_chunks = chunks

    def _get_all_chunks(self) -> list[Chunk]:
        if self._collection is None:
            return []
        try:
            results = self._collection.get(include=["documents", "metadatas"])
        except Exception:
            return []

        if not results.get("ids"):
            return []

        chunks = []
        for id_, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            c = Chunk(
                text=doc,
                source=meta["source"],
                index=meta.get("index", 0),
                chunk_num=meta.get("chunk_num", 0),
                id=id_,
            )
            chunks.append(c)
        return chunks

    @property
    def chunks(self) -> list[Chunk]:
        return self._get_all_chunks()

    def _remove_source(self, source: str) -> None:
        """Xóa tất cả chunks thuộc source khỏi ChromaDB và BM25 (dedup upsert)."""
        if self._collection is None:
            return
        try:
            existing = self._collection.get(where={"source": str(source)}, include=[])
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])
                removed_ids = set(existing["ids"])
                self._bm25_chunks = [c for c in self._bm25_chunks if c.id not in removed_ids]
        except Exception:
            pass

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        if self._collection is None:
            raise RuntimeError(
                "Cần gọi load() hoặc save() để khởi tạo thiết lập Chroma Client trước khi lưu data."
            )

        # Dedup: xóa chunks cũ cùng source trước khi thêm mới (upsert by source)
        sources_to_add = {str(c.source) for c in chunks}
        for source in sources_to_add:
            self._remove_source(source)

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)

        ids = []
        metadatas = []
        for c in chunks:
            ids.append(c.id)
            metadatas.append(
                {"source": str(c.source), "index": int(c.index), "chunk_num": int(c.chunk_num)}
            )

        self._collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

        # BM25 incremental: extend in-memory list thay vì round-trip ChromaDB
        self._bm25_chunks.extend(chunks)
        self._init_bm25(self._bm25_chunks)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self._collection is None:
            return []

        k_fetch = max(20, k * 2)

        # 1. 🔍 Dense Search (Chroma - Semantic Vector DB)
        dense_chunks = []
        try:
            query_vec = self.embedder.embed(query)
            dense_results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=k_fetch,
                include=["documents", "metadatas", "distances"],
            )

            if dense_results and dense_results.get("ids") and dense_results["ids"][0]:
                for id_, doc, meta, dist in zip(
                    dense_results["ids"][0],
                    dense_results["documents"][0],
                    dense_results["metadatas"][0],
                    dense_results["distances"][0],
                ):
                    c = Chunk(
                        text=doc,
                        source=meta["source"],
                        index=meta.get("index", 0),
                        chunk_num=meta.get("chunk_num", 0),
                        id=id_,
                    )
                    sim_score = max(0.0, 1.0 - (dist / 2.0))
                    dense_chunks.append(SearchResult(chunk=c, score=sim_score))
        except Exception as e:
            print(f"Lỗi Dense Search: {e}")

        # 2. 🔍 Sparse Search (BM25 - Keyword Matching theo ngữ pháp tiếng Việt)
        sparse_chunks = []
        if self._bm25 is not None:
            from pyvi import ViTokenizer

            tokenized_query = ViTokenizer.tokenize(query).split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:k_fetch]
            for i in top_indices:
                score = bm25_scores[i]
                if score > 0:
                    c = self._bm25_chunks[i]
                    sparse_chunks.append(SearchResult(chunk=c, score=score))

        # 3. ⚖️ Merge kết quả bằng thuật toán RRF
        rrf_results = self._rrf([dense_chunks, sparse_chunks], k_rrf=60, k_out=k_fetch)

        if not rrf_results:
            return []

        # 4. 🥇 Rerank lại Top-N bằng Cross-Encoder (Multi_Lingual)
        if self._reranker_model is None:
            with self._reranker_lock:
                if self._reranker_model is None:
                    from sentence_transformers import CrossEncoder

                    self._reranker_model = CrossEncoder(
                        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
                    )

        pairs = [[query, res.chunk.text] for res in rrf_results]
        rerank_scores = self._reranker_model.predict(pairs)

        for i, res in enumerate(rrf_results):
            raw_score = float(rerank_scores[i])
            # Normalize bằng Sigmoid để đưa về khoảng [0, 1] thay vì logit âm/dương
            res.score = 1.0 / (1.0 + math.exp(-raw_score))

        # Xếp hạng lại theo điểm Cross-encoder mạnh mẽ rồi giới hạn mốc k cuối cùng nạp vào LLM.
        rrf_results.sort(key=lambda x: x.score, reverse=True)
        return rrf_results[:k]

    def _rrf(
        self, results_list: list[list[SearchResult]], k_rrf: int = 60, k_out: int = 20
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion (Chấm gộp điểm Sparse + Dense)"""
        scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}
        for results in results_list:
            for rank, item in enumerate(results):
                item_id = item.chunk.id
                scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k_rrf + rank + 1)
                chunk_map[item_id] = item.chunk

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:k_out]
        return [SearchResult(chunk=chunk_map[i], score=scores[i]) for i in sorted_ids]

    def save(self, data_dir: str | Path) -> None:
        """Kích hoạt Chroma DB trong data_dir/chroma_db. (Chroma Persistence mode tự động save)."""
        import chromadb

        db_path = str(Path(data_dir) / "chroma_db")
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def load(self, data_dir: str | Path) -> None:
        """Ngàm nạp lại toàn bộ DB lên RAM memory bao gồm BM25."""
        import chromadb

        db_path = str(Path(data_dir) / "chroma_db")
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

        all_chunks = self._get_all_chunks()
        self._init_bm25(all_chunks)
        print(f"Chroma Index loaded: {len(all_chunks)} chunks ← {db_path}")

    def clear(self) -> None:
        """Làm sạch toàn bộ cơ sở dữ liệu trên HDD."""
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        self._init_bm25([])

    @property
    def size(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

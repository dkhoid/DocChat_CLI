import pytest
from pathlib import Path

from docchat.chunker import Chunk
from docchat.store import SimpleVectorStore, SearchResult, cosine_similarity
from tests.test_embedder import FakeEmbedder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def store(embedder: FakeEmbedder) -> SimpleVectorStore:
    return SimpleVectorStore(embedder=embedder)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(text="Python là ngôn ngữ lập trình phổ biến", source="a.txt", index=0, chunk_num=0),
        Chunk(text="Machine learning dùng nhiều toán học", source="b.txt", index=0, chunk_num=0),
        Chunk(text="FastAPI là framework Python cho REST API", source="a.txt", index=100, chunk_num=1),
        Chunk(text="Neural network có nhiều lớp ẩn", source="b.txt", index=100, chunk_num=1),
        Chunk(text="Docker giúp đóng gói ứng dụng", source="c.txt", index=0, chunk_num=0),
    ]


@pytest.fixture
def populated_store(store: SimpleVectorStore, chunks: list[Chunk]) -> SimpleVectorStore:
    store.add(chunks)
    return store


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_cosine_same_vector():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_range():
    import random
    random.seed(42)
    a = [random.random() for _ in range(16)]
    b = [random.random() for _ in range(16)]
    score = cosine_similarity(a, b)
    assert -1.0 <= score <= 1.0


# ── SimpleVectorStore.add ─────────────────────────────────────────────────────

def test_add_increases_size(store: SimpleVectorStore, chunks: list[Chunk]):
    store.add(chunks)
    assert store.size == len(chunks)


def test_add_empty_list(store: SimpleVectorStore):
    store.add([])
    assert store.size == 0


def test_add_multiple_batches(store: SimpleVectorStore, chunks: list[Chunk]):
    store.add(chunks[:2])
    store.add(chunks[2:])
    assert store.size == len(chunks)


def test_vectors_same_length_as_chunks(populated_store: SimpleVectorStore):
    assert len(populated_store.vectors) == len(populated_store.chunks)


# ── SimpleVectorStore.search ──────────────────────────────────────────────────

def test_search_returns_list(populated_store: SimpleVectorStore):
    results = populated_store.search("Python", k=3)
    assert isinstance(results, list)


def test_search_returns_search_results(populated_store: SimpleVectorStore):
    results = populated_store.search("Python", k=3)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_k_limit(populated_store: SimpleVectorStore):
    results = populated_store.search("query", k=2)
    assert len(results) <= 2


def test_search_sorted_by_score(populated_store: SimpleVectorStore):
    results = populated_store.search("Python framework", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_store(store: SimpleVectorStore):
    results = store.search("anything")
    assert results == []


def test_search_k_larger_than_store(populated_store: SimpleVectorStore):
    """k lớn hơn số chunk → trả về tất cả."""
    results = populated_store.search("test", k=100)
    assert len(results) == populated_store.size


# ── Save / Load ───────────────────────────────────────────────────────────────

def test_save_creates_file(populated_store: SimpleVectorStore, tmp_path: Path):
    save_path = tmp_path / "index.pkl"
    populated_store.save(save_path)
    assert save_path.exists()


def test_load_restores_chunks(
    embedder: FakeEmbedder,
    populated_store: SimpleVectorStore,
    tmp_path: Path,
):
    save_path = tmp_path / "index.pkl"
    populated_store.save(save_path)

    new_store = SimpleVectorStore(embedder=embedder)
    new_store.load(save_path)

    assert new_store.size == populated_store.size
    assert [c.text for c in new_store.chunks] == [c.text for c in populated_store.chunks]


def test_load_nonexistent_file(store: SimpleVectorStore, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "nonexistent.pkl")


def test_save_creates_parent_dirs(populated_store: SimpleVectorStore, tmp_path: Path):
    nested_path = tmp_path / "deep" / "nested" / "index.pkl"
    populated_store.save(nested_path)
    assert nested_path.exists()
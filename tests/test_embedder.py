from unittest.mock import MagicMock, patch

import pytest

from docchat.embeddings.embedder import (
    BaseEmbedder,
    EmbedderFactory,
    OpenAIEmbedder,
    retry,
)

# ── Fake embedder dùng trong test (không cần API) ─────────────────────────────


class FakeEmbedder(BaseEmbedder):
    """Embedder giả — trả về vector cố định, không gọi API."""

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        # Mỗi text ra vector khác nhau dựa vào hash
        seed = hash(text) % 1000 / 1000
        return [seed + i * 0.1 for i in range(self._dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=4)


# ── BaseEmbedder (ABC) ────────────────────────────────────────────────────────


def test_cannot_instantiate_base_embedder():
    """ABC không cho khởi tạo trực tiếp."""
    with pytest.raises(TypeError):
        BaseEmbedder()


def test_fake_embedder_is_valid_embedder(fake_embedder: FakeEmbedder):
    assert isinstance(fake_embedder, BaseEmbedder)


def test_embed_returns_list_of_floats(fake_embedder: FakeEmbedder):
    result = fake_embedder.embed("hello")
    assert isinstance(result, list)
    assert all(isinstance(v, float) for v in result)


def test_embed_dimension(fake_embedder: FakeEmbedder):
    result = fake_embedder.embed("test")
    assert len(result) == fake_embedder.dimension


def test_embed_batch_returns_list_of_vectors(fake_embedder: FakeEmbedder):
    texts = ["a", "b", "c"]
    result = fake_embedder.embed_batch(texts)
    assert len(result) == 3
    assert all(len(v) == fake_embedder.dimension for v in result)


def test_embed_documents_uses_embed_batch(fake_embedder: FakeEmbedder):
    """embed_documents nên delegate cho embed_batch."""
    texts = ["doc1", "doc2"]
    result = fake_embedder.embed_documents(texts)
    assert len(result) == 2


def test_different_texts_different_vectors(fake_embedder: FakeEmbedder):
    v1 = fake_embedder.embed("hello")
    v2 = fake_embedder.embed("world")
    assert v1 != v2


# ── OpenAIEmbedder (mock API) ─────────────────────────────────────────────────


@pytest.fixture
def mock_openai_embedder():
    """Mock toàn bộ openai module — không cần API key."""
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3, 0.4]),
    ]

    with patch.dict("sys.modules", {"openai": MagicMock()}):
        import sys

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = fake_response
        sys.modules["openai"].OpenAI.return_value = mock_client
        embedder = OpenAIEmbedder()
        embedder._client = mock_client
        yield embedder, mock_client


def test_openai_embed_calls_api(mock_openai_embedder):
    embedder, mock_client = mock_openai_embedder
    result = embedder.embed("test text")
    mock_client.embeddings.create.assert_called_once()
    assert result == [0.1, 0.2, 0.3, 0.4]


def test_openai_embed_batch(mock_openai_embedder):
    embedder, mock_client = mock_openai_embedder
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3, 0.4]),
        MagicMock(embedding=[0.5, 0.6, 0.7, 0.8]),
    ]
    mock_client.embeddings.create.return_value = fake_response
    result = embedder.embed_batch(["text1", "text2"])
    assert len(result) == 2


# ── @retry decorator ──────────────────────────────────────────────────────────


def test_retry_succeeds_first_try():
    call_count = 0

    @retry(max_attempts=3, delay=0)
    def always_ok():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = always_ok()
    assert result == "ok"
    assert call_count == 1


def test_retry_retries_on_failure():
    call_count = 0

    @retry(max_attempts=3, delay=0)
    def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("network error")
        return "ok"

    result = fail_twice()
    assert result == "ok"
    assert call_count == 3


def test_retry_raises_after_max_attempts():
    @retry(max_attempts=2, delay=0)
    def always_fail():
        raise ValueError("always fails")

    with pytest.raises(ValueError, match="always fails"):
        always_fail()


# ── EmbedderFactory ───────────────────────────────────────────────────────────


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        EmbedderFactory.create("nonexistent")


def test_factory_register_and_create():
    @EmbedderFactory.register("fake")
    class _Fake(BaseEmbedder):
        def embed(self, text):
            return [0.0]

        def embed_batch(self, texts):
            return [[0.0]] * len(texts)

        @property
        def dimension(self):
            return 1

    embedder = EmbedderFactory.create("fake")
    assert isinstance(embedder, BaseEmbedder)

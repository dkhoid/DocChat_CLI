import functools
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

# ── Decorators ────────────────────────────────────────────────────────────────


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry với exponential backoff — dùng cho API calls."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = delay * (2**attempt)
                    print(f"  [retry] attempt {attempt + 1} failed: {e} — wait {wait:.1f}s")
                    time.sleep(wait)

        return wrapper

    return decorator


# ── Base class ────────────────────────────────────────────────────────────────


class BaseEmbedder(ABC):
    """Interface chung cho mọi embedding model."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed một đoạn text, trả về vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed nhiều text cùng lúc — hiệu quả hơn gọi embed() nhiều lần."""
        ...

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        """Convenience method — dùng embed_batch bên dưới."""
        return self.embed_batch(docs)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều của vector output."""
        ...


# ── Implementations ───────────────────────────────────────────────────────────


class OpenAIEmbedder(BaseEmbedder):
    """Dùng OpenAI text-embedding API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            import openai
        except ImportError:
            raise ImportError("Cần cài openai: uv add openai")
        self.model = model
        self._client = openai.OpenAI()
        self._dim = 1536

    @retry(max_attempts=3, delay=1.0)
    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding

    @retry(max_attempts=3, delay=1.0)
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(input=texts, model=self.model)
        return [d.embedding for d in resp.data]

    @property
    def dimension(self) -> int:
        return self._dim


class LocalEmbedder(BaseEmbedder):
    """Dùng sentence-transformers — không cần API key."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Cần cài sentence-transformers: uv add sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ── Factory ───────────────────────────────────────────────────────────────────


class EmbedderFactory:
    """Tạo embedder từ tên provider — không hardcode trong code chính."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator để đăng ký provider mới."""

        def decorator(klass: type) -> type:
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def create(cls, provider: str, **kwargs) -> BaseEmbedder:
        klass = cls._registry.get(provider)
        if not klass:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
        return klass(**kwargs)


EmbedderFactory.register("openai")(OpenAIEmbedder)
EmbedderFactory.register("local")(LocalEmbedder)

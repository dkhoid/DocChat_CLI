import time
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections.abc import AsyncIterator

from docchat.store import SimpleVectorStore
import openai


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: str = (
        "Bạn là trợ lý hỏi đáp tài liệu. "
        "Chỉ trả lời dựa trên nội dung tài liệu được cung cấp. "
        "Nếu tài liệu không có thông tin, hãy nói rõ là không tìm thấy. "
        "Trả lời bằng ngôn ngữ của câu hỏi."
    )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class SessionStats:
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    def report(self) -> str:
        avg = self.total_time / self.call_count if self.call_count else 0
        return (
                f"Calls: {self.call_count} | "
                f"Tokens in/out: {self.total_input_tokens}/{self.total_output_tokens} | "
                f"Avg latency: {avg:.2f}s"
                + (f" | Errors: {len(self.errors)}" if self.errors else "")
        )


# ── LLMSession context manager ────────────────────────────────────────────────

class LLMSession:
    """
    Quản lý một phiên gọi LLM.
    Tự động log stats và đóng client khi thoát.

    Dùng:
        with LLMSession(config) as session:
            async for token in session.stream("câu hỏi", store):
                print(token, end="", flush=True)
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self.stats = SessionStats()
        self._client = None

    def __enter__(self) -> "LLMSession":
        try:
            import openai
            self._client = openai.OpenAI()
        except ImportError:
            raise ImportError("Cần cài openai: uv add openai")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        print(f"\n[session] {self.stats.report()}")
        self._client = None
        return False  # không suppress exceptions

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, context_chunks: list) -> str:
        if not context_chunks:
            return query

        context = "\n\n---\n\n".join(
            f"[{r.chunk.source}]\n{r.chunk.text}"
            for r in context_chunks
        )
        return (
            f"Tài liệu tham khảo:\n\n{context}\n\n"
            f"---\n\nCâu hỏi: {query}"
        )

    # ── Stream ────────────────────────────────────────────────────────────────

    async def stream(
            self,
            query: str,
            store: SimpleVectorStore,
            k: int = 5,
    ) -> AsyncIterator[str]:
        """
        Tìm context từ store, gọi LLM, yield từng token.

        Args:
            query: Câu hỏi của người dùng.
            store: VectorStore đã được index.
            k: Số chunk context lấy ra.
        """
        results = store.search(query, k=k)
        prompt = self._build_prompt(query, results)

        t_start = time.perf_counter()
        self.stats.call_count += 1

        try:
            # Anthropic streaming chạy sync — wrap vào thread để không block event loop
            full_response = await asyncio.to_thread(
                self._stream_sync, prompt
            )
            elapsed = time.perf_counter() - t_start
            self.stats.total_time += elapsed

            for token in full_response:
                yield token

        except Exception as e:
            self.stats.errors.append(str(e))
            raise

    def _stream_sync(self, prompt: str) -> list[str]:
        """Gọi OpenAI streaming API (sync) -Chạy trong thread pool."""
        tokens = []
        stream = self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            stream_options={"include_usage": True}
        )
        
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                tokens.append(text)
                print(text, end="", flush=True)
            if hasattr(chunk, 'usage') and chunk.usage:
                self.stats.total_input_tokens += chunk.usage.prompt_tokens
                self.stats.total_output_tokens += chunk.usage.completion_tokens

        return tokens


    def complete(self, query: str, store: SimpleVectorStore, k: int = 5) -> str:
        """Phiên bản sync của stream — trả về full string."""
        results = store.search(query, k=k)
        prompt = self._build_prompt(query, results)

        t_start = time.perf_counter()
        self.stats.call_count += 1

        try:
            msg = self._client.chat.completions.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            elapsed = time.perf_counter() - t_start
            self.stats.total_time += elapsed
            if msg.usage:
                self.stats.total_input_tokens += msg.usage.prompt_tokens
                self.stats.total_output_tokens += msg.usage.completion_tokens
            return msg.choices[0].message.content or ""

        except Exception as e:
            self.stats.errors.append(str(e))
            raise


# ── Convenience function ──────────────────────────────────────────────────────

async def ask(
        query: str,
        store: SimpleVectorStore,
        config: LLMConfig | None = None,
        k: int = 5,
) -> str:
    """
    Hỏi một câu, nhận full answer — không stream ra terminal.
    Tiện cho pipeline và test.
    """
    with LLMSession(config) as session:
        tokens = []
        async for token in session.stream(query, store, k=k):
            tokens.append(token)
        return "".join(tokens)

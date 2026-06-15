import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import openai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

from docchat.logger import get_logger
from docchat.observability import create_generation, create_trace, end_generation
from docchat.store import BaseStore

logger = get_logger(__name__)


# ── Pricing table (USD per 1M tokens) ────────────────────────────────────────

_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Tinh cost bang USD tu token count."""
    price = _PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def _default_model(provider: str) -> str:
    return "claude-3-5-haiku-latest" if provider == "anthropic" else "gpt-4o-mini"


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    provider: Literal["openai", "anthropic"] = "openai"
    max_tokens: int = 1024
    max_output_tokens: int = 512
    max_input_tokens: int = 8_000
    temperature: float = 0.7
    max_history_tokens: int = 4000
    max_history_turns: int = 5
    min_relevance_score: float = -1.0
    system_prompt: str = (
        "Ban la tro ly hoi dap tai lieu. "
        "Chi tra loi dua tren noi dung tai lieu duoc cung cap. "
        "Neu tai lieu khong co thong tin, hay noi ro la khong tim thay. "
        "Tra loi bang ngon ngu cua cau hoi."
    )

    def __post_init__(self) -> None:
        if not self.model:
            self.model = _default_model(self.provider)


# ── Stats ─────────────────────────────────────────────────────────────────────


@dataclass
class SessionStats:
    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_time: float = 0.0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Cong don token va tinh cost."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.cost_usd += _calc_cost(model, input_tokens, output_tokens)

    def report(self) -> str:
        avg = self.total_time / self.call_count if self.call_count else 0
        parts = [
            f"Calls: {self.call_count}",
            f"Tokens in/out: {self.total_input_tokens}/{self.total_output_tokens}",
            f"Cost: ${self.cost_usd:.6f}",
            f"Avg latency: {avg:.2f}s",
        ]
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        return " | ".join(parts)


_RETRYABLE_EXCEPTIONS: list[type[Exception]] = [
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    TimeoutError,
    ConnectionError,
]
if anthropic is not None:
    for exc_name in ("RateLimitError", "APIConnectionError", "APITimeoutError", "APIStatusError"):
        exc_type = getattr(anthropic, exc_name, None)
        if isinstance(exc_type, type) and issubclass(exc_type, Exception):
            _RETRYABLE_EXCEPTIONS.append(exc_type)


_llm_retry = retry(
    retry=retry_if_exception_type(tuple(_RETRYABLE_EXCEPTIONS)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


# ── LLMSession context manager ────────────────────────────────────────────────


class LLMSession:
    """Quan ly 1 phien goi LLM voi retry, cost tracking va history ngan han."""

    def __init__(self, config: LLMConfig | None = None):
        import threading

        self.config = config or LLMConfig()
        self.stats = SessionStats()
        self.history: list[dict[str, str]] = []
        self._client: openai.OpenAI | None = None
        self._anthropic_client: Any | None = None
        self._shutdown_event = threading.Event()

    def __enter__(self) -> "LLMSession":
        if self.config.provider == "anthropic":
            if anthropic is None:
                raise ImportError("Can cai anthropic: uv add anthropic")
            self._anthropic_client = (
                anthropic.Anthropic(api_key=self.config.api_key)
                if self.config.api_key
                else anthropic.Anthropic()
            )
        else:
            self._client = (
                openai.OpenAI(api_key=self.config.api_key)
                if self.config.api_key
                else openai.OpenAI()
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        logger.info("session_closed", stats=self.stats.report())
        self._client = None
        self._anthropic_client = None
        return False

    # ── History management ────────────────────────────────────────────────────

    def add_to_history(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self) -> None:
        # Gioi han turn truoc, roi trim token neu can.
        max_messages = max(2, self.config.max_history_turns * 2)
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        from docchat.prompt_manager import get_prompt_manager

        pm = get_prompt_manager()
        while len(self.history) > 2:
            total_tokens = pm.count_messages_tokens(self.history, model=self.config.model)
            if total_tokens <= self.config.max_history_tokens:
                break
            self.history.pop(0)

    def clear_history(self) -> None:
        self.history.clear()

    def _build_messages(
        self,
        query: str,
        context_chunks: list,
        use_history: bool = False,
    ) -> list[dict[str, str]]:
        """Build messages voi token budget dung cho RAG context."""
        from docchat.prompt_manager import get_prompt_manager

        pm = get_prompt_manager()
        history = self.history if use_history else []

        if not context_chunks:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self.config.system_prompt}
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": query})
            return messages

        filtered_chunks = [r for r in context_chunks if r.score >= self.config.min_relevance_score]
        context_text = "\n\n---\n\n".join(
            f"[{r.chunk.source}]\n{r.chunk.text}" for r in filtered_chunks
        )

        rendered_empty = pm.render("qa_rag", query=query, context="")
        base_messages: list[dict[str, str]] = [
            {"role": "system", "content": rendered_empty.get("system", self.config.system_prompt)}
        ]
        base_messages.extend(history)
        base_messages.append({"role": "user", "content": rendered_empty.get("user", query)})

        input_budget = max(0, self.config.max_input_tokens - self.config.max_output_tokens)
        base_tokens = pm.count_messages_tokens(base_messages, model=self.config.model)
        context_budget = max(0, input_budget - base_tokens)
        context_text = pm.trim_to_budget(
            context_text, budget=context_budget, model=self.config.model
        )

        rendered = pm.render("qa_rag", query=query, context=context_text)
        messages = [
            {"role": "system", "content": rendered.get("system", self.config.system_prompt)}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": rendered.get("user", query)})
        return messages

    # Legacy method for backward compatibility
    def _build_prompt(self, query: str, context_chunks: list) -> str:
        if not context_chunks:
            return query
        context = "\n\n---\n\n".join(f"[{r.chunk.source}]\n{r.chunk.text}" for r in context_chunks)
        return f"Tai lieu tham khao:\n\n{context}\n\n---\n\nCau hoi: {query}"

    # ── Stream ────────────────────────────────────────────────────────────────

    async def stream(
        self,
        query: str,
        store: BaseStore,
        k: int = 5,
        use_history: bool = False,
    ) -> AsyncIterator[str]:
        results = await asyncio.to_thread(store.search, query, k=k)
        messages = self._build_messages(query, results, use_history=use_history)

        t_start = time.perf_counter()
        self.stats.call_count += 1

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def producer():
            try:
                if self.config.provider == "anthropic":
                    for token in self._stream_sync_anthropic(messages):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                else:
                    for token in self._stream_sync(messages):
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", e))

        import threading

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        tokens = []
        try:
            while True:
                msg_type, data = await queue.get()
                if msg_type == "token":
                    tokens.append(data)
                    yield data
                elif msg_type == "done":
                    break
                elif msg_type == "error":
                    raise data
        except Exception as e:
            self.stats.errors.append(str(e))
            raise
        finally:
            self._shutdown_event.set()
            self.stats.total_time += time.perf_counter() - t_start
            if use_history:
                self.add_to_history("user", query)
                self.add_to_history("assistant", "".join(tokens))

    @_llm_retry
    def _create_openai_stream(self, messages):
        if self._client is None:
            raise RuntimeError("OpenAI client chua duoc khoi tao")
        return self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

    def _stream_sync(self, messages: list[dict[str, str]]):
        """Call provider streaming API (sync) va tra ve generator token."""
        stream = self._create_openai_stream(messages)
        for chunk in stream:
            if self._shutdown_event.is_set():
                break
            if len(chunk.choices) > 0 and getattr(chunk.choices[0].delta, "content", None):
                yield chunk.choices[0].delta.content
            usage = getattr(chunk, "usage", None)
            if usage:
                self.stats.add_usage(
                    self.config.model,
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                )

    @_llm_retry
    def _create_anthropic_stream(self, system_prompt, anthropic_messages):
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client chua duoc khoi tao")
        return self._anthropic_client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=anthropic_messages,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            stream=True,
        )

    def _stream_sync_anthropic(self, messages: list[dict[str, str]]):
        system_prompt = ""
        anthropic_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        stream = self._create_anthropic_stream(system_prompt, anthropic_messages)
        input_tokens = 0
        output_tokens = 0

        for event in stream:
            if self._shutdown_event.is_set():
                break
            event_type = getattr(event, "type", "")
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                text = getattr(delta, "text", "") if delta else ""
                if text:
                    yield text
            elif event_type in {"message_delta", "message_stop"}:
                usage = getattr(event, "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", input_tokens)
                    output_tokens = getattr(usage, "output_tokens", output_tokens)

        if input_tokens or output_tokens:
            self.stats.add_usage(self.config.model, input_tokens, output_tokens)

    # ── Complete (sync) ───────────────────────────────────────────────────────

    def complete(
        self,
        query: str,
        store: BaseStore,
        k: int = 5,
        use_history: bool = False,
    ) -> str:
        results = store.search(query, k=k)
        messages = self._build_messages(query, results, use_history=use_history)

        t_start = time.perf_counter()
        self.stats.call_count += 1

        trace = create_trace(name="llm_complete", input_data={"query": query, "k": k})
        generation = create_generation(
            trace=trace,
            name="chat_completion",
            model=self.config.model,
            input_messages=messages,
            model_parameters={
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
            },
        )

        try:
            response = self._complete_sync(messages)
            latency = time.perf_counter() - t_start
            self.stats.total_time += latency

            end_generation(
                generation,
                output=response,
                usage={
                    "input": self.stats.total_input_tokens,
                    "output": self.stats.total_output_tokens,
                },
            )
            logger.info(
                "llm_complete",
                model=self.config.model,
                latency_s=round(latency, 3),
                input_tokens=self.stats.total_input_tokens,
                output_tokens=self.stats.total_output_tokens,
            )

            if use_history:
                self.add_to_history("user", query)
                self.add_to_history("assistant", response)

            return response

        except Exception as e:
            self.stats.errors.append(str(e))
            end_generation(generation, output=str(e), level="ERROR")
            raise

    @_llm_retry
    def _complete_sync(self, messages: list[dict[str, str]]) -> str:
        if self.config.provider == "anthropic":
            return self._complete_sync_anthropic(messages)

        if self._client is None:
            raise RuntimeError("OpenAI client chua duoc khoi tao")

        msg = self._client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            messages=messages,
        )

        usage = getattr(msg, "usage", None)
        if usage:
            self.stats.add_usage(
                self.config.model,
                usage.prompt_tokens,
                usage.completion_tokens,
            )
        return msg.choices[0].message.content or ""

    def _complete_sync_anthropic(self, messages: list[dict[str, str]]) -> str:
        if self._anthropic_client is None:
            raise RuntimeError("Anthropic client chua duoc khoi tao")

        system_prompt = ""
        anthropic_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

        res = self._anthropic_client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=anthropic_messages,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
        )

        usage = getattr(res, "usage", None)
        if usage:
            self.stats.add_usage(
                self.config.model,
                getattr(usage, "input_tokens", 0),
                getattr(usage, "output_tokens", 0),
            )

        content_blocks = getattr(res, "content", []) or []
        text_parts = [getattr(block, "text", "") for block in content_blocks]
        return "".join(text_parts).strip()


# ── Convenience function ──────────────────────────────────────────────────────


async def ask(
    query: str,
    store: BaseStore,
    config: LLMConfig | None = None,
    k: int = 5,
) -> str:
    with LLMSession(config) as session:
        tokens: list[str] = []
        async for token in session.stream(query, store, k=k):
            tokens.append(token)
        return "".join(tokens)

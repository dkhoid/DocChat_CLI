import asyncio
from unittest.mock import MagicMock, patch

import openai
import pytest

from docchat.core.chunker import Chunk
from docchat.core.prompt_manager import get_prompt_manager
from docchat.llm.session import LLMConfig, LLMSession, SessionStats, ask
from docchat.storage.store import SearchResult, SimpleVectorStore
from tests.test_embedder import FakeEmbedder

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_store(*texts: str) -> SimpleVectorStore:
    """Tạo store đã populated với FakeEmbedder."""
    embedder = FakeEmbedder(dim=4)
    store = SimpleVectorStore(embedder=embedder)
    chunks = [
        Chunk(text=t, source=f"doc{i}.txt", index=0, chunk_num=i) for i, t in enumerate(texts)
    ]
    store.add(chunks)
    return store


def make_mock_client(response_text: str = "Đây là câu trả lời"):
    """Tạo OpenAI client mock trả về response_text."""
    mock_client = MagicMock()

    # Mock cho complete()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock()]
    mock_msg.choices[0].message.content = response_text
    mock_msg.choices[0].message.refusal = None
    mock_msg.usage.prompt_tokens = 100
    mock_msg.usage.completion_tokens = 50
    mock_client.chat.completions.create.return_value = mock_msg

    # Mock cho stream()
    mock_stream = []
    for chunk_text in ["y ", "l ", "cu ", "tr ", "li."]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = chunk_text
        chunk.usage = None
        mock_stream.append(chunk)

    last_chunk = MagicMock()
    last_chunk.choices = []
    last_chunk.usage.prompt_tokens = 100
    last_chunk.usage.completion_tokens = 50
    mock_stream.append(last_chunk)

    # trả về 1 iterator thay vì stream ctx
    mock_client.chat.completions.create.return_value = iter(mock_stream)

    # Note: complete uses object not stream, stream uses iterator.
    # To properly mock both depending on kwargs, we can use a side_effect
    def create_mock(*args, **kwargs):
        if kwargs.get("stream"):
            return iter(mock_stream)
        else:
            return mock_msg

    mock_client.chat.completions.create.side_effect = create_mock

    return mock_client


# ── LLMConfig ─────────────────────────────────────────────────────────────────


def test_default_config():
    config = LLMConfig()
    assert config.model == "gpt-4o-mini"
    assert config.max_tokens == 1024
    assert 0.0 <= config.temperature <= 1.0


def test_custom_config():
    config = LLMConfig(model="gpt-4o", max_tokens=2048)
    assert config.model == "gpt-4o"
    assert config.max_tokens == 2048


def test_config_defaults_model_from_provider():
    cfg = LLMConfig(provider="anthropic", model="")
    assert cfg.model == "claude-3-5-haiku-latest"


# ── SessionStats ──────────────────────────────────────────────────────────────


def test_stats_report_format():
    stats = SessionStats(call_count=3, total_time=6.0)
    report = stats.report()
    assert "Calls: 3" in report
    assert "2.00s" in report


def test_stats_report_no_errors():
    stats = SessionStats()
    assert "Error" not in stats.report()


def test_stats_report_with_errors():
    stats = SessionStats(errors=["timeout", "rate limit"])
    assert "Errors: 2" in stats.report()


# ── LLMSession context manager ────────────────────────────────────────────────


def test_session_enters_and_exits():
    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as session:
            assert session._client is not None
        assert session._client is None  # cleanup


def test_session_openai_constructor_failure():
    # Kiểm tra rằng lỗi khởi tạo OpenAI() được propagate đúng.
    with patch(
        "docchat.llm.session.openai.OpenAI",
        side_effect=openai.APIConnectionError(request=MagicMock()),
    ):
        with pytest.raises(openai.APIConnectionError):
            with LLMSession():
                pass


def test_session_stats_reset_each_session():
    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession():
            pass
        with LLMSession() as s2:
            assert s2.stats.call_count == 0


# ── _build_prompt ─────────────────────────────────────────────────────────────


def test_build_prompt_no_context():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    result = session._build_prompt("câu hỏi?", [])
    assert result == "câu hỏi?"


def test_build_prompt_with_context():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()

    chunk = Chunk(text="nội dung tài liệu", source="a.txt", index=0, chunk_num=0)
    results = [SearchResult(chunk=chunk, score=0.9)]
    prompt = session._build_prompt("câu hỏi?", results)

    assert "nội dung tài liệu" in prompt
    assert "a.txt" in prompt
    assert "câu hỏi?" in prompt


def test_build_prompt_multiple_chunks():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()

    results = [
        SearchResult(chunk=Chunk("chunk A", "a.txt", 0, 0), score=0.9),
        SearchResult(chunk=Chunk("chunk B", "b.txt", 0, 0), score=0.8),
    ]
    prompt = session._build_prompt("query", results)
    assert "chunk A" in prompt
    assert "chunk B" in prompt


# ── complete() ────────────────────────────────────────────────────────────────


def test_complete_returns_string():
    store = make_store("Python là ngôn ngữ lập trình.")
    mock_client = make_mock_client("Câu trả lời đây.")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            result = session.complete("Python là gì?", store)

    assert isinstance(result, str)
    assert result == "Câu trả lời đây."


def test_complete_updates_stats():
    store = make_store("nội dung")
    mock_client = make_mock_client()

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            session.complete("query", store)
            assert session.stats.call_count == 1
            assert session.stats.total_input_tokens == 100
            assert session.stats.total_output_tokens == 50


def test_complete_records_error_on_exception():
    store = make_store("nội dung")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ConnectionError("network down")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            with pytest.raises(ConnectionError):
                session.complete("query", store)
            assert len(session.stats.errors) == 1


def test_complete_multiple_calls():
    store = make_store("nội dung A", "nội dung B")
    mock_client = make_mock_client()

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            session.complete("câu hỏi 1", store)
            session.complete("câu hỏi 2", store)
            assert session.stats.call_count == 2


# ── ask() convenience function ────────────────────────────────────────────────


def test_ask_returns_string():
    store = make_store("ti li‡u test")
    mock_client = make_mock_client("kt qu")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client

        # Patch _stream_sync để trả về list token thay vì gọi API thật
        with patch.object(LLMSession, "_stream_sync", return_value=["kết ", "quả"]):
            result = asyncio.run(ask("query", store))

    assert isinstance(result, str)


def test_ask_empty_store():
    store = SimpleVectorStore(embedder=FakeEmbedder(dim=4))
    mock_client = make_mock_client("không có thng tin")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with patch.object(LLMSession, "_stream_sync", return_value=["không ", "có ", "thông tin"]):
            result = asyncio.run(ask("query?", store))

    assert isinstance(result, str)


# ── SessionStats.add_usage + cost ────────────────────────────────────────────


def test_add_usage_accumulates_tokens():
    stats = SessionStats()
    stats.add_usage("gpt-4o-mini", input_tokens=100, output_tokens=50)
    assert stats.total_input_tokens == 100
    assert stats.total_output_tokens == 50


def test_add_usage_calculates_cost_nonzero():
    stats = SessionStats()
    stats.add_usage("gpt-4o-mini", input_tokens=1_000_000, output_tokens=0)
    # gpt-4o-mini input: $0.15 / 1M tokens
    assert abs(stats.cost_usd - 0.15) < 0.001


def test_add_usage_unknown_model_zero_cost():
    stats = SessionStats()
    stats.add_usage("unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert stats.cost_usd == 0.0


def test_report_includes_cost():
    stats = SessionStats(call_count=1, total_time=1.0)
    stats.add_usage("gpt-4o-mini", input_tokens=100, output_tokens=50)
    report = stats.report()
    assert "Cost:" in report
    assert "$" in report


def test_report_multiple_calls_accumulate():
    stats = SessionStats()
    stats.add_usage("gpt-4o-mini", input_tokens=500, output_tokens=200)
    stats.add_usage("gpt-4o-mini", input_tokens=500, output_tokens=200)
    assert stats.total_input_tokens == 1000
    assert stats.total_output_tokens == 400


# ── Conversation history ──────────────────────────────────────────────────────


def test_add_to_history_appends_turns():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    session.history = []
    session.add_to_history("user", "Xin chào")
    session.add_to_history("assistant", "Chào bạn!")
    assert len(session.history) == 2
    assert session.history[0]["role"] == "user"
    assert session.history[1]["role"] == "assistant"


def test_clear_history():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    session.history = [{"role": "user", "content": "hi"}]
    session.clear_history()
    assert session.history == []


def test_history_not_included_when_disabled():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    session.history = [{"role": "user", "content": "câu cũ"}]
    messages = session._build_messages("câu mới", [], use_history=False)
    contents = [m["content"] for m in messages]
    assert not any("câu cũ" in c for c in contents)


def test_history_included_when_enabled():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    session.history = [{"role": "user", "content": "câu cũ"}]
    messages = session._build_messages("câu mới", [], use_history=True)
    contents = [m["content"] for m in messages]
    assert any("câu cũ" in c for c in contents)


# ── _build_messages ───────────────────────────────────────────────────────────


def test_build_messages_has_system_first():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    session.history = []
    messages = session._build_messages("query", [])
    assert messages[0]["role"] == "system"


def test_build_messages_last_is_user():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig()
    session.history = []
    messages = session._build_messages("câu hỏi", [])
    assert messages[-1]["role"] == "user"
    assert "câu hỏi" in messages[-1]["content"]


# ── Retry logic ───────────────────────────────────────────────────────────────


def test_complete_retries_on_rate_limit():
    """_complete_sync phải retry khi gặp RateLimitError."""
    import openai

    store = make_store("nội dung")
    call_count = 0

    def failing_then_ok(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise openai.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body={},
            )
        # Lần 3: thành công
        mock_msg = MagicMock()
        mock_msg.choices = [MagicMock()]
        mock_msg.choices[0].message.content = "ok"
        mock_msg.choices[0].message.refusal = None
        mock_msg.usage.prompt_tokens = 10
        mock_msg.usage.completion_tokens = 5
        return mock_msg

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = failing_then_ok
        mock_openai_class.return_value = mock_client

        with LLMSession() as session:
            result = session.complete("query", store)

    assert call_count == 3
    assert result == "ok"


def test_complete_raises_after_max_retries():
    """Sau 3 lần retry, phải raise exception."""
    import openai

    store = make_store("nội dung")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="always rate limited",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        mock_openai_class.return_value = mock_client

        with LLMSession() as session:
            with pytest.raises(openai.RateLimitError):
                session.complete("query", store)


def test_history_trim_to_max_turns():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig(max_history_turns=2)
    session.history = []

    for i in range(6):
        session.add_to_history("user", f"q{i}")
        session.add_to_history("assistant", f"a{i}")

    assert len(session.history) == 4
    assert session.history[0]["content"] == "q4"


def test_build_messages_respects_context_budget():
    session = LLMSession.__new__(LLMSession)
    session.config = LLMConfig(max_input_tokens=160, max_output_tokens=120)
    session.history = []

    long_text = " ".join(["python"] * 600)
    results = [SearchResult(chunk=Chunk(long_text, "big.txt", 0, 0), score=0.9)]

    messages = session._build_messages("query", results)
    pm = get_prompt_manager()
    token_count = pm.count_messages_tokens(messages, model=session.config.model)
    assert token_count <= session.config.max_input_tokens + 20


def test_complete_anthropic_provider():
    store = make_store("nội dung")
    mock_anthropic = MagicMock()

    block = MagicMock()
    block.text = "Claude trả lời"
    resp = MagicMock()
    resp.content = [block]
    resp.usage.input_tokens = 12
    resp.usage.output_tokens = 7
    mock_anthropic.messages.create.return_value = resp

    with patch("docchat.llm.session.anthropic.Anthropic", return_value=mock_anthropic):
        cfg = LLMConfig(provider="anthropic", model="claude-3-5-haiku-latest")
        with LLMSession(cfg) as session:
            result = session.complete("query", store)

    assert result == "Claude trả lời"
    assert session.stats.total_input_tokens == 12
    assert session.stats.total_output_tokens == 7


# ── Async streaming tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_returns_tokens():
    """stream() phải yield từng token."""
    store = make_store("tài liệu test")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as session:
            with patch.object(session, "_stream_sync", return_value=["xin ", "chào"]):
                tokens = []
                async for token in session.stream("hello", store):
                    tokens.append(token)

    assert tokens == ["xin ", "chào"]


@pytest.mark.asyncio
async def test_stream_empty_store():
    """stream() với store rỗng vẫn hoạt động."""
    empty_store = SimpleVectorStore(embedder=FakeEmbedder(dim=4))

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as session:
            with patch.object(session, "_stream_sync", return_value=["không ", "có"]):
                tokens = []
                async for token in session.stream("query", empty_store):
                    tokens.append(token)

    assert "".join(tokens) == "không có"


@pytest.mark.asyncio
async def test_stream_updates_stats():
    """stream() phải tăng call_count."""
    store = make_store("nội dung")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as session:
            with patch.object(session, "_stream_sync", return_value=["ok"]):
                async for _ in session.stream("query", store):
                    pass
            assert session.stats.call_count == 1


@pytest.mark.asyncio
async def test_stream_with_history():
    """stream() với use_history=True phải lưu vào history."""
    store = make_store("data")

    with patch("docchat.llm.session.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as session:
            with patch.object(session, "_stream_sync", return_value=["trả ", "lời"]):
                async for _ in session.stream("câu hỏi", store, use_history=True):
                    pass
            assert len(session.history) == 2
            assert session.history[0]["role"] == "user"
            assert session.history[0]["content"] == "câu hỏi"
            assert session.history[1]["role"] == "assistant"
            assert session.history[1]["content"] == "trả lời"

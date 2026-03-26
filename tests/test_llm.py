import asyncio
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from docchat.chunker import Chunk
from docchat.llm import LLMConfig, LLMSession, SessionStats, ask
from docchat.store import SimpleVectorStore, SearchResult
from tests.test_embedder import FakeEmbedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_store(*texts: str) -> SimpleVectorStore:
    """Tạo store đã populated với FakeEmbedder."""
    embedder = FakeEmbedder(dim=4)
    store = SimpleVectorStore(embedder=embedder)
    chunks = [
        Chunk(text=t, source=f"doc{i}.txt", index=0, chunk_num=i)
        for i, t in enumerate(texts)
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
        if kwargs.get('stream'):
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
    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as session:
            assert session._client is not None
        assert session._client is None  # cleanup


def test_session_missing_openai():
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="openai"):
            with LLMSession():
                pass


def test_session_stats_reset_each_session():
    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = MagicMock()
        with LLMSession() as s1:
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

    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            result = session.complete("Python là gì?", store)

    assert isinstance(result, str)
    assert result == "Câu trả lời đây."


def test_complete_updates_stats():
    store = make_store("nội dung")
    mock_client = make_mock_client()

    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
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

    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            with pytest.raises(ConnectionError):
                session.complete("query", store)
            assert len(session.stats.errors) == 1


def test_complete_multiple_calls():
    store = make_store("nội dung A", "nội dung B")
    mock_client = make_mock_client()

    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with LLMSession() as session:
            session.complete("câu hỏi 1", store)
            session.complete("câu hỏi 2", store)
            assert session.stats.call_count == 2


# ── ask() convenience function ────────────────────────────────────────────────

def test_ask_returns_string():
    store = make_store("ti li‡u test")
    mock_client = make_mock_client("kt qu")

    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client

        # Patch _stream_sync để trả về list token thay vì gọi API thật
        with patch.object(LLMSession, "_stream_sync", return_value=["kết ", "quả"]):
            result = asyncio.run(ask("query", store))

    assert isinstance(result, str)


def test_ask_empty_store():
    store = SimpleVectorStore(embedder=FakeEmbedder(dim=4))
    mock_client = make_mock_client("không có thng tin")

    with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
        mock_openai_class.return_value = mock_client
        with patch.object(LLMSession, "_stream_sync", return_value=["không ", "có ", "thông tin"]):
            result = asyncio.run(ask("query?", store))

    assert isinstance(result, str)
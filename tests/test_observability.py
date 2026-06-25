"""Tests for docchat.observability — Langfuse integration (graceful no-op when unconfigured)."""

import os
from unittest.mock import MagicMock, patch

import pytest

import docchat.infrastructure.observability as obs_module
from docchat.infrastructure.observability import (
    create_generation,
    create_trace,
    end_generation,
    flush,
    get_langfuse,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before every test."""
    obs_module._langfuse_client = None
    obs_module._initialized = False
    yield
    obs_module._langfuse_client = None
    obs_module._initialized = False


# ── get_langfuse ──────────────────────────────────────────────────────────────


def test_get_langfuse_returns_none_when_keys_missing():
    """No credentials → returns None, no import of langfuse."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    }
    with patch.dict(os.environ, env, clear=True):
        result = get_langfuse()
    assert result is None


def test_get_langfuse_returns_none_when_only_public_key_set():
    env = {"LANGFUSE_PUBLIC_KEY": "pk-test"}
    with patch.dict(os.environ, env, clear=True):
        result = get_langfuse()
    assert result is None


def test_get_langfuse_returns_none_when_only_secret_key_set():
    env = {"LANGFUSE_SECRET_KEY": "sk-test"}
    with patch.dict(os.environ, env, clear=True):
        result = get_langfuse()
    assert result is None


def test_get_langfuse_returns_client_when_both_keys_set():
    env = {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}
    mock_client = MagicMock()
    mock_langfuse_cls = MagicMock(return_value=mock_client)

    with patch.dict(os.environ, env, clear=True):
        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
            result = get_langfuse()

    assert result is mock_client


def test_get_langfuse_uses_custom_host():
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
        "LANGFUSE_BASE_URL": "https://my.langfuse.com",
    }
    mock_client = MagicMock()
    mock_langfuse_cls = MagicMock(return_value=mock_client)

    with patch.dict(os.environ, env, clear=True):
        with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=mock_langfuse_cls)}):
            get_langfuse()

    mock_langfuse_cls.assert_called_once_with(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://my.langfuse.com",
    )


def test_get_langfuse_is_cached_after_first_call():
    """Singleton: second call must NOT re-initialize."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    }
    with patch.dict(os.environ, env, clear=True):
        first = get_langfuse()
        second = get_langfuse()

    assert first is second  # both None, but only one initialization


def test_get_langfuse_returns_none_on_import_error():
    """If langfuse package raises on import → gracefully return None."""
    env = {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test"}

    with patch.dict(os.environ, env, clear=True):
        with patch.dict("sys.modules", {"langfuse": None}):
            result = get_langfuse()

    assert result is None


# ── create_trace ──────────────────────────────────────────────────────────────


def test_create_trace_returns_none_when_langfuse_disabled():
    with patch("docchat.infrastructure.observability.get_langfuse", return_value=None):
        result = create_trace(name="test-trace")
    assert result is None


def test_create_trace_returns_trace_object():
    mock_trace = MagicMock()
    mock_lf = MagicMock()
    mock_lf.trace.return_value = mock_trace

    with patch("docchat.infrastructure.observability.get_langfuse", return_value=mock_lf):
        result = create_trace(name="ask", input_data={"q": "hello"}, user_id="u1")

    assert result is mock_trace
    mock_lf.trace.assert_called_once_with(name="ask", input={"q": "hello"}, user_id="u1")


def test_create_trace_returns_none_on_exception():
    mock_lf = MagicMock()
    mock_lf.trace.side_effect = RuntimeError("network error")

    with patch("docchat.infrastructure.observability.get_langfuse", return_value=mock_lf):
        result = create_trace(name="ask")

    assert result is None


# ── create_generation ─────────────────────────────────────────────────────────


def test_create_generation_returns_none_when_trace_is_none():
    result = create_generation(trace=None, name="llm_call", model="gpt-4o-mini")
    assert result is None


def test_create_generation_returns_span():
    mock_span = MagicMock()
    mock_trace = MagicMock()
    mock_trace.generation.return_value = mock_span

    result = create_generation(
        trace=mock_trace,
        name="llm_call",
        model="gpt-4o-mini",
        input_messages=[{"role": "user", "content": "hi"}],
        model_parameters={"temperature": 0.7},
    )

    assert result is mock_span
    mock_trace.generation.assert_called_once()


def test_create_generation_returns_none_on_exception():
    mock_trace = MagicMock()
    mock_trace.generation.side_effect = RuntimeError("span error")

    result = create_generation(trace=mock_trace, name="llm_call", model="gpt-4o")

    assert result is None


# ── end_generation ────────────────────────────────────────────────────────────


def test_end_generation_noop_when_generation_is_none():
    """Should not raise when generation is None."""
    end_generation(None, output="answer", usage={"input": 10, "output": 5})


def test_end_generation_calls_end_on_span():
    mock_gen = MagicMock()
    end_generation(mock_gen, output="result", usage={"input": 5, "output": 3}, level="DEFAULT")
    mock_gen.end.assert_called_once_with(
        output="result", usage={"input": 5, "output": 3}, level="DEFAULT"
    )


def test_end_generation_swallows_exception():
    mock_gen = MagicMock()
    mock_gen.end.side_effect = RuntimeError("flush failed")
    end_generation(mock_gen, output="x")  # must not raise


# ── flush ─────────────────────────────────────────────────────────────────────


def test_flush_noop_when_langfuse_disabled():
    with patch("docchat.infrastructure.observability.get_langfuse", return_value=None):
        flush()  # must not raise


def test_flush_calls_lf_flush():
    mock_lf = MagicMock()
    with patch("docchat.infrastructure.observability.get_langfuse", return_value=mock_lf):
        flush()
    mock_lf.flush.assert_called_once()


def test_flush_swallows_exception():
    mock_lf = MagicMock()
    mock_lf.flush.side_effect = ConnectionError("unreachable")
    with patch("docchat.infrastructure.observability.get_langfuse", return_value=mock_lf):
        flush()  # must not raise

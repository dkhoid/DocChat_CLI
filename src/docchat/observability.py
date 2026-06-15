"""Langfuse LLM Observability integration.

Provides tracing for LLM calls (generation spans) with token usage,
cost tracking, and latency metrics.

Gracefully degrades to no-op when Langfuse credentials are not configured.

Usage:
    from docchat.observability import get_langfuse, create_generation

    lf = get_langfuse()
    trace = lf.trace(name="ask", input={"query": query}) if lf else None

    generation = create_generation(
        trace=trace, name="llm_call", model="gpt-4o-mini",
        input_messages=messages,
    )
    # ... call LLM ...
    if generation:
        generation.end(output=answer, usage={"input": in_tok, "output": out_tok})
"""

import os

from docchat.logger import get_logger

logger = get_logger(__name__)

_langfuse_client = None
_initialized = False


def get_langfuse():
    """Return the singleton Langfuse client, or None if not configured."""
    global _langfuse_client, _initialized

    if _initialized:
        return _langfuse_client

    _initialized = True

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.info(
            "langfuse_disabled",
            reason="LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set",
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("langfuse_enabled", host=host)
        return _langfuse_client
    except Exception as e:
        logger.error("langfuse_init_failed", error=str(e))
        return None


def create_trace(*, name: str, input_data: dict | None = None, user_id: str | None = None):
    """Create a Langfuse trace. Returns None if Langfuse is disabled."""
    lf = get_langfuse()
    if lf is None:
        return None

    try:
        return lf.trace(name=name, input=input_data, user_id=user_id)
    except Exception as e:
        logger.warning("langfuse_trace_failed", error=str(e))
        return None


def create_generation(
    *,
    trace,
    name: str,
    model: str,
    input_messages: list[dict] | None = None,
    model_parameters: dict | None = None,
):
    """Create a generation span on a trace. Returns None if trace is None."""
    if trace is None:
        return None

    try:
        return trace.generation(
            name=name,
            model=model,
            input=input_messages,
            model_parameters=model_parameters,
        )
    except Exception as e:
        logger.warning("langfuse_generation_failed", error=str(e))
        return None


def end_generation(
    generation,
    *,
    output: str | None = None,
    usage: dict | None = None,
    level: str = "DEFAULT",
):
    """End a generation span with output and usage data."""
    if generation is None:
        return

    try:
        generation.end(output=output, usage=usage, level=level)
    except Exception as e:
        logger.warning("langfuse_end_failed", error=str(e))


def flush():
    """Flush any pending Langfuse events. Call on shutdown."""
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass

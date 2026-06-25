"""Tests for docchat.server — uvicorn entry point configuration."""

import os
from unittest.mock import call, patch

import pytest

from docchat.interfaces.server import main


# ── main() env-var parsing ────────────────────────────────────────────────────


def test_server_main_uses_default_host_and_port():
    """Defaults: host=0.0.0.0, port=8000, reload=False."""
    env = {k: v for k, v in os.environ.items() if k not in ("DOCCHAT_HOST", "PORT", "DOCCHAT_PORT", "DOCCHAT_RELOAD")}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000
    assert kwargs["reload"] is False


def test_server_main_respects_docchat_host():
    """DOCCHAT_HOST overrides the bind address."""
    env = {"DOCCHAT_HOST": "127.0.0.1"}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs["host"] == "127.0.0.1"


def test_server_main_respects_port_env():
    """PORT (Railway-style) takes precedence over DOCCHAT_PORT."""
    env = {"PORT": "9000", "DOCCHAT_PORT": "8080"}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs["port"] == 9000


def test_server_main_falls_back_to_docchat_port():
    """DOCCHAT_PORT used when PORT is absent."""
    env = {"DOCCHAT_PORT": "8080"}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs["port"] == 8080


def test_server_main_reload_true_when_env_set():
    """DOCCHAT_RELOAD=true enables hot-reload."""
    env = {"DOCCHAT_RELOAD": "true"}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs["reload"] is True


def test_server_main_reload_case_insensitive():
    """DOCCHAT_RELOAD=TRUE (uppercase) still enables reload."""
    env = {"DOCCHAT_RELOAD": "TRUE"}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs["reload"] is True


def test_server_main_reload_false_for_random_value():
    """Any value other than 'true' keeps reload=False."""
    env = {"DOCCHAT_RELOAD": "yes"}
    with patch.dict(os.environ, env, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs["reload"] is False


def test_server_main_targets_correct_app_string():
    """uvicorn must receive 'docchat.interfaces.api:app' as the first positional arg."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    args, _ = mock_run.call_args
    assert args[0] == "docchat.interfaces.api:app"


def test_server_main_proxy_headers_enabled():
    """proxy_headers and forwarded_allow_ips must always be set for PaaS."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("uvicorn.run") as mock_run:
            main()

    _, kwargs = mock_run.call_args
    assert kwargs.get("proxy_headers") is True
    assert kwargs.get("forwarded_allow_ips") == "*"

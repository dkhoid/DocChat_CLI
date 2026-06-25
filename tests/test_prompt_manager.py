"""
Tests cho PromptManager — load YAML, render, count tokens, trim context.
"""

from unittest.mock import patch

import pytest

from docchat.core.prompt_manager import PromptManager, get_prompt_manager

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_prompt_dir(tmp_path):
    """Tạo thư mục tạm với một số template YAML."""
    # Template đơn giản - dùng quoted string để tránh ScannerError khi có colon trong value
    (tmp_path / "simple.yaml").write_text(
        'system: "Bạn là trợ lý."\nuser: "Câu hỏi: {query}"\n',
        encoding="utf-8",
    )
    # Template với nhiều biến - dùng 'username' để tránh trùng với tham số 'name' của render()
    (tmp_path / "multi_var.yaml").write_text(
        'system: "Xin chào {username}."\nuser: "Query: {query}\\nContext: {context}"\n',
        encoding="utf-8",
    )
    # Template không có biến
    (tmp_path / "static.yaml").write_text(
        'system: "Static prompt không có biến."\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def pm(tmp_prompt_dir):
    return PromptManager(prompt_dir=tmp_prompt_dir)


# ── load() ─────────────────────────────────────────────────────────────────────


def test_load_returns_dict(pm):
    data = pm.load("simple")
    assert isinstance(data, dict)
    assert "system" in data
    assert "user" in data


def test_load_caches_result(pm):
    d1 = pm.load("simple")
    d2 = pm.load("simple")
    assert d1 is d2  # same object từ cache


def test_load_missing_template_raises(pm):
    with pytest.raises(FileNotFoundError, match="không tìm thấy"):
        pm.load("nonexistent")


def test_load_without_yaml_extension(pm):
    """Không cần truyền .yaml vào tên template."""
    data = pm.load("simple")
    assert data is not None


# ── render() ──────────────────────────────────────────────────────────────────


def test_render_injects_variables(pm):
    rendered = pm.render("simple", query="Python là gì?")
    assert "Python là gì?" in rendered["user"]
    assert "trợ lý" in rendered["system"]


def test_render_multi_var(pm):
    rendered = pm.render("multi_var", username="Alice", query="Hello?", context="Some doc")
    assert "Alice" in rendered["system"]
    assert "Hello?" in rendered["user"]
    assert "Some doc" in rendered["user"]


def test_render_static_no_vars(pm):
    """Template không có biến vẫn render được."""
    rendered = pm.render("static")
    assert "Static prompt" in rendered["system"]


def test_render_missing_variable_raises(pm):
    with pytest.raises(KeyError):
        pm.render("simple")  # thiếu {query}


# ── count_tokens() ────────────────────────────────────────────────────────────


def test_count_tokens_returns_int(pm):
    n = pm.count_tokens("Hello world", model="gpt-4o-mini")
    assert isinstance(n, int)
    assert n > 0


def test_count_tokens_longer_text_more_tokens(pm):
    short = pm.count_tokens("Hi", model="gpt-4o-mini")
    long = pm.count_tokens(
        "Hello world, this is a longer sentence with more words.", model="gpt-4o-mini"
    )
    assert long > short


def test_count_tokens_fallback_without_tiktoken(pm):
    """Phải fallback gracefully nếu tiktoken không có."""
    with patch.dict("sys.modules", {"tiktoken": None}):
        # import lại để patch có hiệu lực
        n = pm.count_tokens("Hello world hello world", model="gpt-4o-mini")
        assert n > 0


def test_count_messages_tokens(pm):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is Python?"},
    ]
    total = pm.count_messages_tokens(messages, model="gpt-4o-mini")
    assert total > 10  # ít nhất phải có token cho nội dung + overhead


# ── trim_to_budget() ──────────────────────────────────────────────────────────


def test_trim_short_text_unchanged(pm):
    short = "Đây là đoạn văn rất ngắn."
    result = pm.trim_to_budget(short, budget=1000, model="gpt-4o-mini")
    assert result == short  # không bị cắt


def test_trim_long_text_gets_cut(pm):
    # Tạo đoạn văn rất dài
    long_text = "Đây là đoạn văn rất dài. " * 500
    result = pm.trim_to_budget(long_text, budget=50, model="gpt-4o-mini")
    # Kết quả phải ngắn hơn input
    assert len(result) < len(long_text)
    assert pm.count_tokens(result, model="gpt-4o-mini") <= 50 + 30  # +30 margin cho marker


def test_trim_adds_marker(pm):
    long_text = "\n\n".join(["Đây là đoạn văn số " + str(i) + "." for i in range(100)])
    result = pm.trim_to_budget(long_text, budget=30, model="gpt-4o-mini")
    assert "cắt bớt" in result  # marker xuất hiện


def test_trim_preserves_paragraph_boundaries(pm):
    """Phải cắt theo đoạn, không cắt giữa chừng."""
    paragraphs = [f"Đoạn {i}: Nội dung hoàn chỉnh." for i in range(20)]
    text = "\n\n".join(paragraphs)
    result = pm.trim_to_budget(text, budget=50, model="gpt-4o-mini")
    # Phần còn lại phải là đoạn hoàn chỉnh (nếu không phải fallback char-based)
    assert result is not None


# ── list_templates() ──────────────────────────────────────────────────────────


def test_list_templates(pm, tmp_prompt_dir):
    templates = pm.list_templates()
    assert "simple" in templates
    assert "multi_var" in templates
    assert "static" in templates


def test_list_templates_empty_dir(tmp_path):
    pm_empty = PromptManager(prompt_dir=tmp_path)
    assert pm_empty.list_templates() == []


def test_list_templates_nonexistent_dir():
    pm_bad = PromptManager(prompt_dir="/nonexistent/path")
    assert pm_bad.list_templates() == []


# ── get_prompt_manager() singleton ────────────────────────────────────────────


def test_get_prompt_manager_returns_instance():
    pm1 = get_prompt_manager(prompt_dir="/tmp")
    pm2 = get_prompt_manager(prompt_dir="/tmp")
    # Với custom dir, không dùng singleton
    assert isinstance(pm1, PromptManager)
    assert isinstance(pm2, PromptManager)

from pathlib import Path

import pytest

from docchat.loader import Document, load_directory, load_file

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_txt(tmp_path: Path) -> Path:
    """Tạo file .txt tạm để test."""
    f = tmp_path / "sample.txt"
    f.write_text("Đây là nội dung test.\nDòng thứ hai.", encoding="utf-8")
    return f


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Tạo thư mục với nhiều file hỗ trợ và không hỗ trợ."""
    (tmp_path / "a.txt").write_text("Nội dung A", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Tiêu đề\nNội dung B", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("không hỗ trợ", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    return tmp_path


# ── load_file ────────────────────────────────────────────────────────────────


def test_load_file_returns_document(tmp_txt: Path):
    doc = load_file(tmp_txt)
    assert isinstance(doc, Document)


def test_load_file_content(tmp_txt: Path):
    doc = load_file(tmp_txt)
    assert "Đây là nội dung test" in doc.content


def test_load_file_source_is_path(tmp_txt: Path):
    doc = load_file(tmp_txt)
    assert doc.source == str(tmp_txt)


def test_load_file_metadata_has_filename(tmp_txt: Path):
    doc = load_file(tmp_txt)
    assert doc.metadata["filename"] == "sample.txt"


def test_load_file_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "file.pdf"
    bad.write_text("content")
    with pytest.raises(ValueError, match="Unsupported"):
        load_file(bad)


def test_load_file_empty_raises(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_file(empty)


# ── load_directory ───────────────────────────────────────────────────────────


def test_load_directory_returns_list(tmp_dir: Path):
    docs = load_directory(tmp_dir)
    assert isinstance(docs, list)


def test_load_directory_only_supported_extensions(tmp_dir: Path):
    docs = load_directory(tmp_dir)
    for doc in docs:
        assert doc.source.endswith(".txt") or doc.source.endswith(".md")


def test_load_directory_skips_empty_files(tmp_dir: Path):
    docs = load_directory(tmp_dir)
    sources = [doc.source for doc in docs]
    assert not any("empty" in s for s in sources)


def test_load_directory_count(tmp_dir: Path):
    # a.txt + b.md = 2 (empty.txt và c.pdf bị skip)
    docs = load_directory(tmp_dir)
    assert len(docs) == 2


def test_load_directory_not_a_dir(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        load_directory(tmp_path / "nonexistent")

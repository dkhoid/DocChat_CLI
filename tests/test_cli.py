from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docchat.cli import cmd_ask, cmd_index, cmd_info, main
from tests.test_embedder import FakeEmbedder

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Thư mục có sẵn 2 file tài liệu."""
    (tmp_path / "a.txt").write_text("Python là ngôn ngữ lập trình phổ biến.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# RAG\nRetrieval Augmented Generation.", encoding="utf-8")
    return tmp_path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def populated_index(docs_dir: Path, data_dir: Path) -> Path:
    """Tạo index sẵn để test cmd_ask và cmd_info."""
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        cmd_index(str(docs_dir), data_dir)
    return data_dir


# ── cmd_index ─────────────────────────────────────────────────────────────────


def test_index_returns_0_on_success(docs_dir: Path, data_dir: Path):
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        code = cmd_index(str(docs_dir), data_dir)
    assert code == 0


def test_index_creates_index_file(docs_dir: Path, data_dir: Path):
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        cmd_index(str(docs_dir), data_dir)
    chroma_path = data_dir / "chroma_db"
    assert chroma_path.exists()


def test_index_invalid_directory(data_dir: Path):
    code = cmd_index("/nonexistent/path", data_dir)
    assert code == 1


def test_index_empty_directory(tmp_path: Path, data_dir: Path):
    """Thư mục không có file hỗ trợ → lỗi."""
    (tmp_path / "data.csv").write_text("a,b,c")
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        code = cmd_index(str(tmp_path), data_dir)
    assert code == 1


# ── cmd_info ──────────────────────────────────────────────────────────────────


def test_info_no_index(tmp_path: Path, capsys):
    code = cmd_info(tmp_path / "nonexistent_subdir")
    assert code == 1
    out = capsys.readouterr().out
    assert "Chưa có index" in out


def test_info_shows_chunk_count(populated_index: Path, capsys):
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        cmd_info(populated_index)
    out = capsys.readouterr().out
    assert "Chunks:" in out


def test_info_shows_file_names(populated_index: Path, capsys):
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        cmd_info(populated_index)
    out = capsys.readouterr().out
    assert "a.txt" in out
    assert "b.md" in out


# ── cmd_ask ───────────────────────────────────────────────────────────────────


def test_ask_no_index_returns_1(tmp_path: Path):
    code = cmd_ask("câu hỏi?", data_dir=tmp_path / "nonexistent_subdir")
    assert code == 1


def test_ask_returns_0_on_success(populated_index: Path):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock()]
    mock_msg.choices[0].message.content = "Câu trả lời"
    mock_msg.usage.prompt_tokens = 10
    mock_msg.usage.completion_tokens = 5
    mock_client.chat.completions.create.return_value = mock_msg

    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_client
            code = cmd_ask("Python là gì?", data_dir=populated_index, stream=False)

    assert code == 0


def test_ask_no_stream_prints_answer(populated_index: Path, capsys):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock()]
    mock_msg.choices[0].message.content = "Đây là câu trả lời"
    mock_msg.usage.prompt_tokens = 10
    mock_msg.usage.completion_tokens = 5
    mock_client.chat.completions.create.return_value = mock_msg

    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_client
            main(["ask", "--data-dir", str(populated_index), "--no-stream", "query?"])

    out = capsys.readouterr().out
    assert "Đây là câu trả lời" in out


def test_main_ask_passes_provider_and_model(populated_index: Path):
    with patch("docchat.cli.cmd_ask", return_value=0) as mock_cmd_ask:
        code = main(
            [
                "ask",
                "query?",
                "--data-dir",
                str(populated_index),
                "--provider",
                "anthropic",
                "--model",
                "claude-3-5-haiku-latest",
                "--max-output-tokens",
                "256",
                "--max-input-tokens",
                "4096",
                "--temperature",
                "0.2",
            ]
        )

    assert code == 0
    cfg = mock_cmd_ask.call_args.kwargs["config"]
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-3-5-haiku-latest"
    assert cfg.max_output_tokens == 256
    assert cfg.max_input_tokens == 4096
    assert abs(cfg.temperature - 0.2) < 1e-9


# ── main() / argparse ─────────────────────────────────────────────────────────


def test_main_index_command(docs_dir: Path, data_dir: Path):
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        code = main(["index", str(docs_dir), "--data-dir", str(data_dir)])
    assert code == 0


def test_main_info_command(populated_index: Path, capsys):
    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        code = main(["info", "--data-dir", str(populated_index)])
    assert code == 0


def test_main_no_command_exits(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_main_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["unknown"])


# ── parametrize: nhiều câu hỏi ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "query",
    [
        "Python là gì?",
        "RAG hoạt động như thế nào?",
        "Làm sao để cài đặt?",
    ],
)
def test_ask_various_queries(populated_index: Path, query: str):
    mock_client = MagicMock()

    # Mock stream response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "trả lời"
    mock_chunk.usage = None

    mock_last_chunk = MagicMock()
    mock_last_chunk.choices = []
    mock_last_chunk.usage.prompt_tokens = 10
    mock_last_chunk.usage.completion_tokens = 5

    mock_client.chat.completions.create.return_value = iter([mock_chunk, mock_last_chunk])

    with patch("docchat.cli.get_embedder", return_value=FakeEmbedder(dim=4)):
        with patch("docchat.llm.openai.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = mock_client
            code = main(["ask", "--data-dir", str(populated_index), query])

    assert code == 0

"""Tests for DocChat FastAPI API layer.

Uses FastAPI TestClient (sync, no real server needed).
All external dependencies (embedder, LLM) are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from docchat.chunker import Chunk
from docchat.store import ChromaVectorStore, SearchResult
# pyrefly: ignore [missing-import]
from tests.test_embedder import FakeEmbedder


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_store(tmp_path: Path) -> ChromaVectorStore:
    """ChromaVectorStore with FakeEmbedder, pre-initialized."""
    embedder = FakeEmbedder(dim=4)
    store = ChromaVectorStore(embedder=embedder)
    store.save(tmp_path)
    chunks = [
        Chunk(text="Python là ngôn ngữ lập trình phổ biến.", source="/docs/a.txt", index=0, chunk_num=0),
        Chunk(text="RAG kết hợp retrieval và generation.", source="/docs/b.md", index=0, chunk_num=0),
    ]
    store.add(chunks)
    return store


@pytest.fixture
def client(fake_store: ChromaVectorStore, tmp_path: Path):
    """TestClient with mocked store — no real ChromaDB or embedder needed."""
    with patch("docchat.api._get_store", return_value=fake_store):
        with patch("docchat.api._store", fake_store):
            with patch("docchat.api.DATA_DIR", tmp_path):
                with patch("docchat.api.UPLOAD_DIR", tmp_path / "uploads"):
                    from docchat.api import app

                    with TestClient(app, raise_server_exceptions=True) as c:
                        yield c


@pytest.fixture
def empty_client(tmp_path: Path):
    """TestClient with empty store (no indexed documents)."""
    embedder = FakeEmbedder(dim=4)
    empty_store = ChromaVectorStore(embedder=embedder)
    empty_store.save(tmp_path)

    with patch("docchat.api._get_store", return_value=empty_store):
        with patch("docchat.api._store", empty_store):
            with patch("docchat.api.DATA_DIR", tmp_path):
                with patch("docchat.api.UPLOAD_DIR", tmp_path / "uploads"):
                    from docchat.api import app

                    with TestClient(app, raise_server_exceptions=True) as c:
                        yield c


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    """Thư mục có file tài liệu để test indexing."""
    doc_dir = tmp_path / "documents"
    doc_dir.mkdir()
    (doc_dir / "intro.txt").write_text("Python là ngôn ngữ lập trình.", encoding="utf-8")
    (doc_dir / "rag.md").write_text("# RAG\nRetrieval Augmented Generation.", encoding="utf-8")
    return doc_dir


# ── GET /health ───────────────────────────────────────────────────────────────


def test_health_returns_200(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["index_size"], int)
    assert data["index_size"] > 0


def test_health_includes_data_dir(client: TestClient):
    resp = client.get("/health")
    data = resp.json()
    assert "data_dir" in data
    assert isinstance(data["data_dir"], str)


# ── GET /info ─────────────────────────────────────────────────────────────────


def test_info_returns_file_list(client: TestClient):
    resp = client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 2
    assert "a.txt" in data["files"]
    assert "b.md" in data["files"]


def test_info_empty_index(empty_client: TestClient):
    resp = empty_client.get("/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 0
    assert data["files"] == {}


# ── POST /index ───────────────────────────────────────────────────────────────


def test_index_valid_directory(client: TestClient, docs_dir: Path):
    with patch("docchat.api.EmbedderFactory.create", return_value=FakeEmbedder(dim=4)):
        resp = client.post("/index", json={"directory": str(docs_dir)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["files_count"] == 2
    assert data["chunks_count"] > 0
    assert data["total_indexed"] > 0


def test_index_invalid_directory(client: TestClient):
    resp = client.post("/index", json={"directory": "/nonexistent/path/xyz"})
    assert resp.status_code == 400
    assert "không phải thư mục" in resp.json()["detail"]


def test_index_empty_directory(client: TestClient, tmp_path: Path):
    empty_dir = tmp_path / "empty_docs"
    empty_dir.mkdir()
    (empty_dir / "data.csv").write_text("a,b,c")
    with patch("docchat.api.EmbedderFactory.create", return_value=FakeEmbedder(dim=4)):
        resp = client.post("/index", json={"directory": str(empty_dir)})
    assert resp.status_code == 400
    assert "Không tìm thấy file" in resp.json()["detail"]


def test_index_missing_directory_field(client: TestClient):
    resp = client.post("/index", json={})
    assert resp.status_code == 422


# ── POST /upload-and-index ────────────────────────────────────────────────────


def test_upload_valid_files(client: TestClient):
    files = [
        ("files", ("test.txt", b"Noi dung tai lieu test.", "text/plain")),
        ("files", ("guide.md", b"# Guide\nHuong dan su dung.", "text/markdown")),
    ]
    resp = client.post("/upload-and-index", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["files_count"] == 2
    assert data["chunks_count"] > 0


def test_upload_no_files(client: TestClient):
    resp = client.post("/upload-and-index", files=[])
    assert resp.status_code == 422


def test_upload_unsupported_files_only(client: TestClient):
    files = [
        ("files", ("image.png", b"\x89PNG\r\n", "image/png")),
        ("files", ("data.csv", b"a,b,c\n1,2,3", "text/csv")),
    ]
    resp = client.post("/upload-and-index", files=files)
    assert resp.status_code == 400
    assert "Không có file .txt/.md" in resp.json()["detail"]


# ── POST /ask ─────────────────────────────────────────────────────────────────


def test_ask_returns_answer(client: TestClient):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock()]
    mock_msg.choices[0].message.content = "Python rất phổ biến."
    mock_msg.usage.prompt_tokens = 50
    mock_msg.usage.completion_tokens = 10
    mock_client.chat.completions.create.return_value = mock_msg

    with patch("docchat.llm.openai.OpenAI", return_value=mock_client):
        resp = client.post("/ask", json={"query": "Python là gì?", "stream": False})

    assert resp.status_code == 200
    data = resp.json()
    assert "Python" in data["answer"]
    assert isinstance(data["sources"], list)
    assert "input_tokens" in data["stats"]


def test_ask_empty_query(client: TestClient):
    resp = client.post("/ask", json={"query": ""})
    assert resp.status_code == 422


def test_ask_no_index(empty_client: TestClient):
    resp = empty_client.post("/ask", json={"query": "test?"})
    assert resp.status_code == 400
    assert "Chưa có index" in resp.json()["detail"]


def test_ask_with_custom_params(client: TestClient):
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock()]
    mock_msg.choices[0].message.content = "Answer"
    mock_msg.usage.prompt_tokens = 20
    mock_msg.usage.completion_tokens = 5
    mock_client.chat.completions.create.return_value = mock_msg

    with patch("docchat.llm.openai.OpenAI", return_value=mock_client):
        resp = client.post("/ask", json={
            "query": "test?",
            "top_k": 3,
            "temperature": 0.3,
            "max_output_tokens": 256,
            "stream": False,
        })
    assert resp.status_code == 200


def test_ask_stream_returns_sse(client: TestClient):
    mock_client = MagicMock()

    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "token1"
    mock_chunk.usage = None

    mock_last = MagicMock()
    mock_last.choices = []
    mock_last.usage.prompt_tokens = 10
    mock_last.usage.completion_tokens = 3

    mock_client.chat.completions.create.return_value = iter([mock_chunk, mock_last])

    with patch("docchat.llm.openai.OpenAI", return_value=mock_client):
        resp = client.post("/ask", json={"query": "test?", "stream": True})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    lines = [l for l in resp.text.strip().split("\n") if l.startswith("data:")]
    assert len(lines) >= 1

    first_event = json.loads(lines[0].removeprefix("data: "))
    assert "token" in first_event or "done" in first_event


# ── POST /chat/create + /chat/{id} ───────────────────────────────────────────


def test_chat_create_returns_session_id(client: TestClient):
    with patch("docchat.llm.openai.OpenAI", return_value=MagicMock()):
        resp = client.post("/chat/create")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert len(data["session_id"]) > 0


def test_chat_create_no_index(empty_client: TestClient):
    resp = empty_client.post("/chat/create")
    assert resp.status_code == 400


def test_chat_message_flow(client: TestClient):
    """Full flow: create → send message → check stats → delete."""
    mock_openai_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.choices = [MagicMock()]
    mock_msg.choices[0].message.content = "Trả lời từ chat"
    mock_msg.usage.prompt_tokens = 30
    mock_msg.usage.completion_tokens = 8
    mock_openai_client.chat.completions.create.return_value = mock_msg

    with patch("docchat.llm.openai.OpenAI", return_value=mock_openai_client):
        # 1. Create session
        resp = client.post("/chat/create")
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # 2. Send message
        resp = client.post(f"/chat/{session_id}", json={"query": "Xin chào?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Trả lời từ chat"
        assert data["turn"] >= 1
        assert isinstance(data["sources"], list)

        # 3. Check stats
        resp = client.get(f"/chat/{session_id}/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["session_id"] == session_id
        assert stats["history_length"] >= 2

        # 4. Clear history
        resp = client.post(f"/chat/{session_id}/clear")
        assert resp.status_code == 200

        # 5. Verify history cleared
        resp = client.get(f"/chat/{session_id}/stats")
        assert resp.json()["history_length"] == 0

        # 6. Delete session
        resp = client.delete(f"/chat/{session_id}")
        assert resp.status_code == 200


def test_chat_nonexistent_session(client: TestClient):
    resp = client.post("/chat/nonexistent-id", json={"query": "test?"})
    assert resp.status_code == 404


def test_chat_stats_nonexistent(client: TestClient):
    resp = client.get("/chat/nonexistent-id/stats")
    assert resp.status_code == 404


def test_chat_delete_nonexistent(client: TestClient):
    resp = client.delete("/chat/nonexistent-id")
    assert resp.status_code == 404


def test_chat_clear_nonexistent(client: TestClient):
    resp = client.post("/chat/nonexistent-id/clear")
    assert resp.status_code == 404


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_ask_invalid_temperature(client: TestClient):
    resp = client.post("/ask", json={"query": "test?", "temperature": 5.0})
    assert resp.status_code == 422


def test_ask_invalid_top_k(client: TestClient):
    resp = client.post("/ask", json={"query": "test?", "top_k": 0})
    assert resp.status_code == 422


def test_ask_invalid_top_k_too_high(client: TestClient):
    resp = client.post("/ask", json={"query": "test?", "top_k": 100})
    assert resp.status_code == 422


def test_openapi_schema_available(client: TestClient):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["info"]["title"] == "DocChat API"
    assert "/health" in schema["paths"]
    assert "/ask" in schema["paths"]
    assert "/info" in schema["paths"]

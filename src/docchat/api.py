"""FastAPI REST API wrapper for DocChat.

Thin layer that reuses core modules (store, llm, embedder, chunker, loader)
to expose DocChat capabilities via HTTP endpoints.
"""

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from docchat.chunker import chunk_documents
from docchat.embedder import BaseEmbedder, EmbedderFactory
from docchat.llm import LLMConfig, LLMSession
from docchat.loader import SUPPORTED_EXTENSIONS, load_directory
from docchat.store import ChromaVectorStore

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path(os.environ.get("DOCCHAT_DATA_DIR", str(Path.home() / ".docchat")))
UPLOAD_DIR = Path(os.environ.get("DOCCHAT_UPLOAD_DIR", str(DATA_DIR / "uploads")))
CHUNK_SIZE = int(os.environ.get("DOCCHAT_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("DOCCHAT_CHUNK_OVERLAP", "64"))
DEFAULT_TOP_K = int(os.environ.get("DOCCHAT_TOP_K", "5"))


def _get_embedder() -> BaseEmbedder:
    provider = os.environ.get("EMBEDDER", "local")
    return EmbedderFactory.create(provider)


# ── Shared state ──────────────────────────────────────────────────────────────

_store: ChromaVectorStore | None = None
_chat_sessions: dict[str, LLMSession] = {}


def _get_store() -> ChromaVectorStore:
    global _store
    if _store is not None:
        return _store

    embedder = _get_embedder()
    _store = ChromaVectorStore(embedder=embedder)

    chroma_path = DATA_DIR / "chroma_db"
    if chroma_path.exists():
        _store.load(DATA_DIR)
    else:
        _store.save(DATA_DIR)

    return _store


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _get_store()
    yield
    for session in _chat_sessions.values():
        session.__exit__(None, None, None)
    _chat_sessions.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocChat API",
    description="REST API cho hệ thống hỏi đáp tài liệu DocChat (RAG).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    index_size: int
    data_dir: str


class IndexRequest(BaseModel):
    directory: str = Field(..., description="Đường dẫn thư mục chứa file .txt/.md trên server")
    embedder: Literal["local", "openai"] = "local"


class IndexResponse(BaseModel):
    files_count: int
    chunks_count: int
    total_indexed: int


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Câu hỏi")
    api_key: str | None = Field(
        default=None, description="Tự thêm API Key của bạn (OpenAI/Anthropic)"
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1, le=8192)
    stream: bool = Field(default=False, description="True = SSE streaming response")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    stats: dict


class ChatCreateResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    turn: int
    sources: list[str]
    stats: dict


class ChatStatsResponse(BaseModel):
    session_id: str
    history_length: int
    stats: dict


class InfoResponse(BaseModel):
    data_dir: str
    total_chunks: int
    files: dict[str, int]


class ErrorResponse(BaseModel):
    detail: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    store = _get_store()
    return HealthResponse(
        status="ok",
        index_size=store.size,
        data_dir=str(DATA_DIR),
    )


@app.post("/index", response_model=IndexResponse)
async def index_documents(req: IndexRequest):
    dir_path = Path(req.directory)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"'{req.directory}' không phải thư mục hợp lệ.")

    docs = await asyncio.to_thread(load_directory, dir_path)
    if not docs:
        raise HTTPException(status_code=400, detail="Không tìm thấy file .txt hoặc .md nào.")

    chunks = list(chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))

    global _store
    embedder = EmbedderFactory.create(req.embedder)
    _store = ChromaVectorStore(embedder=embedder)
    _store.save(DATA_DIR)
    _store.add(chunks)

    return IndexResponse(
        files_count=len(docs),
        chunks_count=len(chunks),
        total_indexed=_store.size,
    )


@app.post("/upload-and-index", response_model=IndexResponse)
async def upload_and_index(files: list[UploadFile]):
    if not files:
        raise HTTPException(status_code=400, detail="Không có file nào được upload.")

    upload_path = UPLOAD_DIR / str(uuid.uuid4())
    upload_path.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        dest = upload_path / (f.filename or f"file_{saved_count}{suffix}")
        content = await f.read()
        dest.write_bytes(content)
        saved_count += 1

    if saved_count == 0:
        shutil.rmtree(upload_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Không có file .txt/.md hợp lệ.")

    docs = await asyncio.to_thread(load_directory, upload_path)
    chunks = list(chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP))

    store = _get_store()
    store.add(chunks)

    return IndexResponse(
        files_count=len(docs),
        chunks_count=len(chunks),
        total_indexed=store.size,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    store = _get_store()
    if store.size == 0:
        raise HTTPException(status_code=400, detail="Chưa có index. Gọi /index trước.")

    if req.stream:
        return StreamingResponse(
            _stream_ask(store, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    config = LLMConfig(
        api_key=req.api_key,
        provider=req.provider,
        model=req.model,
        max_output_tokens=req.max_output_tokens,
        temperature=req.temperature,
    )

    with LLMSession(config) as session:
        answer = await asyncio.to_thread(session.complete, req.query, store, req.top_k)

    results = store.search(req.query, k=req.top_k)
    sources = list(
        {Path(r.chunk.source).name for r in results if r.score >= config.min_relevance_score}
    )

    return AskResponse(
        answer=answer,
        sources=sources,
        stats={
            "input_tokens": session.stats.total_input_tokens,
            "output_tokens": session.stats.total_output_tokens,
            "cost_usd": round(session.stats.cost_usd, 8),
            "latency_s": round(session.stats.total_time, 3),
        },
    )


async def _stream_ask(store: ChromaVectorStore, req: AskRequest):
    """SSE generator for streaming LLM response."""
    import json

    config = LLMConfig(
        api_key=req.api_key,
        provider=req.provider,
        model=req.model,
        max_output_tokens=req.max_output_tokens,
        temperature=req.temperature,
    )

    with LLMSession(config) as session:
        async for token in session.stream(req.query, store, k=req.top_k):
            yield f"data: {json.dumps({'token': token})}\n\n"

        results = store.search(req.query, k=req.top_k)
        sources = list({Path(r.chunk.source).name for r in results})
        done_payload = {
            "done": True,
            "sources": sources,
            "stats": {
                "input_tokens": session.stats.total_input_tokens,
                "output_tokens": session.stats.total_output_tokens,
                "cost_usd": round(session.stats.cost_usd, 8),
            },
        }
        yield f"data: {json.dumps(done_payload)}\n\n"


# ── Chat sessions ────────────────────────────────────────────────────────────


@app.post("/chat/create", response_model=ChatCreateResponse)
async def chat_create(
    provider: Literal["openai", "anthropic"] = "openai",
    model: str = "",
    temperature: float = 0.7,
    api_key: str | None = None,
):
    store = _get_store()
    if store.size == 0:
        raise HTTPException(status_code=400, detail="Chưa có index. Gọi /index trước.")

    session_id = str(uuid.uuid4())
    config = LLMConfig(api_key=api_key, provider=provider, model=model, temperature=temperature)
    session = LLMSession(config)
    session.__enter__()
    _chat_sessions[session_id] = session

    return ChatCreateResponse(session_id=session_id)


@app.post("/chat/{session_id}", response_model=ChatResponse)
async def chat_message(session_id: str, req: ChatRequest):
    session = _chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại.")

    store = _get_store()
    answer = await asyncio.to_thread(
        session.complete,
        req.query,
        store,
        req.top_k,
        use_history=True,
    )

    results = store.search(req.query, k=req.top_k)
    sources = list({Path(r.chunk.source).name for r in results})

    return ChatResponse(
        answer=answer,
        turn=len(session.history) // 2,
        sources=sources,
        stats={
            "input_tokens": session.stats.total_input_tokens,
            "output_tokens": session.stats.total_output_tokens,
            "cost_usd": round(session.stats.cost_usd, 8),
        },
    )


@app.get("/chat/{session_id}/stats", response_model=ChatStatsResponse)
async def chat_stats(session_id: str):
    session = _chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại.")

    return ChatStatsResponse(
        session_id=session_id,
        history_length=len(session.history),
        stats={
            "calls": session.stats.call_count,
            "input_tokens": session.stats.total_input_tokens,
            "output_tokens": session.stats.total_output_tokens,
            "cost_usd": round(session.stats.cost_usd, 8),
            "avg_latency": round(
                session.stats.total_time / session.stats.call_count
                if session.stats.call_count
                else 0,
                3,
            ),
        },
    )


@app.delete("/chat/{session_id}")
async def chat_delete(session_id: str):
    session = _chat_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại.")

    session.__exit__(None, None, None)
    return {"detail": f"Session '{session_id}' đã được xóa."}


@app.post("/chat/{session_id}/clear")
async def chat_clear_history(session_id: str):
    session = _chat_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' không tồn tại.")

    session.clear_history()
    return {"detail": "Đã xóa lịch sử hội thoại."}


# ── Info ──────────────────────────────────────────────────────────────────────


@app.get("/info", response_model=InfoResponse)
async def info():
    store = _get_store()
    if store.size == 0:
        return InfoResponse(data_dir=str(DATA_DIR), total_chunks=0, files={})

    chunks = await asyncio.to_thread(lambda: store.chunks)
    file_counts: dict[str, int] = {}
    for c in chunks:
        name = Path(c.source).name
        file_counts[name] = file_counts.get(name, 0) + 1

    return InfoResponse(
        data_dir=str(DATA_DIR),
        total_chunks=store.size,
        files=file_counts,
    )

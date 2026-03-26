import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from docchat.chunker import chunk_documents
from docchat.embedder import EmbedderFactory, BaseEmbedder
from docchat.llm import LLMConfig, LLMSession
from docchat.loader import load_directory
from docchat.store import SimpleVectorStore


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_INDEX_PATH = Path.home() / ".docchat" / "index.pkl"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5


def get_embedder() -> BaseEmbedder:
    """Tạo embedder từ env var EMBEDDER (mặc định: local)."""
    provider = os.environ.get("EMBEDDER", "local")
    if provider == "openai":
        return EmbedderFactory.create("openai")
    return EmbedderFactory.create("local")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_index(directory: str, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    """
    Index tất cả file trong directory.
    Trả về exit code: 0 = ok, 1 = lỗi.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        print(f"Lỗi: '{directory}' không phải thư mục.", file=sys.stderr)
        return 1

    print(f"Indexing {directory} ...")

    try:
        docs = load_directory(dir_path)
    except Exception as e:
        print(f"Lỗi đọc file: {e}", file=sys.stderr)
        return 1

    if not docs:
        print("Không tìm thấy file .txt hoặc .md nào.", file=sys.stderr)
        return 1

    chunks = list(
        chunk_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    )

    print(f"  {len(docs)} file → {len(chunks)} chunks")
    print("Đang tạo embedding ...")

    embedder = get_embedder()
    store = SimpleVectorStore(embedder=embedder)

    # In tiến trình từng file
    for doc in docs:
        doc_chunks = [c for c in chunks if c.source == str(dir_path / doc.source)
                      or c.source == doc.source]
        print(f"  ✓ {Path(doc.source).name:40s} → {len(doc_chunks)} chunks")

    store.add(chunks)
    store.save(index_path)
    print(f"\nHoàn thành: {store.size} chunks đã được index.")
    return 0


def cmd_ask(
    query: str,
    index_path: Path = DEFAULT_INDEX_PATH,
    k: int = DEFAULT_TOP_K,
    stream: bool = True,
) -> int:
    """Hỏi một câu, in câu trả lời ra stdout."""
    if not index_path.exists():
        print(
            "Lỗi: chưa có index. Chạy 'docchat index <thư mục>' trước.",
            file=sys.stderr,
        )
        return 1

    embedder = get_embedder()
    store = SimpleVectorStore(embedder=embedder)
    store.load(index_path)

    config = LLMConfig()
    print()  # dòng trống trước câu trả lời

    with LLMSession(config) as session:
        if stream:
            asyncio.run(_stream_answer(session, query, store, k))
        else:
            answer = session.complete(query, store, k=k)
            print(answer)

    print()  # dòng trống sau câu trả lời
    return 0


async def _stream_answer(
    session: LLMSession,
    query: str,
    store: SimpleVectorStore,
    k: int,
) -> None:
    async for _ in session.stream(query, store, k=k):
        pass  # token đã được in trong _stream_sync


def cmd_info(index_path: Path = DEFAULT_INDEX_PATH) -> int:
    """In thông tin về index hiện tại."""
    if not index_path.exists():
        print("Chưa có index. Chạy 'docchat index <thư mục>' để tạo.")
        return 1

    embedder = get_embedder()
    store = SimpleVectorStore(embedder=embedder)
    store.load(index_path)

    sources = {Path(c.source).name for c in store.chunks}
    print(f"\nIndex: {index_path}")
    print(f"Chunks: {store.size}")
    print(f"Files:  {len(sources)}")
    for name in sorted(sources):
        count = sum(1 for c in store.chunks if Path(c.source).name == name)
        print(f"  {name:40s} {count} chunks")
    return 0


# ── CLI parser ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """Entry point. Trả về exit code."""
    import argparse

    # Nạp file .env trước khi chạy bất kỳ lệnh nào
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="docchat",
        description="Hỏi đáp tài liệu từ terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # docchat index <dir>
    p_index = sub.add_parser("index", help="Index tài liệu trong thư mục.")
    p_index.add_argument("directory", help="Thư mục chứa file .txt / .md")
    p_index.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))

    # docchat ask <query>
    p_ask = sub.add_parser("ask", help="Hỏi một câu dựa trên tài liệu đã index.")
    p_ask.add_argument("query", help="Câu hỏi")
    p_ask.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    p_ask.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_ask.add_argument("--no-stream", action="store_true")

    # docchat info
    p_info = sub.add_parser("info", help="Xem thông tin index hiện tại.")
    p_info.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))

    args = parser.parse_args(argv)

    if args.command == "index":
        return cmd_index(args.directory, Path(args.index_path))

    if args.command == "ask":
        return cmd_ask(
            args.query,
            index_path=Path(args.index_path),
            k=args.top_k,
            stream=not args.no_stream,
        )

    if args.command == "info":
        return cmd_info(Path(args.index_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
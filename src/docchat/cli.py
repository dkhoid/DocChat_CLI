import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from docchat.chunker import chunk_documents
from docchat.embedder import BaseEmbedder, EmbedderFactory
from docchat.llm import LLMConfig, LLMSession
from docchat.loader import load_directory
from docchat.store import ChromaVectorStore

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path.home() / ".docchat"
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOP_K = 5


def _index_exists(data_dir: Path) -> bool:
    """Kiểm tra index tồn tại (ChromaDB hoặc legacy Pickle)."""
    return (data_dir / "chroma_db").exists() or (data_dir / "index.pkl").exists()


def get_embedder(provider: str | None = None) -> BaseEmbedder:
    """Tạo embedder từ provider (ưu tiên) hoặc env var EMBEDDER (mặc định: local)."""
    if provider is None:
        provider = os.environ.get("EMBEDDER", "local")
    if provider == "openai":
        return EmbedderFactory.create("openai")
    return EmbedderFactory.create("local")


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_index(directory: str, data_dir: Path = DEFAULT_DATA_DIR, embedder_provider: str = "local") -> int:
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
    print(f"Đang tạo embedding ({embedder_provider})...")

    embedder = get_embedder(embedder_provider)
    store = ChromaVectorStore(embedder=embedder)
    store.save(data_dir)

    # In tiến trình từng file
    for doc in docs:
        doc_chunks = [c for c in chunks if c.source == str(dir_path / doc.source)
                      or c.source == doc.source]
        print(f"  ✓ {Path(doc.source).name:40s} → {len(doc_chunks)} chunks")

    store.add(chunks)
    print(f"\nHoàn thành: {store.size} chunks đã được index vào ChromaDB.")
    return 0


def cmd_ask(
    query: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    k: int = DEFAULT_TOP_K,
    stream: bool = True,
    config: LLMConfig | None = None,
    embedder_provider: str = "local"
) -> int:
    """Hỏi một câu, in câu trả lời ra stdout."""
    if not _index_exists(data_dir):
        print(
            "Lỗi: chưa có index. Chạy 'docchat index <thư mục>' trước.",
            file=sys.stderr,
        )
        return 1

    embedder = get_embedder(embedder_provider)
    store = ChromaVectorStore(embedder=embedder)
    store.load(data_dir)

    config = config or LLMConfig()
    
    # In ra log debug
    print("\n[Debug] Retrieved Chunks:")
    results = store.search(query, k=k)
    filtered = [r for r in results if r.score >= config.min_relevance_score]
    if not filtered:
        print("  (Không có chunk nào đạt min_relevance_score)")
    for i, r in enumerate(filtered, 1):
        preview = r.chunk.text.replace("\n", " ")[:70]
        print(f"  {i}. [{Path(r.chunk.source).name}] (score: {r.score:.3f}) - {preview}...")
    
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
    store: ChromaVectorStore,
    k: int,
    use_history: bool = False,
) -> None:
    async for token in session.stream(query, store, k=k, use_history=use_history):
        print(token, end="", flush=True)


def cmd_chat(
    data_dir: Path = DEFAULT_DATA_DIR,
    k: int = DEFAULT_TOP_K,
    config: LLMConfig | None = None,
    embedder_provider: str = "local",
) -> int:
    """
    Interactive multi-turn chat với tài liệu đã index.
    Gõ /exit hoặc Ctrl+C để thoát.
    Gõ /clear để xóa history.
    Gõ /stats để xem thống kê session.
    """
    if not _index_exists(data_dir):
        print(
            "Lỗi: chưa có index. Chạy 'docchat index <thư mục>' trước.",
            file=sys.stderr,
        )
        return 1

    embedder = get_embedder(embedder_provider)
    store = ChromaVectorStore(embedder=embedder)
    store.load(data_dir)

    config = config or LLMConfig()

    print("\n" + "=" * 60)
    print("  DocChat — Chế độ chat đa lượt")
    print("=" * 60)
    print("  /exit  — Thoát")
    print("  /clear — Xóa lịch sử hội thoại")
    print("  /stats — Xem thống kê session")
    print("=" * 60 + "\n")

    with LLMSession(config) as session:
        turn = 1
        while True:
            try:
                query = input(f"[{turn}] Bạn: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nThoát.")
                break

            if not query:
                continue

            # ── Lệnh đặc biệt ──
            if query == "/exit":
                print("Thoát.")
                break
            if query == "/clear":
                session.clear_history()
                print("  [✓] Đã xóa lịch sử hội thoại.\n")
                turn = 1
                continue
            if query == "/stats":
                print(f"  [stats] {session.stats.report()}\n")
                continue

            # ── Câu hỏi thông thường ──
            # In ra chunk retrieval
            results = store.search(query, k=k)
            filtered = [r for r in results if r.score >= config.min_relevance_score]
            if filtered:
                print("  [Debug] Using contexts: ", end="")
                sources = set(Path(r.chunk.source).name for r in filtered)
                print(", ".join(sources))
            else:
                print("  [Debug] No contexts found matching relevance threshold.")

            print("\nDocChat: ", end="", flush=True)
            try:
                asyncio.run(
                    _stream_answer(session, query, store, k, use_history=True)
                )
            except Exception as e:
                print(f"\n[Lỗi] {e}", file=sys.stderr)

            print("\n")
            turn += 1

    return 0


def cmd_info(data_dir: Path = DEFAULT_DATA_DIR, embedder_provider: str = "local") -> int:
    """In thông tin về index hiện tại."""
    if not _index_exists(data_dir):
        print("Chưa có index. Chạy 'docchat index <thư mục>' để tạo.")
        return 1

    embedder = get_embedder(embedder_provider)
    store = ChromaVectorStore(embedder=embedder)
    store.load(data_dir)

    sources = {Path(c.source).name for c in store.chunks}
    print(f"\nIndex: {data_dir / 'chroma_db'}")
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
    p_index.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Thư mục lưu ChromaDB index")
    p_index.add_argument("--embedder", choices=["local", "openai"], default="local")

    # docchat ask <query>
    p_ask = sub.add_parser("ask", help="Hỏi một câu dựa trên tài liệu đã index.")
    p_ask.add_argument("query", help="Câu hỏi")
    p_ask.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Thư mục chứa ChromaDB index")
    p_ask.add_argument("--embedder", choices=["local", "openai"], default="local")
    p_ask.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_ask.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p_ask.add_argument("--model", default="")
    p_ask.add_argument("--max-output-tokens", type=int, default=512)
    p_ask.add_argument("--max-input-tokens", type=int, default=8000)
    p_ask.add_argument("--temperature", type=float, default=0.7)
    p_ask.add_argument("--min-relevance-score", type=float, default=0.0)
    p_ask.add_argument("--no-stream", action="store_true")

    # docchat chat (multi-turn REPL)
    p_chat = sub.add_parser("chat", help="Chat đa lượt với tài liệu (có memory).")
    p_chat.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Thư mục chứa ChromaDB index")
    p_chat.add_argument("--embedder", choices=["local", "openai"], default="local")
    p_chat.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p_chat.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    p_chat.add_argument("--model", default="")
    p_chat.add_argument("--max-output-tokens", type=int, default=512)
    p_chat.add_argument("--max-input-tokens", type=int, default=8000)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--min-relevance-score", type=float, default=0.0)

    # docchat info
    p_info = sub.add_parser("info", help="Xem thông tin index hiện tại.")
    p_info.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Thư mục chứa ChromaDB index")
    p_info.add_argument("--embedder", choices=["local", "openai"], default="local")

    args = parser.parse_args(argv)

    if args.command == "index":
        return cmd_index(
            args.directory, 
            Path(args.data_dir),
            embedder_provider=args.embedder
        )

    if args.command == "ask":
        config = LLMConfig(
            provider=args.provider,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            max_input_tokens=args.max_input_tokens,
            temperature=args.temperature,
            min_relevance_score=args.min_relevance_score,
        )
        return cmd_ask(
            args.query,
            data_dir=Path(args.data_dir),
            k=args.top_k,
            stream=not args.no_stream,
            config=config,
            embedder_provider=args.embedder,
        )

    if args.command == "chat":
        config = LLMConfig(
            provider=args.provider,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            max_input_tokens=args.max_input_tokens,
            temperature=args.temperature,
            min_relevance_score=args.min_relevance_score,
        )
        return cmd_chat(
            data_dir=Path(args.data_dir),
            k=args.top_k,
            config=config,
            embedder_provider=args.embedder,
        )

    if args.command == "info":
        return cmd_info(Path(args.data_dir), embedder_provider=args.embedder)

    return 0


if __name__ == "__main__":
    sys.exit(main())
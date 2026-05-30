import argparse
import pickle
import sys
import time
from pathlib import Path

# Giả lập môi trường module để chạy script ngoài src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docchat.chunker import Chunk
from docchat.embedder import EmbedderFactory
from docchat.store import ChromaVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Migrate từ index.pkl (SimpleVectorStore) sang ChromaDB."
    )
    parser.add_argument(
        "--old-index",
        default=str(Path.home() / ".docchat" / "index.pkl"),
        help="Đường dẫn file index.pkl cũ",
    )
    parser.add_argument(
        "--embedder",
        default="openai",
        choices=["local", "openai"],
        help="Cơ chế nhúng (Embedder) sử dụng lại để re-embed",
    )

    args = parser.parse_args()

    old_path = Path(args.old_index)
    if not old_path.exists():
        print(f"Lỗi: Không tìm thấy file index.pkl cũ tại {old_path}")
        return 1

    print(f"Bắt đầu đọc dữ liệu cũ từ: {old_path}")
    try:
        with open(old_path, "rb") as f:
            data = pickle.load(f)
        chunks: list[Chunk] = data.get("chunks", [])
        if not chunks:
            print("File index cũ không có chunks nào.")
            return 0
    except Exception as e:
        print(f"Lỗi khi đọc file pickle: {e}")
        return 1

    print(f"Đã đọc thành công {len(chunks)} chunks cũ.")
    print(f"Khởi tạo embedder [{args.embedder}] để tiến hành Re-Embed hệ thống...")

    try:
        embedder = EmbedderFactory.create(args.embedder)
        store = ChromaVectorStore(embedder=embedder)
        # Khởi tạo db_path cùng cấp với index.pkl
        store.save(old_path.parent)

        # Có thể chunks quá lớn, cần chia batch
        batch_size = 100
        total = len(chunks)
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            print(f" Đang nhúng và chép batch {(i // batch_size) + 1} ({i}/{total})...")
            # add() tự động nhúng văn bản và put vào ChromaDB
            store.add(batch)

        print(f"\nMigration hoàn tất. {total} chunks đã được đẩy vào ChromaDB mới.")
        print(f"Thời gian: {time.time() - start_time:.2f}s")
        print(
            "Lưu ý: Index cũ vẫn được giữ nguyên. Hãy xoá file index.pkl thủ công nếu không cần dùng tới."  # noqa: E501
        )
        return 0
    except Exception as e:
        print(f"\n[Lỗi Migration] Không thể re-embed hoặc lưu vào Chroma DB: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

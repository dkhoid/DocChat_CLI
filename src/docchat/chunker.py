from dataclasses import dataclass
from collections.abc import Iterator

from docchat.loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    index: int        # vị trí ký tự bắt đầu trong document gốc
    chunk_num: int    # thứ tự chunk trong document (0-based)


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64,
) -> Iterator[Chunk]:
    """
    Tách một Document thành các Chunk có overlap.

    Args:
        doc: Document cần tách.
        chunk_size: Số ký tự tối đa mỗi chunk.
        overlap: Số ký tự overlap giữa các chunk liền kề.

    Yields:
        Chunk theo thứ tự từ đầu đến cuối document.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size phải > 0")
    if overlap < 0:
        raise ValueError("overlap phải >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")

    text = doc.content
    step = chunk_size - overlap
    chunk_num = 0

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            yield Chunk(
                text=chunk_text,
                source=doc.source,
                index=start,
                chunk_num=chunk_num,
            )
            chunk_num += 1

        # Dừng nếu đã đến cuối
        if end >= len(text):
            break


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 512,
    overlap: int = 64,
) -> Iterator[Chunk]:
    """
    Tách nhiều Document — generator pipeline, không tạo list trung gian.
    """
    for doc in docs:
        yield from chunk_document(doc, chunk_size, overlap)
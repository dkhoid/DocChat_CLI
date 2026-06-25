import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field

from docchat.core.loader import Document


@dataclass
class Chunk:
    text: str
    source: str
    index: int  # vị trí ký tự bắt đầu trong document gốc
    chunk_num: int  # thứ tự chunk trong document (0-based)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


def chunk_document(
    doc: Document,
    chunk_size: int = 400,
    overlap: int = 50,
) -> Iterator[Chunk]:
    """
    Tách một Document thành các Chunk có overlap sử dụng RecursiveCharacterTextSplitter
    từ langchain_text_splitters (chia theo token).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size phải > 0")
    if overlap < 0:
        raise ValueError("overlap phải >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError("Cần cài đặt langchain-text-splitters (uv add langchain-text-splitters)")

    # Dùng tiktoken để đếm token; add_start_index=True để ghi character offset thực vào metadata
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True,
    )

    docs = splitter.create_documents([doc.content])
    for i, d in enumerate(docs):
        yield Chunk(
            text=d.page_content,
            source=doc.source,
            index=d.metadata.get("start_index", 0),
            chunk_num=i,
        )


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 400,
    overlap: int = 50,
) -> Iterator[Chunk]:
    """
    Tách nhiều Document — generator pipeline, không tạo list trung gian.
    """
    for doc in docs:
        yield from chunk_document(doc, chunk_size, overlap)

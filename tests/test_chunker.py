import pytest

from docchat.loader import Document
from docchat.chunker import Chunk, chunk_document, chunk_documents


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def short_doc() -> Document:
    return Document(content="Hello world", source="test.txt")


@pytest.fixture
def long_doc() -> Document:
    # 1000 ký tự 'a'
    return Document(content="a" * 1000, source="long.txt")


@pytest.fixture
def multi_docs(long_doc: Document) -> list[Document]:
    doc2 = Document(content="b" * 800, source="second.txt")
    return [long_doc, doc2]


# ── chunk_document ────────────────────────────────────────────────────────────

def test_chunk_returns_chunks(long_doc: Document):
    chunks = list(chunk_document(long_doc))
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_size_respected(long_doc: Document):
    chunks = list(chunk_document(long_doc, chunk_size=100, overlap=0))
    for chunk in chunks:
        assert len(chunk.text) <= 100


def test_chunk_overlap(long_doc: Document):
    """Chunk sau phải bắt đầu bằng phần cuối chunk trước (overlap)."""
    chunks = list(chunk_document(long_doc, chunk_size=100, overlap=20))
    for i in range(1, len(chunks)):
        tail = chunks[i - 1].text[-20:]
        head = chunks[i].text[:20]
        assert tail == head


def test_chunk_num_sequential(long_doc: Document):
    chunks = list(chunk_document(long_doc, chunk_size=100, overlap=0))
    nums = [c.chunk_num for c in chunks]
    assert nums == list(range(len(chunks)))


def test_chunk_source_preserved(long_doc: Document):
    chunks = list(chunk_document(long_doc))
    assert all(c.source == "long.txt" for c in chunks)


def test_chunk_short_doc(short_doc: Document):
    """Document ngắn hơn chunk_size → chỉ ra 1 chunk."""
    chunks = list(chunk_document(short_doc, chunk_size=512))
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"


def test_chunk_invalid_chunk_size(long_doc: Document):
    with pytest.raises(ValueError, match="chunk_size"):
        list(chunk_document(long_doc, chunk_size=0))


def test_chunk_invalid_overlap_negative(long_doc: Document):
    with pytest.raises(ValueError, match="overlap"):
        list(chunk_document(long_doc, chunk_size=100, overlap=-1))


def test_chunk_overlap_too_large(long_doc: Document):
    with pytest.raises(ValueError, match="overlap"):
        list(chunk_document(long_doc, chunk_size=100, overlap=100))


# ── parametrize: nhiều cấu hình ──────────────────────────────────────────────

@pytest.mark.parametrize("size,overlap", [
    (256, 0),
    (256, 32),
    (512, 64),
    (128, 16),
])
def test_chunk_parametrize(long_doc: Document, size: int, overlap: int):
    chunks = list(chunk_document(long_doc, chunk_size=size, overlap=overlap))
    assert len(chunks) > 0
    for c in chunks:
        assert len(c.text) <= size


# ── chunk_documents ───────────────────────────────────────────────────────────

def test_chunk_documents_is_generator(multi_docs):
    import types
    result = chunk_documents(multi_docs)
    assert isinstance(result, types.GeneratorType)


def test_chunk_documents_all_sources(multi_docs):
    chunks = list(chunk_documents(multi_docs))
    sources = {c.source for c in chunks}
    assert sources == {"long.txt", "second.txt"}


def test_chunk_documents_total_count(multi_docs):
    chunks_1 = list(chunk_document(multi_docs[0], chunk_size=512, overlap=64))
    chunks_2 = list(chunk_document(multi_docs[1], chunk_size=512, overlap=64))
    all_chunks = list(chunk_documents(multi_docs, chunk_size=512, overlap=64))
    assert len(all_chunks) == len(chunks_1) + len(chunks_2)
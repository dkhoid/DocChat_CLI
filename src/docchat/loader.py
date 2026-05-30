from dataclasses import dataclass, field
from pathlib import Path

from docchat.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_file(path: Path) -> Document:
    """Đọc một file, trả về Document."""
    if path.suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"File is empty: {path}")

    return Document(
        content=content,
        source=str(path),
        metadata={"filename": path.name, "size_bytes": path.stat().st_size},
    )


def load_directory(directory: str | Path) -> list[Document]:
    """
    Đọc tất cả file hỗ trợ trong thư mục.
    Dùng generator nội bộ để không load hết RAM.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    docs = []
    errors = []

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(dir_path.rglob(f"*{ext}")):
            try:
                docs.append(load_file(file_path))
            except ValueError as e:
                errors.append(str(e))

    if errors:
        for err in errors:
            logger.warning("file_skipped", reason=err)

    return docs

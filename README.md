# DocChat CLI

Hỏi đáp tài liệu từ terminal — nhét file `.txt` / `.md` vào, hỏi bằng tiếng Việt, nhận câu trả lời từ LLM.

---

## Tính năng

- Index nhiều file văn bản cùng lúc
- Tìm kiếm theo ngữ nghĩa (semantic search) bằng embedding
- Trả lời dựa trên nội dung tài liệu, không bịa
- Stream token ra terminal từng chữ (không chờ hết)
- Lưu index ra file — không cần embed lại mỗi lần chạy

---

## Cài đặt

**Yêu cầu:** Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/yourname/docchat
cd docchat
uv sync
```

Tạo file `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
# hoặc dùng local embedding (không cần API key):
EMBEDDER=local
```

---

## Sử dụng

### 1. Index tài liệu

```bash
docchat index ./docs/
```

Đọc toàn bộ file `.txt` và `.md` trong thư mục, tách chunk, tạo embedding, lưu index vào `~/.docchat/index.pkl`.

### 2. Hỏi đáp

```bash
docchat ask "Quy trình onboarding nhân viên mới?"
```

Tìm các đoạn liên quan trong index, gửi cho LLM, stream câu trả lời ra terminal.

### 3. Xem thông tin index hiện tại

```bash
docchat info
```

---

## Ví dụ

```
$ docchat index ./company-docs/

Indexing 12 files...
  ✓ handbook.md         → 34 chunks
  ✓ onboarding.md       → 18 chunks
  ✓ benefits.txt        → 9 chunks
Index saved: 61 chunks total

$ docchat ask "Tôi có bao nhiêu ngày nghỉ phép mỗi năm?"

Theo chính sách công ty, nhân viên chính thức có 12 ngày nghỉ
phép có lương mỗi năm. Sau 3 năm làm việc, số ngày tăng lên
15 ngày. Nghỉ phép tích lũy được chuyển sang năm tiếp theo
tối đa 5 ngày...
```

---

## Cấu trúc project

```
docchat/
├── src/docchat/
│   ├── __init__.py
│   ├── loader.py      # đọc file, trả về generator Document
│   ├── chunker.py     # tách text thành Chunk có overlap
│   ├── embedder.py    # ABC BaseEmbedder + OpenAI / Local impl
│   ├── store.py       # SimpleVectorStore: lưu và tìm kiếm vector
│   ├── llm.py         # LLMSession context manager, async stream
│   └── cli.py         # entry point: lệnh index / ask / info
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   └── test_store.py
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Chạy test

```bash
uv run pytest --cov=docchat --cov-report=term-missing
```

Mục tiêu: coverage trên 80%.

---

## Thiết kế

| Module | Pattern áp dụng |
|---|---|
| `loader.py` | Generator pipeline |
| `chunker.py` | Generator, dataclass |
| `embedder.py` | ABC + 2 implementation, @retry, @lru_cache |
| `store.py` | Composition |
| `llm.py` | Context manager, async/await, Strategy |
| `cli.py` | Factory (tạo embedder từ config) |

---

## Roadmap

- [ ] Hỗ trợ file PDF
- [ ] Thay SimpleVectorStore bằng ChromaDB
- [ ] Conversation history (nhớ câu hỏi trước)
- [ ] Web UI đơn giản bằng FastAPI

---

## License

MIT
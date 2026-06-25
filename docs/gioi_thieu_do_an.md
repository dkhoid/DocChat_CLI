# Law Assistant — Giới Thiệu Đồ Án

> **Hệ thống Trợ lý Tra cứu Pháp luật Lao động & Dân sự Việt Nam bằng công nghệ RAG**
> Phiên bản: 1.0.0 · Python 3.11+ · Tối ưu Tiếng Việt · Hybrid Search

---

## 1. Mục Tiêu & Bài Toán

### 1.1 Bối cảnh

Hệ thống pháp luật Việt Nam đồ sộ với hơn 500.000 văn bản quy phạm, trong đó các bộ luật cốt lõi về Lao động, Bảo hiểm xã hội và Dân sự có mối liên kết chéo (cross-reference) phức tạp. Người lao động, sinh viên luật và doanh nghiệp thường gặp khó khăn khi tra cứu chính xác điều khoản liên quan đến quyền lợi và nghĩa vụ của mình.

### 1.2 Bài toán

Người dùng cần **đặt câu hỏi bằng ngôn ngữ tự nhiên** (tiếng Việt) về các vấn đề pháp luật lao động & dân sự, và nhận được câu trả lời:
- **Chính xác** — trích dẫn đúng Điều, Khoản, Luật cụ thể
- **Đầy đủ** — tổng hợp từ nhiều văn bản luật có liên quan (cross-reference)
- **Trung thực** — không bịa đặt (anti-hallucination), dẫn nguồn rõ ràng

### 1.3 Giải pháp

**Law Assistant** xây dựng một hệ thống **RAG (Retrieval-Augmented Generation)** chuyên biệt cho văn bản pháp luật Việt Nam, cho phép:
- Nạp và xử lý văn bản luật từ file PDF/HTML (giữ nguyên cấu trúc Chương/Mục/Điều/Khoản)
- Tìm kiếm kết hợp ngữ nghĩa + từ khóa (Hybrid Search) với metadata pháp lý
- Trả lời câu hỏi pháp luật qua giao diện dòng lệnh (CLI) với dẫn nguồn chính xác

---

## 2. Phạm Vi Corpus — "Bộ 3 Pháp luật Lao động"

> **Quyết định thiết kế:** Sử dụng 3 bộ luật có liên kết chặt chẽ thay vì 1 luật đơn lẻ, vì cross-reference giữa chúng xảy ra thường xuyên trong thực tế. Nếu chỉ dùng 1 luật, câu trả lời sẽ bị "đứt" ngữ cảnh.

| # | Văn bản | Số hiệu | Số điều |
|---|---------|---------|---------|
| 1 | **Bộ luật Lao động 2019** | 45/2019/QH14 | 220 |
| 2 | **Luật Bảo hiểm Xã hội 2014** (sửa đổi 2019) | 58/2014/QH13 | 141 |
| 3 | **Bộ luật Dân sự 2015** | 91/2015/QH13 | 689 |
| | **Tổng cộng** | | **~1.050 điều** |

**Sau chunking:** ~3.000 – 4.000 chunks → phù hợp ChromaDB, đủ cho người dùng thực tế.

---

## 3. Kiến Trúc Pipeline RAG

```
                        ┌──────────────────────────────────────────────────────┐
                        │               INDEXING PIPELINE                      │
                        │                                                      │
  Văn bản Pháp luật    │  loader.py ──→ cleaner.py ──→ LegalChunker           │
  (PDF / HTML)         │  (Đọc file)   (Chuẩn hóa     (Chia theo              │
  ────────────────────►│                Unicode VN,     Chương/Điều/Khoản,     │
                        │                loại header)    parent-child)          │
                        │                     │                                 │
                        │                     ▼                                 │
                        │  extractor.py ──→ embedder.py ──→ ChromaDB           │
                        │  (Trích metadata:   (Tạo vector)   (Lưu trữ vector  │
                        │   số hiệu, điều,                    + legal metadata)│
                        │   cross-reference)                                    │
                        └──────────────────────────────────────────────────────┘

                        ┌──────────────────────────────────────────────────────┐
                        │              RETRIEVAL PIPELINE (4 bước)             │
                        │                                                      │
  Câu hỏi pháp luật   │  [0] Pre-filter Metadata (status, law_domain)        │
  ────────────────────►│  [1] Dense Search   (ChromaDB — Semantic)            │
                        │  [2] Sparse Search  (BM25 + PyVi — Keyword)         │
                        │  [3] RRF Fusion     (Hợp nhất kết quả)              │
                        │  [4] CrossEncoder   (Chấm lại điểm đa ngữ)         │
                        └──────────────────┬───────────────────────────────────┘
                                           │ Top-K Chunks + Citations
                                           ▼
                        ┌──────────────────────────────────────────────────────┐
                        │             GENERATION PIPELINE                      │
                        │                                                      │
                        │  prompt_manager.py ──→ llm.py ──→ Streaming Answer  │
                        │  (Legal prompt VN)    (GPT-4o / Claude)              │
                        │                                    + Trích dẫn nguồn │
                        └──────────────────────────────────────────────────────┘
```

---

## 4. Sáu Thách Thức Kỹ Thuật Đặc Thù

Văn bản pháp luật Việt Nam đặt ra các thách thức mà hệ thống RAG tổng quát không xử lý được:

| # | Thách thức | Ví dụ cụ thể | Giải pháp kỹ thuật |
|---|------------|---------------|---------------------|
| 1 | **Chuỗi tham chiếu chéo** | Điều 35 → 36 → 37 → Luật BHXH | Shadow context + `metadata.references[]` |
| 2 | **Phiên bản thời gian** | Cùng điều luật, 2 phiên bản khác nhau | Metadata `status` (còn hiệu lực / hết hiệu lực) + filter |
| 3 | **Mệnh đề phủ định** | "PHẢI báo trước" vs "KHÔNG cần báo trước" | Hybrid Search — BM25 bắt được phủ định mà Dense bỏ sót |
| 4 | **Bảng biểu pháp luật** | Bảng thời gian làm việc bị vỡ cấu trúc khi OCR | `pdfplumber` table extraction → chuyển sang văn xuôi |
| 5 | **Đa nghĩa thuật ngữ** | "hợp đồng lao động" ≠ "hợp đồng dân sự" | Metadata `law_domain` → pre-filter trước khi search |
| 6 | **Cấu trúc lồng nhau** | Chunk điểm a, b, c bị thiếu ngữ cảnh khoản mẹ | Parent-child chunking — overlap = phần dẫn đầu khoản |

---

## 5. Công Nghệ Sử Dụng

| Lớp | Thư Viện / Công Nghệ | Vai Trò |
|---|---|---|
| **Vector DB** | `ChromaDB >= 1.5.5` | Lưu trữ vector embedding + legal metadata |
| **Sparse Search** | `rank-bm25` + `PyVi` | Tìm kiếm từ khóa với tokenizer tiếng Việt |
| **Embedding** | `sentence-transformers` / OpenAI API | Sinh vector ngữ nghĩa cho văn bản pháp luật |
| **Reranker** | `CrossEncoder mmarco-mMiniLMv2` | Chấm lại điểm độ phù hợp đa ngôn ngữ |
| **LLM** | OpenAI GPT-4o / Anthropic Claude | Sinh câu trả lời pháp luật từ context |
| **PDF Parsing** | `pdfplumber` | Trích xuất văn bản + bảng biểu từ file PDF luật |
| **HTML Parsing** | `beautifulsoup4` + `lxml` | Phân tích cấu trúc HTML từ thuvienphapluat.vn |
| **Legal Chunking** | Regex-based structural splitter | Chia theo Chương/Mục/Điều/Khoản (không dùng fixed-size) |
| **Token Counting** | `tiktoken` | Kiểm soát giới hạn token cho embedding & LLM |
| **Đánh giá RAG** | `RAGAS` | Chấm điểm Faithfulness, Relevancy, Precision |
| **REST API** | `FastAPI` + `Uvicorn` | Phơi bày chức năng tra cứu qua HTTP |
| **Observability** | `structlog` + `Langfuse` | Ghi log có cấu trúc, theo dõi LLM call |
| **Package Manager** | `uv` + `hatchling` | Quản lý dependency, chuẩn PEP 517 |
| **Containerization** | `Docker` + `Docker Compose` | Đóng gói và triển khai |
| **Testing** | `pytest` + `pytest-cov` | Unit test + kiểm tra regression |
| **Linting** | `ruff` + `pre-commit` | Kiểm tra chất lượng code tự động |

---

## 6. Cấu Trúc Module

```
src/docchat/
├── loader.py              # Đọc file PDF/HTML văn bản pháp luật
├── chunker.py             # Chia văn bản theo Token Limit (generic)
├── embedder.py            # Factory tạo embedding (Local / OpenAI)
├── store.py               # ChromaVectorStore: BM25 + Dense + RRF + CrossEncoder
├── llm.py                 # Kết nối LLM, quản lý memory, đếm token/chi phí
├── prompt_manager.py      # Đọc prompt từ file YAML, hỗ trợ tiếng Việt
├── api.py                 # Các endpoint FastAPI (index, ask, chat, info)
├── cli.py                 # Giao diện dòng lệnh
├── observability.py       # Tích hợp Langfuse tracing
├── server.py              # Entry point khởi động Uvicorn
│
└── legal/                 # ★ Module xử lý đặc thù Pháp luật VN
    ├── __init__.py
    ├── cleaner.py         # Chuẩn hóa Unicode tiếng Việt, loại header/footer
    ├── chunker.py         # LegalChunker: chia theo Chương/Điều/Khoản (parent-child)
    ├── extractor.py       # Trích xuất metadata + phát hiện cross-reference
    └── bm25_index.py      # BM25 index builder & searcher cho legal corpus
```

### Thư mục dữ liệu pháp luật

```
data/legal/
├── raw/                   # File gốc PDF/HTML (3 bộ luật)
├── processed/             # JSONL sau clean + chunk + metadata
└── metadata/
    └── corpus_info.json   # Thông tin tổng quan corpus
```

---

## 7. Metadata Schema Pháp luật

Mỗi chunk được gán metadata đầy đủ để hỗ trợ tìm kiếm chính xác:

```json
{
  "doc_id": "bllđ_2019_dieu_35_khoan_1",
  "source": {
    "law_name": "Bộ luật Lao động 2019",
    "law_number": "45/2019/QH14",
    "law_domain": "labor",
    "effective_date": "2021-01-01",
    "status": "in_effect"
  },
  "location": {
    "chapter": "IV",
    "chapter_name": "HỢP ĐỒNG LAO ĐỘNG",
    "article": "35",
    "clause": "1"
  },
  "references": ["Điều 36", "Điều 37", "BHXH Điều 19"],
  "content_type": "article_clause",
  "char_count": 850
}
```

---

## 8. Tính Năng Nổi Bật

### 🔍 Hybrid Search chuyên biệt Pháp luật
Kết hợp **Semantic Search** (hiểu ngữ nghĩa) và **BM25 Keyword Search** (bắt từ khóa, mệnh đề phủ định) rồi hợp nhất bằng **RRF (Reciprocal Rank Fusion)** — đảm bảo không bỏ sót kết quả dù câu hỏi diễn đạt theo cách nào. Hỗ trợ **pre-filter** theo metadata pháp lý (domain, hiệu lực, số hiệu luật).

### 📜 LegalChunker — Chia văn bản theo cấu trúc luật
Không dùng fixed-size chunking. Thay vào đó, nhận diện cấu trúc **Chương → Mục → Điều → Khoản → Điểm** bằng regex, giữ nguyên ngữ cảnh cha-con (parent-child) để tránh mất thông tin.

### 🌐 Tối Ưu Tiếng Việt
Tích hợp **PyVi** (tokenizer tiếng Việt) trực tiếp vào BM25 để ngắt từ chính xác, xử lý chuẩn hóa Unicode tiếng Việt (NFC/NFD), loại bỏ header/footer trang.

### 🔗 Phát hiện Tham chiếu chéo (Cross-Reference)
Tự động phát hiện các cụm như *"theo Điều X Luật Y"*, *"quy định tại Khoản Z"* và ghi nhận trong metadata `references[]` để hỗ trợ tra cứu liên văn bản.

### 🥇 CrossEncoder Reranking
Model `mmarco-mMiniLMv2` chấm lại từng cặp *(câu hỏi, đoạn luật)* để đảm bảo context nạp vào LLM là **chính xác nhất**, giảm nhiễu.

### 💬 Chat Đa Lượt (Memory)
Lưu lịch sử hội thoại, cho phép hỏi theo ngữ cảnh: *"Còn trường hợp nào được nghỉ không lương không?"*, *"So sánh với Luật BHXH đi"*.

### 💰 Tự Động Theo Dõi Chi Phí
Đếm token và tính hóa đơn USD sau mỗi lượt hỏi đáp.

### 📊 Đánh Giá Tự Động (RAGAS)
Chạy bộ 50 câu hỏi pháp luật chuẩn (ground truth), hệ thống tự chấm điểm và xuất báo cáo với các chỉ số: *Faithfulness*, *Answer Relevancy*, *Context Precision*.

---

## 9. Câu Hỏi Mẫu (Ground Truth)

```
# Loại 1 — Tra cứu đơn (Single-doc lookup)
Q: "Người lao động có thể nghỉ việc không cần báo trước trong trường hợp nào?"
A: BLLĐ 2019, Điều 37, Khoản 1 điểm a-g

# Loại 2 — Tổng hợp liên văn bản (Cross-doc reasoning)
Q: "Tôi mang thai 7 tháng, được nghỉ thai sản bao lâu và hưởng bao nhiêu?"
A: BLLĐ Điều 137 (quyền nghỉ) + BHXH Điều 34 (mức hưởng 100% lương bình quân)

# Loại 3 — Mệnh đề phủ định (Negative clause)
Q: "Công ty có thể sa thải tôi khi tôi đang điều trị bệnh không?"
A: BLLĐ Điều 37 Khoản 4 — CẤM chấm dứt HĐLĐ trong thời gian điều trị bệnh

# Loại 4 — Định lượng (Quantitative)
Q: "Làm thêm giờ tối đa bao nhiêu một năm?"
A: BLLĐ Điều 107 — không quá 200h/năm, trường hợp đặc biệt tối đa 300h
```

---

## 10. Cách Sử Dụng

```bash
# Cài đặt nhanh
uv sync

# Nạp corpus pháp luật
uv run python tools/ingest_legal.py

# Hỏi đáp đơn
uv run docchat ask "Người lao động nghỉ việc không cần báo trước khi nào?"

# Chat đa lượt về pháp luật
uv run docchat chat

# Khởi động REST API
uv run docchat-api

# Đánh giá chất lượng RAG
uv run python tools/evaluate_rag.py

# Triển khai Docker
docker compose up --build -d
```

---

## 11. Tiêu Chí Thành Công

### Giai đoạn ĐAN (Đồ Án Nhập Môn)
- [ ] Corpus: 3 bộ luật được index (~3.000 – 4.000 chunks)
- [ ] Demo CLI: trả lời được 50 câu hỏi pháp luật thực tế
- [ ] RAGAS Faithfulness ≥ 0.70
- [ ] RAGAS Answer Relevance ≥ 0.70
- [ ] Hybrid Search Recall@10 > Dense-only Recall@10

### Giai đoạn KLTN (Khóa Luận Tốt Nghiệp) — Mở rộng
- [ ] Corpus: ≥ 50 văn bản pháp luật (crawler tự động)
- [ ] So sánh ≥ 3 embedding models (PhoBERT vs multilingual-e5 vs OpenAI)
- [ ] Cross-reference resolution tự động
- [ ] Web UI hoạt động
- [ ] Báo cáo khoa học có contribution rõ ràng

---

## 12. Kiểm Thử & Chất Lượng

- Unit test phủ toàn bộ module (loader, chunker, embedder, store, llm, legal/*)
- Code coverage tracking với `pytest-cov`
- Tự động kiểm tra lint với `ruff` và `pre-commit` hook
- Pipeline RAGAS đánh giá chất lượng RAG end-to-end
- Bộ 50 câu hỏi ground truth được kiểm chứng thủ công

---

> **Kết luận:** Law Assistant là hệ thống RAG chuyên biệt cho **Pháp luật Lao động & Dân sự Việt Nam**, được thiết kế để xử lý chính xác các đặc thù của văn bản pháp luật (cấu trúc lồng nhau, tham chiếu chéo, mệnh đề phủ định, đa nghĩa thuật ngữ). Hệ thống tích hợp đầy đủ pipeline từ **Ingestion → Retrieval → Generation → Evaluation**, với kiến trúc module rõ ràng và bộ đánh giá RAGAS tự động.

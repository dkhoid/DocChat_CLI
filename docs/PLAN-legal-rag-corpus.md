# PLAN: DocChat × Pháp Luật Việt Nam — Legal RAG System

> **Phiên bản:** Draft 1.0  
> **Ngày tạo:** 2026-06-21  
> **Loại dự án:** BACKEND + NLP/RAG  
> **Scope:** ĐAN (Đồ Án Nhập Môn) → KLTN (Khóa Luận Tốt Nghiệp)

---

## 1. Tổng quan (Overview)

### 1.1 Bối cảnh

Hệ thống **DocChat CLI** hiện có đang là một RAG (Retrieval-Augmented Generation) framework đa năng. Theo góp ý của thầy hướng dẫn, cần thu hẹp về 1 lĩnh vực cụ thể để kết quả sâu và chính xác hơn. Domain được chọn: **Pháp luật Lao động & Dân sự Việt Nam**.

### 1.2 Lý do chọn domain này

- 📦 Data phong phú & miễn phí (vbpl.vn, thuvienphapluat.vn — 500K+ văn bản)
- 🔧 Khó xử lý đúng chỗ → tạo contribution kỹ thuật rõ ràng (cross-reference, versioning, nested structure)
- 🎓 Tác động xã hội cao → người dân tra cứu quyền lợi lao động
- 🛠️ Stack hiện tại (ChromaDB + LangChain + RAGAS + rank_bm25) hoàn toàn phù hợp
- 🔬 Tiềm năng nghiên cứu: Vietnamese Legal RAG (còn ít công trình)

### 1.3 Mục tiêu

| Giai đoạn | Mục tiêu | Deliverable |
|-----------|----------|-------------|
| **ĐAN** | Hệ thống hỏi đáp pháp luật lao động cơ bản | Demo CLI + Báo cáo |
| **KLTN** | Hệ thống RAG pháp luật đầy đủ + đánh giá khoa học | Web UI + Paper + Evaluation |

---

## 2. Thiết kế Corpus

### 2.1 Phạm vi ĐAN — "Bộ 3 Pháp luật Lao động"

> **Quyết định:** Dùng 3 bộ luật có liên kết chặt chẽ (cross-reference thường xuyên trong thực tế).  
> Không dùng 1 luật đơn lẻ vì cross-reference sẽ bị "đứt" — người dùng không có đủ thông tin.

| # | Văn bản | Số hiệu | Số trang | Số điều |
|---|---------|---------|----------|---------|
| 1 | **Bộ luật Lao động 2019** | 45/2019/QH14 | ~80 | 220 |
| 2 | **Luật BHXH 2014** (sửa đổi 2019) | 58/2014/QH13 | ~70 | 141 |
| 3 | **Bộ luật Dân sự 2015** | 91/2015/QH13 | ~130 | 689 |
| | **Tổng cộng** | | ~280 trang | ~1050 điều |

**Sau chunking:** ~3000-4000 chunks → phù hợp ChromaDB, đủ cho người dùng thực tế.

### 2.2 Nguồn thu thập (ĐAN: Thủ công)

```
data/legal/raw/
├── luat_lao_dong_2019.pdf        ← congbao.gov.vn
├── luat_lao_dong_2019.html       ← thuvienphapluat.vn
├── luat_bhxh_2014_sua_doi.pdf
├── luat_bhxh_2014_sua_doi.html
├── bo_luat_dan_su_2015.pdf
└── bo_luat_dan_su_2015.html
```

---

## 3. Pipeline Xử lý Data

### 3.1 Sơ đồ tổng thể

```
Raw (PDF/HTML)
    │
    ▼ loader.py [MODIFY]
    │
    ▼ legal/cleaner.py [MỚI]
    │   - Chuẩn hóa Unicode tiếng Việt
    │   - Loại bỏ header/footer trang
    │   - Convert bảng biểu → văn xuôi
    │
    ▼ legal/chunker.py [MỚI — LegalChunker]
    │   - Chunking theo Chương/Mục/Điều/Khoản
    │   - Parent-child: giữ context phần dẫn đầu Khoản
    │   - KHÔNG dùng fixed-size chunking
    │
    ▼ legal/extractor.py [MỚI]
    │   - Trích xuất: số hiệu, ngày ban hành, cơ quan
    │   - Phát hiện cross-references ("theo Điều X")
    │   - Gán: law_domain, status, effective_date
    │
    ▼ embedder.py + store.py [MODIFY]
        - ChromaDB với legal metadata fields
        - Hybrid Search: BM25 + Dense
```

### 3.2 Sáu Case Study kỹ thuật đặc thù pháp luật VN

| # | Case | Vấn đề | Giải pháp kỹ thuật |
|---|------|---------|---------------------|
| 1 | **Cross-Reference Chain** | Điều 35 → 36 → 37 → Luật BHXH... | Shadow context + metadata.references[] |
| 2 | **Temporal Versioning** | Cùng điều luật, 2 phiên bản khác nhau | Metadata: status (in_effect/repealed) + filter trước search |
| 3 | **Negative Clause** | "PHẢI báo trước" vs "KHÔNG cần báo trước" | Hybrid Search BM25 + Dense (Dense miss negative clause) |
| 4 | **Bảng biểu sau OCR** | Bảng thời gian làm việc bị vỡ cấu trúc | pdfplumber table extraction → convert sang văn xuôi |
| 5 | **Polysemy** | "hợp đồng lao động" ≠ "hợp đồng dân sự" | Metadata law_domain → pre-filter collection |
| 6 | **Nested Structure** | Chunk điểm a,b,c thiếu context khoản mẹ | Parent-child chunking — overlap = phần dẫn đầu khoản |

### 3.3 Metadata Schema

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
  "text": "...",
  "char_count": 850
}
```

---

## 4. Kiến trúc Tìm kiếm

### 4.1 Hybrid Search (BẮT BUỘC)

> Dense embedding một mình không đủ cho văn bản pháp luật (Case 3 — negative clause).

```
Query
  │
  ├─→ BM25 Search (keyword)      → bắt được "không cần", "cấm", số điều cụ thể
  │
  └─→ Dense Search (embedding)   → bắt được paraphrase, ngữ nghĩa
          │
          └─→ RRF Fusion (alpha tunable)
                  │
                  └─→ Top-K results → LLM → Response + Citations
```

### 4.2 Search Flow chi tiết

```python
def legal_search(query, domain=None, k=10):
    # Step 1: Pre-filter theo metadata
    filters = {"status": "in_effect"}
    if domain:
        filters["law_domain"] = domain

    # Step 2: Hybrid search
    dense = chroma.query(embed(query), where=filters, n=k*2)
    bm25  = bm25_index.search(query, filters, n=k*2)

    # Step 3: RRF Fusion
    return reciprocal_rank_fusion(dense, bm25, k=k)
```

---

## 5. Cấu trúc File

```
DocChat_CLI/
├── data/
│   └── legal/                     [MỚI]
│       ├── raw/                   ← File gốc PDF/HTML
│       ├── processed/             ← JSONL sau clean + chunk
│       └── metadata/
│           └── corpus_info.json
│
├── src/docchat/
│   ├── loader.py                  [MODIFY] Thêm PDF + HTML legal loader
│   ├── chunker.py                 [MODIFY] Thêm LegalChunker class
│   ├── embedder.py                [MODIFY] Thêm legal metadata fields
│   ├── store.py                   [MODIFY] Thêm hybrid_search()
│   └── legal/                     [MỚI]
│       ├── __init__.py
│       ├── cleaner.py             ← Xử lý đặc thù VN (Unicode, bảng biểu)
│       ├── chunker.py             ← LegalChunker (parent-child)
│       ├── extractor.py           ← Metadata + cross-reference detector
│       └── bm25_index.py          ← BM25 index builder & searcher
│
├── tools/
│   ├── ingest_legal.py            [MỚI] Script ingest toàn bộ corpus
│   └── crawlers/                  [KLTN]
│       ├── vbpl_crawler.py
│       └── thuvienphapluat_crawler.py
│
└── tests/legal/                   [MỚI]
    ├── test_cleaner.py
    ├── test_chunker.py
    └── test_hybrid_search.py
```

---

## 6. Lộ trình thực hiện

### 🎓 ĐAN — 10 tuần

| Tuần | Việc làm | Output | Verify |
|------|----------|--------|--------|
| 1 | Thu thập data thủ công (3 bộ luật) | 6 file (PDF+HTML) × 3 | File tồn tại, nội dung đúng |
| 2 | Viết `cleaner.py` | Clean text chuẩn UTF-8 | Unit test: không mất dữ liệu |
| 3 | Viết `LegalChunker` | JSONL chunks có metadata | Chunk giữ nguyên số điều luật |
| 4 | Viết `extractor.py` | Metadata JSON đầy đủ | Detect được cross-references |
| 5 | Implement BM25 index | BM25 index file | Keyword search recall > 80% |
| 6 | Implement Hybrid Search | Hybrid results | Case 3 negative clause handled |
| 7 | Ingest vào ChromaDB | ~3500 chunks indexed | Demo 10 câu hỏi cơ bản |
| 8 | Tạo 50 câu hỏi test (ground truth) | QA dataset JSON | Verified by human |
| 9 | Đánh giá RAGAS | Metrics report | Faithfulness ≥ 0.7 |
| 10 | Viết báo cáo ĐAN | Báo cáo hoàn chỉnh | Nộp đúng hạn |

### 📚 KLTN — Mở rộng từ ĐAN

| Giai đoạn | Nội dung | Contribution |
|-----------|----------|-------------|
| P1 | Crawler tự động → 50+ văn bản | Vietnamese Legal Corpus (large-scale) |
| P2 | So sánh embedding: PhoBERT vs multilingual-e5 vs OpenAI | Empirical comparison for legal VN text |
| P3 | Cross-reference resolution tự động | Novel pipeline component |
| P4 | LangGraph multi-step legal QA agent | Complex reasoning |
| P5 | Web UI (đã có `/web` directory) | End-to-end usable system |
| P6 | Báo cáo KLTN | Scientific paper |

---

## 7. Stack kỹ thuật

| Thành phần | Công nghệ | Trạng thái |
|-----------|-----------|-----------|
| Vector DB | ChromaDB | ✅ Đã có |
| Embedding | OpenAI text-embedding-3-small | ✅ Đã có |
| Keyword Search | rank_bm25 | ✅ Đã có trong pyproject.toml |
| LLM | GPT-4o / Claude | ✅ Đã có |
| Evaluation | ragas | ✅ Đã có |
| Orchestration | LangChain + LangGraph | ✅ Đã có |
| PDF Parsing | pdfplumber | ❌ Cần cài thêm |
| HTML Parsing | beautifulsoup4 | ❌ Cần cài thêm |

---

## 8. Câu hỏi test mẫu (Ground Truth)

```
# Loại 1 — Single-doc lookup
Q: "Người lao động có thể nghỉ việc không cần báo trước trong trường hợp nào?"
A: BLLĐ 2019, Điều 37, Khoản 1 điểm a-g

# Loại 2 — Cross-doc reasoning (cần 2 luật)
Q: "Tôi mang thai 7 tháng, được nghỉ thai sản bao lâu và hưởng bao nhiêu?"
A: BLLĐ Điều 137 (quyền nghỉ) + BHXH Điều 34 (mức hưởng 100% lương bình quân)

# Loại 3 — Negative clause
Q: "Công ty có thể sa thải tôi khi tôi đang điều trị bệnh không?"
A: BLLĐ Điều 37 Khoản 4 — CẤM chấm dứt HĐLĐ trong thời gian điều trị bệnh

# Loại 4 — Định lượng
Q: "Làm thêm giờ tối đa bao nhiêu một năm?"
A: BLLĐ Điều 107 — không quá 200h/năm, trường hợp đặc biệt tối đa 300h
```

---

## 9. Rủi ro & Phòng ngừa

| Rủi ro | Mức độ | Phòng ngừa |
|--------|--------|-----------|
| PDF scan OCR chất lượng thấp | Cao | Ưu tiên dùng HTML (thuvienphapluat.vn luôn có HTML tốt hơn) |
| Cross-reference quá phức tạp ở ĐAN | Trung bình | Ghi nhận trong metadata, chưa auto-resolve — để KLTN |
| RAGAS score thấp với câu hỏi pháp lý | Trung bình | Chia câu hỏi test theo độ khó (easy/medium/hard) |
| API cost embedding 3500+ chunks | Thấp | text-embedding-3-small rẻ nhất; cache kết quả |

---

## 10. Tiêu chí Thành công

### ĐAN
- [ ] Corpus: 3 bộ luật được index (~3000-4000 chunks)
- [ ] Demo CLI: trả lời được 50 câu hỏi thực tế
- [ ] RAGAS Faithfulness ≥ 0.70
- [ ] RAGAS Answer Relevance ≥ 0.70
- [ ] Hybrid search Recall@10 > Dense-only

### KLTN
- [ ] Corpus: ≥ 50 văn bản pháp luật
- [ ] So sánh ≥ 3 embedding models có kết quả định lượng
- [ ] Hybrid recall cải thiện ≥ 10% vs dense-only
- [ ] Web UI hoạt động
- [ ] Báo cáo khoa học có contribution rõ ràng

---

## 11. Câu hỏi mở — Cần thảo luận thêm

1. **Corpus scope:** Thu hẹp theo chủ đề (chỉ lấy phần liên quan lao động từ 3 luật) hay lấy toàn bộ 3 luật?
2. **Hybrid Search:** Thảo luận chi tiết cách chọn alpha (trọng số BM25 vs Dense)?
3. **Evaluation:** Tự xây 50 câu hỏi thủ công hay dùng LLM generate từ tài liệu?
4. **KLTN advisor:** Thầy/cô hướng dẫn KLTN có background NLP/ML hay Engineering?

---

*Plan: `docs/PLAN-legal-rag-corpus.md` | Ngày: 2026-06-21 | Agent: project-planner*

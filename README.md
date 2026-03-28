# DocChat CLI — Advanced RAG System

Hệ thống hỏi đáp tài liệu cấp doanh nghiệp chạy trực tiếp trên Terminal. Được thiết kế đặc biệt tối ưu cho **Tiếng Việt**, kết hợp hoàn hảo công nghệ Hybrid Search, Multilingual Reranking và sinh AI chống ảo giác mạnh mẽ.

---

## Tính năng Nổi bật 

- **Vector Database Hiện Đại**: Storage bền vững bằng **ChromaDB**, không lo tràn RAM, xoá bỏ phương pháp pickle lỗi thời.
- **Hybrid Retrieval (Dense + Sparse)**: Tích hợp Semantic Search (nhúng bằng OpenAI/SentencesTransformers) + Cỗ máy từ khoá **BM25**, kết hợp bộ phận ngắt từ tiếng Việt **PyVi**.
- **Thuật toán RRF (Reciprocal Rank Fusion)**: Ghép chồng thông minh kết quả từ Vector và Khoá (Keywords) để có độ phủ 100% ngữ nghĩa.
- **Node Lọc Đa Ngôn Ngữ (Cross-Encoder Reranker)**: Sử dụng model `mmarco-mMiniLMv2` chấm lại điểm từng Chunk trước khi giao cho LLM để tạo sự chuẩn xác tuyệt đối.
- **Tự động Quản Lý Chi Phí**: Tính toán hoá đơn USD tự động ở mỗi lượt hỏi đáp (Track Tokens).
- **Hệ thống Đánh giá RAGAS (LLM-as-a-Judge)**: Có hẳn công cụ chạy bài thi tự động trả về điểm số độ tin cậy bằng file CSV.
- **Chat Đa Lượt (Memory)**: Lưu giữ ngữ cảnh cuộc trò chuyện với lịch sử hội thoại (`docchat chat`).

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
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...

# (Không bắt buộc) Lựa chọn Embedder: local hoặc openai
EMBEDDER=local
```

---

## Cách Sử Dụng

### 1. Nạp tài liệu (Indexing)
Bạn trỏ vào thư mục chứa dải file `.txt` / `.md`. Hệ thống sẽ băm nội dung theo token và đưa vào csdl `ChromaDB`.

```bash
uv run docchat index ./my_docs/ --embedder openai
```
*Lưu ý: Nếu không thả cờ `--embedder`, hệ thống mặc định sẽ down weight `sentence-transformers` dùng CPU để chạy local offline.*

### 2. Trò chuyện nhanh (Single-turn Ask)
Hỏi trực tiếp 1 câu, in đáp án siêu tốc dạng Streaming (Từng chữ chạy ra).
```bash
uv run docchat ask "Chiến lược tăng doanh số quý 3 là gì?" --embedder openai
```

### 3. Trò chuyện sâu (Multi-turn Chat)
Đăng nhập giao diện Chat REPL để có bộ nhớ ngữ cảnh cực khôn:
```bash
uv run docchat chat --embedder openai
```
- Các lệnh phụ hỗ trợ: `/clear` (xoá bộ nhớ ngữ cảnh), `/stats` (xem tiền đã tiêu tốn), `/exit` (thoát).

### 4. Kiểm tra sức khoẻ DB (Info)
Bạn cần kiểm toán xem DB Chroma đã cắn bao nhiêu nghìn chunks:
```bash
uv run docchat info --embedder openai
```

---

## Hệ thống Kiểm chuẩn (RAGAS Evaluation)
Nếu bạn phân vân về chất lượng Bot, hãy tạo một mảng bộ test nhỏ tên `test_data.json`:
```json
[
  {
    "question": "Tính năng chính là gì?",
    "ground_truth": "Là Hybrid Search tích hợp Chroma DB."
  }
]
```
Gõ lệnh để mời "Giám Khảo" chấm thi (sẽ xuất ra file Report CSV cho bạn):
```bash
uv run python scripts/evaluate_rag.py --dataset test_data.json --embedder openai
```

---

## Thiết kế Kiến trúc (Architecture)

DocChat được chia cắt thành các mô-đun siêu rành mạch:
- `loader.py`: Đọc Data, quét sâu cây thư mục.
- `chunker.py`: Băm dữ liệu theo mật độ Token Limit bảo vệ dung lượng gởi đi.
- `store.py`: **ChromaVectorStore** — Interface của Chroma, Hybrid BM25, Hàm gộp điểm RRF, và Cross-Encoder Reranker Node.
- `embedder.py`: Cơ sở Factory Sinh mã Embedding siêu tối ưu, dễ mở rộng.
- `llm.py`: Logic đếm token, quản lý Memory và tạo kết nối đến OpenAI/Anthropic.
- `prompt_manager.py`: Mạch Prompt Tiếng Việt lưu gốc qua YAML dễ chỉnh sửa.

---

## Script Tiện Ích Khác
- **Test Hệ thống (130+ bài kiểm thử):** Chạy `uv run pytest` để chắc chắn không có lỗi thoái hoá (Regression Test).
- **Migration Data Cũ:** Chạy `uv run python scripts/migrate_to_chroma.py` để nhấc Pickle RAM cũ bỏ lên Chroma DB chuyên nghiệp nếu bạn nâng cấp từ Version bé hơn.

*Project được hoàn thiện đầy đủ cho việc mang đi chinh chiến trên Web hay API Backend.*
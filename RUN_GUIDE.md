# Hướng dẫn chạy DocChat_CLI (REST API & Docker)

Tài liệu này hướng dẫn bạn cách khởi động API server ở môi trường Local (dev) và qua Docker.

---

## Cách 1: Chạy trực tiếp bằng `uv` (Local Development)

Cách này phù hợp khi bạn đang viết code, test API và muốn server tự động reload khi sửa code.

**1. Đảm bảo bạn đã đồng bộ dependencies:**
```bash
uv sync
```

**2. Khởi động API server (có reload):**
```bash
# Thiết lập biến môi trường để hỗ trợ Unicode trên Windows và bật reload
$env:PYTHONIOENCODING="utf-8"
$env:DOCCHAT_RELOAD="true"

# Chạy server
uv run docchat-api
```

**3. Kiểm tra:**
Mở trình duyệt và truy cập:
- Swagger UI (để test API): [http://localhost:8000/docs](http://localhost:8000/docs)
- API Health check: [http://localhost:8000/health](http://localhost:8000/health)

---

## Cách 2: Chạy bằng Docker Compose (Dành cho Deploy / MLOps)

Cách này sẽ đóng gói ứng dụng vào một container độc lập, đồng thời tự động mount thư mục `data` để giữ lại dữ liệu ChromaDB giữa các lần restart.

**1. Đảm bảo bạn đã cài đặt Docker Desktop** (đang chạy ngầm).

**2. Build và khởi động bằng lệnh:**
```bash
docker compose up --build -d
```
*Lưu ý: Flag `-d` giúp chạy ngầm (detached mode).*

**3. Xem logs của container (nếu cần):**
```bash
docker compose logs -f
```

**4. Dừng container:**
```bash
docker compose down
```

---

## Ví dụ sử dụng API sau khi chạy

Dù bạn chạy bằng Cách 1 hay Cách 2, API đều mở ở `http://localhost:8000`.

**Ví dụ 1: Index một thư mục**
Gửi `POST /index` với body:
```json
{
  "directory": "./my_docs",
  "embedder": "local"
}
```

**Ví dụ 2: Chat với mô hình (OpenAI)**
Gửi `POST /ask` với body:
```json
{
  "query": "Tóm tắt tài liệu này",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "stream": false
}
```

> **Lưu ý:** Nếu bạn dùng OpenAI hoặc Anthropic, hãy chắc chắn đã điền API key vào file `.env` (ví dụ: `OPENAI_API_KEY=sk-...`).

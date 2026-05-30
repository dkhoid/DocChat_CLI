# API + Docker Setup (Phase 1)

## Goal
Wrap DocChat CLI thành FastAPI REST API + đóng gói Docker container.

## Tasks
- [x] Task 1: Tạo `src/docchat/api.py` — FastAPI app với endpoints (`/health`, `/index`, `/upload-and-index`, `/ask`, `/chat/*`, `/info`) → ✅ Done
- [x] Task 2: Cập nhật `pyproject.toml` — thêm `fastapi`, `uvicorn`, `python-multipart` + script `docchat-api` → ✅ `uv sync` OK
- [x] Task 3: Tạo `Dockerfile` — multi-stage build (builder + runtime) dùng `uv` + `python:3.13-slim` → ✅ Done
- [x] Task 4: Tạo `docker-compose.yml` — service docchat + volume ChromaDB + env_file → ✅ Done
- [x] Task 5: Tạo `.dockerignore` — exclude `.venv`, `.git`, `__pycache__`, `.agent` → ✅ Done
- [x] Task 6: Test toàn bộ — 137 passed, 0 failed (105s) → ✅ No breaking changes

## Done When
- [x] API chạy được local
- [x] Docker files tạo xong
- [x] Test cũ vẫn pass 100%

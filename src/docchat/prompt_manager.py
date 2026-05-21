"""
PromptManager — quản lý prompt template tập trung.

Tính năng:
- Load template từ file YAML trong thư mục prompts/
- Render template với biến động (Jinja2-style dùng str.format_map)
- Đếm token chính xác bằng tiktoken
- Trim context để không vượt ngưỡng token budget
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Thư mục mặc định chứa template
_DEFAULT_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


class PromptManager:
    """
    Quản lý prompt template từ file YAML.

    Cấu trúc YAML:
        system: |
            Bạn là trợ lý ...
        user: |
            Câu hỏi: {query}
            Context: {context}
    """

    def __init__(self, prompt_dir: str | Path | None = None):
        self._dir = Path(prompt_dir) if prompt_dir else _DEFAULT_PROMPT_DIR
        self._cache: dict[str, dict] = {}

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, name: str) -> dict[str, str]:
        """
        Load template theo tên (không cần .yaml).
        Kết quả được cache trong bộ nhớ.

        Returns:
            dict với keys 'system' và/hoặc 'user'.
        """
        if name in self._cache:
            return self._cache[name]

        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template không tìm thấy: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Template '{name}' phải là dict YAML, nhận được: {type(data)}")

        self._cache[name] = data
        return data

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self, name: str, **kwargs) -> dict[str, str]:
        """
        Render template với variables.

        Args:
            name: Tên template (không cần .yaml).
            **kwargs: Biến để inject vào template.

        Returns:
            dict với keys đã được render.

        Example:
            pm.render("qa_rag", query="Python là gì?", context="...")
        """
        template = self.load(name)
        rendered: dict[str, str] = {}

        for key, value in template.items():
            if isinstance(value, str):
                try:
                    rendered[key] = value.format_map(kwargs)
                except KeyError as e:
                    raise KeyError(
                        f"Template '{name}' cần biến {e} nhưng không được cung cấp."
                    ) from e
            else:
                rendered[key] = str(value)

        return rendered

    # ── Token counting ────────────────────────────────────────────────────────

    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """
        Đếm số token chính xác bằng tiktoken.
        Fallback về ước tính nếu tiktoken không hỗ trợ model.
        """
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback: ~4 chars per token với tiếng Anh
            return len(text) // 4

    def count_messages_tokens(self, messages: list[dict], model: str = "gpt-4o-mini") -> int:
        """Đếm tổng token của messages array (theo OpenAI convention)."""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""), model)
            total += 4  # overhead mỗi message (role, separators)
        total += 2  # overhead toàn bộ conversation
        return total

    # ── Trim context ──────────────────────────────────────────────────────────

    def trim_to_budget(
        self,
        context: str,
        budget: int,
        model: str = "gpt-4o-mini",
        truncation_marker: str = "\n\n[... nội dung bị cắt bớt do giới hạn context ...]",
    ) -> str:
        """
        Cắt context để không vượt quá số token cho phép.
        Cắt tại ranh giới từ (không cắt giữa chừng).

        Args:
            context: Đoạn văn bản cần cắt.
            budget: Số token tối đa.
            model: Model để đếm token.
            truncation_marker: Dấu hiệu cắt bớt thêm vào cuối.

        Returns:
            Context đã được trim (hoặc nguyên bản nếu đủ ngân sách).
        """
        if self.count_tokens(context, model) <= budget:
            return context

        # Cắt dần cho đến khi vừa budget
        # Thử cắt theo đoạn (paragraph) trước
        paragraphs = context.split("\n\n")
        result_parts: list[str] = []
        used_tokens = self.count_tokens(truncation_marker, model)

        for para in paragraphs:
            para_tokens = self.count_tokens(para, model)
            if used_tokens + para_tokens > budget:
                break
            result_parts.append(para)
            used_tokens += para_tokens

        if result_parts:
            return "\n\n".join(result_parts) + truncation_marker

        # Nếu không cắt được theo paragraph, cắt thô theo ký tự
        # Ước tính: 1 token ≈ 4 chars
        approx_chars = budget * 4
        return context[:approx_chars] + truncation_marker

    # ── List available templates ───────────────────────────────────────────────

    def list_templates(self) -> list[str]:
        """Liệt kê tất cả template có sẵn."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.yaml")]


# ── Singleton ─────────────────────────────────────────────────────────────────

# Instance mặc định dùng prompts/ trong thư mục gốc project
_default_manager: PromptManager | None = None


def get_prompt_manager(prompt_dir: str | Path | None = None) -> PromptManager:
    """Trả về PromptManager singleton hoặc tạo mới với custom dir."""
    global _default_manager
    if prompt_dir is not None:
        return PromptManager(prompt_dir)
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager

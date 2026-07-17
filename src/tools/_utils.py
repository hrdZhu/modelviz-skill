"""Shared helpers for template matching tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import TemplateMatchingError


def error_result(
    error_type: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    suggestion: str = "",
) -> dict[str, Any]:
    """构造统一错误结果。"""

    return TemplateMatchingError(
        error_type=error_type,
        message=message,
        details=details or {},
        suggestion=suggestion,
    ).model_dump()


def validation_error_result(error: ValidationError, *, suggestion: str) -> dict[str, Any]:
    """将 Pydantic 校验错误转成结构化错误。"""

    return error_result(
        "validation_error",
        "Pydantic 校验失败。",
        details={"errors": error.errors()},
        suggestion=suggestion,
    )


def normalize_text(value: Any) -> str:
    """简单文本标准化：去空格、英文小写，不拆中文字符。"""

    return str(value or "").strip().lower()


def unique_non_empty(values: list[Any]) -> list[str]:
    """去重、去空字符串，并保留顺序。"""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            result.append(str(value).strip())
            seen.add(text)
    return result


def text_matches(term: str, candidate: str) -> bool:
    """支持完全匹配和包含匹配。"""

    left = normalize_text(term)
    right = normalize_text(candidate)
    return bool(left and right and (left == right or left in right or right in left))


def any_text_matches(term: str, candidates: list[str]) -> bool:
    return any(text_matches(term, candidate) for candidate in candidates)


def path_for_error(path: Path | str) -> str:
    return str(Path(path))

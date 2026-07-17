"""Tool: load and validate user requirement JSON."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas import PlotRequirement
from src.schemas import LoadUserRequirementInput
from src.tools._utils import error_result, path_for_error, validation_error_result


@tool(
    "load_user_requirement",
    args_schema=LoadUserRequirementInput,
    description=(
        "读取并验证 workspace/user_requirement.json。"
        "该工具只负责读取和验证用户需求，不负责模板匹配。"
    ),
)
def load_user_requirement(requirement_path: str = "workspace/user_requirement.json") -> dict:
    """读取用户需求文件，并使用 PlotRequirement 进行校验。"""

    path = Path(requirement_path)
    if not path.exists():
        return error_result(
            "file_not_found",
            "用户需求文件不存在。",
            details={"requirement_path": path_for_error(path)},
            suggestion="请先运行阶段二需求解析流程，生成 workspace/user_requirement.json。",
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return error_result(
            "json_decode_error",
            "用户需求 JSON 格式错误。",
            details={"requirement_path": path_for_error(path), "error": str(error)},
            suggestion="请检查 JSON 是否完整、是否存在多余日志文本或非法字符。",
        )

    try:
        requirement = PlotRequirement.model_validate(data)
    except ValidationError as error:
        return validation_error_result(
            error, suggestion="请按 PlotRequirement Schema 修正用户需求文件。"
        )

    return {"success": True, "requirement": requirement.model_dump()}

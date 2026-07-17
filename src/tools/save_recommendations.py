"""Tool: save template recommendations to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import SaveTemplateRecommendationsInput, TemplateRecommendationOutput
from src.tools._utils import error_result


@tool(
    "save_template_recommendations",
    args_schema=SaveTemplateRecommendationsInput,
    description=(
        "将最终推荐结果保存为 workspace/template_recommendations.json。"
        "保存前使用 Pydantic 校验，JSON 使用 UTF-8 且中文不转义。"
    ),
)
def save_template_recommendations(
    recommendation_result: dict,
    output_path: str = "workspace/template_recommendations.json",
) -> dict:
    """保存推荐结果，不混入日志文本。"""

    try:
        SaveTemplateRecommendationsInput(
            recommendation_result=recommendation_result,
            output_path=output_path,
        )
        output = TemplateRecommendationOutput.model_validate(recommendation_result)
    except ValidationError as error:
        return error_result(
            "validation_error",
            "推荐结果保存前校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查 recommendation_result 是否符合 TemplateRecommendationOutput。",
        )

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return error_result(
            "output_directory_error",
            "输出目录无法创建。",
            details={"output_path": str(path), "error": str(error)},
            suggestion="请检查输出路径和目录权限。",
        )

    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(output.model_dump(), file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError as error:
        return error_result(
            "save_failed",
            "推荐结果保存失败。",
            details={"output_path": str(path), "error": str(error)},
            suggestion="请检查文件是否被占用、路径是否合法或磁盘权限是否足够。",
        )

    return {"success": True, "output_path": str(path), "recommendation_result": output.model_dump()}

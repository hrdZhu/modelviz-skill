"""阶段三模板匹配固定流程。

按顺序调用 LangChain tools：
load_user_requirement -> load_template_catalog -> match_templates -> rank_templates
-> save_template_recommendations。
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.schemas import PipelineInput
from src.tools import (
    load_template_catalog,
    load_user_requirement,
    match_templates,
    rank_templates,
    save_template_recommendations,
)
from src.tools._utils import error_result


def _invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """统一调用 tool，并把参数校验异常转成结构化错误。"""

    try:
        result = tool_obj.invoke(payload)
    except ValidationError as error:
        return error_result(
            "validation_error",
            f"{tool_obj.name} 工具参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查工具输入参数是否符合 Pydantic args_schema。",
        )
    except Exception as error:  # noqa: BLE001
        return error_result(
            "tool_execution_error",
            f"{tool_obj.name} 工具执行失败。",
            details={"error": str(error)},
            suggestion="请查看输入文件路径、文件内容和工具参数。",
        )
    if isinstance(result, dict):
        return result
    return error_result(
        "invalid_tool_result",
        f"{tool_obj.name} 返回了非结构化结果。",
        details={"result_type": type(result).__name__},
        suggestion="请确保工具返回可序列化 dict。",
    )


def _stop_if_failed(step: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("success") is True:
        return None
    return {
        "success": False,
        "failed_step": step,
        "error": result,
    }


def run_template_matching_pipeline(
    requirement_path: str = "workspace/user_requirement.json",
    catalog_path: str = "docs/template_catalog.yaml",
    top_k: int = 5,
    min_score: float = 0.15,
    output_path: str = "workspace/template_recommendations.json",
) -> dict[str, Any]:
    """运行阶段三固定多工具流程。

    本入口实际调用 @tool 创建的工具，不绕过工具函数。
    任一步失败都会停止后续流程并返回结构化错误。
    """

    try:
        params = PipelineInput(
            requirement_path=requirement_path,
            catalog_path=catalog_path,
            top_k=top_k,
            min_score=min_score,
            output_path=output_path,
        )
    except ValidationError as error:
        return {
            "success": False,
            "failed_step": "validate_pipeline_input",
            "error": error_result(
                "validation_error",
                "阶段三入口参数校验失败。",
                details={"errors": error.errors()},
                suggestion="请确认 top_k 大于 0 且不超过 20，min_score 在 0 到 1 之间。",
            ),
        }

    requirement_result = _invoke_tool(
        load_user_requirement,
        {"requirement_path": params.requirement_path},
    )
    failed = _stop_if_failed("load_user_requirement", requirement_result)
    if failed:
        return failed

    catalog_result = _invoke_tool(
        load_template_catalog,
        {"catalog_path": params.catalog_path},
    )
    failed = _stop_if_failed("load_template_catalog", catalog_result)
    if failed:
        return failed

    match_result = _invoke_tool(
        match_templates,
        {
            "user_requirement": requirement_result["requirement"],
            "templates": catalog_result["templates"],
            "min_score": params.min_score,
        },
    )
    failed = _stop_if_failed("match_templates", match_result)
    if failed:
        return failed

    rank_result = _invoke_tool(
        rank_templates,
        {"candidates": match_result["candidates"], "top_k": params.top_k},
    )
    failed = _stop_if_failed("rank_templates", rank_result)
    if failed:
        return failed

    ranked_output = rank_result["ranked_result"]
    ranked_output["original_request"] = match_result["original_request"]
    ranked_output["total_templates"] = match_result["total_templates"]
    ranked_output["qualified_templates"] = match_result["qualified_templates"]
    ranked_output["warnings"] = match_result.get("warnings", [])

    save_result = _invoke_tool(
        save_template_recommendations,
        {
            "recommendation_result": ranked_output,
            "output_path": params.output_path,
        },
    )
    failed = _stop_if_failed("save_template_recommendations", save_result)
    if failed:
        return failed

    return {
        "success": True,
        "output_path": save_result["output_path"],
        "recommendation_result": save_result["recommendation_result"],
    }

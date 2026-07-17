"""阶段三候选模板召回固定流程。"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.schemas import CandidatePipelineInput
from src.tools import (
    load_template_catalog,
    load_user_requirement,
    match_candidate_templates,
    rank_candidate_templates,
    save_candidate_templates,
)
from src.tools._utils import error_result


def _invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """统一调用 @tool 对象，并返回结构化错误。"""

    try:
        result = tool_obj.invoke(payload)
    except ValidationError as error:
        return error_result(
            "validation_error",
            f"{tool_obj.name} 工具参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查工具输入是否符合 args_schema。",
        )
    except Exception as error:  # noqa: BLE001
        return error_result(
            "tool_execution_error",
            f"{tool_obj.name} 工具执行失败。",
            details={"error": str(error)},
            suggestion="请检查输入文件、路径和权限。",
        )
    return (
        result
        if isinstance(result, dict)
        else error_result(
            "invalid_tool_result",
            f"{tool_obj.name} 返回非结构化结果。",
            details={"result_type": type(result).__name__},
            suggestion="请确保工具返回 dict。",
        )
    )


def _failed(step: str, result: dict[str, Any]) -> dict[str, Any] | None:
    if result.get("success") is True:
        return None
    return {"success": False, "failed_step": step, "error": result}


def run_candidate_matching_pipeline(
    requirement_path: str = "workspace/user_requirement.json",
    catalog_path: str = "docs/template_catalog.yaml",
    output_path: str = "workspace/candidate_templates.json",
    top_k: int = 8,
    min_score: float = 0.10,
) -> dict[str, Any]:
    """按固定顺序调用阶段三 tools，输出 candidate_templates.json。"""

    try:
        params = CandidatePipelineInput(
            requirement_path=requirement_path,
            catalog_path=catalog_path,
            output_path=output_path,
            top_k=top_k,
            min_score=min_score,
        )
    except ValidationError as error:
        return {
            "success": False,
            "failed_step": "validate_pipeline_input",
            "error": error_result(
                "validation_error",
                "阶段三候选召回入口参数校验失败。",
                details={"errors": error.errors()},
                suggestion="请确认 top_k 在 5 到 10 之间，min_score 在 0 到 1 之间。",
            ),
        }

    requirement_result = _invoke_tool(
        load_user_requirement, {"requirement_path": params.requirement_path}
    )
    if failed := _failed("load_user_requirement", requirement_result):
        return failed

    catalog_result = _invoke_tool(load_template_catalog, {"catalog_path": params.catalog_path})
    if failed := _failed("load_template_catalog", catalog_result):
        return failed

    match_result = _invoke_tool(
        match_candidate_templates,
        {
            "user_requirement": requirement_result["requirement"],
            "templates": catalog_result["templates"],
            "min_score": params.min_score,
        },
    )
    if failed := _failed("match_candidate_templates", match_result):
        return failed

    rank_result = _invoke_tool(
        rank_candidate_templates,
        {"candidates": match_result["candidates"], "top_k": params.top_k},
    )
    if failed := _failed("rank_candidate_templates", rank_result):
        return failed

    candidate_output = rank_result["candidate_result"]
    candidate_output["original_request"] = match_result["original_request"]
    candidate_output["total_templates"] = match_result["total_templates"]
    candidate_output["candidate_count"] = len(candidate_output["candidates"])
    candidate_output["warnings"] = match_result.get("warnings", [])

    save_result = _invoke_tool(
        save_candidate_templates,
        {"candidate_result": candidate_output, "output_path": params.output_path},
    )
    if failed := _failed("save_candidate_templates", save_result):
        return failed

    return {
        "success": True,
        "output_path": save_result["output_path"],
        "candidate_result": save_result["candidate_result"],
    }

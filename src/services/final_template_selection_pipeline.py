"""阶段四最终模板选择流程。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from schemas import PlotRequirement
from src.prompts import FINAL_TEMPLATE_SELECTOR_PROMPT
from src.schemas import (
    CandidateTemplateOutput,
    DatasetContext,
    FinalSelectionPipelineInput,
    FinalSelectionValidationContext,
    FinalTemplateSelection,
    validate_final_selection_references,
)
from src.tools import load_user_requirement, prepare_dataset_context
from src.tools._utils import error_result


def _invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = tool_obj.invoke(payload)
    except Exception as error:  # noqa: BLE001
        return error_result(
            "tool_execution_error",
            f"{tool_obj.name} 工具执行失败。",
            details={"error": str(error)},
            suggestion="请检查输入文件和参数。",
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


def _read_candidate_templates(path: str) -> dict[str, Any]:
    candidate_path = Path(path)
    if not candidate_path.exists():
        return error_result(
            "candidate_file_not_found",
            "候选模板文件不存在。",
            details={"candidate_path": str(candidate_path)},
            suggestion="请先运行阶段三，生成 workspace/candidate_templates.json。",
        )
    try:
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return error_result(
            "json_decode_error",
            "候选模板 JSON 格式错误。",
            details={"candidate_path": str(candidate_path), "error": str(error)},
            suggestion="请确认 candidate_templates.json 未混入日志或非法字符。",
        )
    try:
        candidates = CandidateTemplateOutput.model_validate(data)
    except ValidationError as error:
        return error_result(
            "validation_error",
            "候选模板文件校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查 candidate_templates.json 是否符合 CandidateTemplateOutput。",
        )
    if not candidates.candidates:
        return error_result(
            "empty_candidates",
            "候选模板为空。",
            details={"candidate_path": str(candidate_path)},
            suggestion="请降低阶段三 min_score 或检查用户需求是否过于模糊。",
        )
    return {"success": True, "candidate_output": candidates.model_dump()}


def _structured_model(model: Any):
    if hasattr(model, "with_structured_output"):
        return model.with_structured_output(FinalTemplateSelection)
    return model


def _invoke_model(model: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        chain = FINAL_TEMPLATE_SELECTOR_PROMPT | _structured_model(model)
        result = chain.invoke(payload)
    except Exception as error:  # noqa: BLE001
        return error_result(
            "model_call_failed",
            "模型调用失败。",
            details={"error": str(error)},
            suggestion="请检查模型配置、API Key 或 mock 输出。",
        )
    if isinstance(result, BaseMessage):
        return error_result(
            "structured_output_error",
            "模型未返回结构化输出。",
            details={"content": result.content},
            suggestion="请使用支持结构化输出的模型或传入返回 dict 的 mock。",
        )
    try:
        selection = FinalTemplateSelection.model_validate(result)
    except ValidationError as error:
        return error_result(
            "structured_output_error",
            "模型结构化输出无法通过 Pydantic 校验。",
            details={"errors": error.errors()},
            suggestion="请检查模型输出是否符合 FinalTemplateSelection Schema。",
        )
    return {"success": True, "selection": selection.model_dump()}


def _save_final_selection(selection: FinalTemplateSelection, output_path: str) -> dict[str, Any]:
    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(selection.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        return error_result(
            "save_failed",
            "最终模板选择结果保存失败。",
            details={"output_path": str(path), "error": str(error)},
            suggestion="请检查输出目录权限和文件占用情况。",
        )
    return {"success": True, "output_path": str(path), "selection": selection.model_dump()}


def run_final_template_selection_pipeline(
    model: Any,
    data_path: str,
    requirement_path: str = "workspace/user_requirement.json",
    candidate_path: str = "workspace/candidate_templates.json",
    output_path: str = "workspace/final_template_selection.json",
    dataset_context_path: str = "workspace/dataset_context.json",
    sheet_name: str | int | None = None,
    max_sample_rows: int = 20,
) -> dict[str, Any]:
    """运行阶段四：由模型分析数据并从候选模板中选择最终模板。"""

    try:
        params = FinalSelectionPipelineInput(
            requirement_path=requirement_path,
            candidate_path=candidate_path,
            data_path=data_path,
            output_path=output_path,
            dataset_context_path=dataset_context_path,
            sheet_name=sheet_name,
            max_sample_rows=max_sample_rows,
        )
    except ValidationError as error:
        return {
            "success": False,
            "failed_step": "validate_pipeline_input",
            "error": error_result(
                "validation_error",
                "阶段四入口参数校验失败。",
                details={"errors": error.errors()},
                suggestion="请检查数据路径、候选路径和样例行数。",
            ),
        }

    requirement_result = _invoke_tool(
        load_user_requirement, {"requirement_path": params.requirement_path}
    )
    if requirement_result.get("success") is not True:
        return {
            "success": False,
            "failed_step": "load_user_requirement",
            "error": requirement_result,
        }
    requirement = PlotRequirement.model_validate(requirement_result["requirement"])

    candidate_result = _read_candidate_templates(params.candidate_path)
    if candidate_result.get("success") is not True:
        return {
            "success": False,
            "failed_step": "load_candidate_templates",
            "error": candidate_result,
        }
    candidate_output = CandidateTemplateOutput.model_validate(candidate_result["candidate_output"])

    dataset_result = _invoke_tool(
        prepare_dataset_context,
        {
            "data_path": params.data_path,
            "output_path": params.dataset_context_path,
            "sheet_name": params.sheet_name,
            "max_sample_rows": params.max_sample_rows,
        },
    )
    if dataset_result.get("success") is not True:
        return {"success": False, "failed_step": "prepare_dataset_context", "error": dataset_result}
    dataset_context = DatasetContext.model_validate(dataset_result["dataset_context"])

    model_result = _invoke_model(
        model,
        {
            "original_request": requirement.original_request,
            "structured_requirement": requirement.model_dump(),
            "dataset_context": dataset_context.model_dump(),
            "candidate_templates": [item.model_dump() for item in candidate_output.candidates],
        },
    )
    if model_result.get("success") is not True:
        return {"success": False, "failed_step": "select_final_template", "error": model_result}

    try:
        selection = FinalTemplateSelection.model_validate(model_result["selection"])
        selection = validate_final_selection_references(
            selection,
            FinalSelectionValidationContext(
                candidate_ids_to_names={
                    item.template_id: item.template_name for item in candidate_output.candidates
                },
                column_names=dataset_context.column_names,
            ),
        )
    except ValueError as error:
        message = str(error)
        error_type = (
            "invalid_data_column" if "不存在的数据列" in message else "invalid_candidate_template"
        )
        return {
            "success": False,
            "failed_step": "validate_model_selection",
            "error": error_result(
                error_type,
                "模型输出引用了非法模板或数据列。",
                details={"error": message},
                suggestion="请要求模型只能使用候选模板 ID 和真实数据列名。",
            ),
        }
    except ValidationError as error:
        return {
            "success": False,
            "failed_step": "validate_model_selection",
            "error": error_result(
                "structured_output_error",
                "模型输出二次校验失败。",
                details={"errors": error.errors()},
                suggestion="请检查 FinalTemplateSelection 输出。",
            ),
        }

    save_result = _save_final_selection(selection, params.output_path)
    if save_result.get("success") is not True:
        return {"success": False, "failed_step": "save_final_selection", "error": save_result}
    return {"success": True, **save_result}

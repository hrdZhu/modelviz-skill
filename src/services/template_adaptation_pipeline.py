"""阶段五：依赖检查、模板适配和绘图执行流程。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from schemas import PlotRequirement
from src.prompts import TEMPLATE_ADAPTATION_PLAN_PROMPT, TEMPLATE_CODE_ADAPTER_PROMPT
from src.schemas import (
    AdaptationPlan,
    AdaptationResult,
    DatasetContext,
    DependencyInstallResult,
    DependencyReport,
    FinalTemplateSelection,
    TemplateAdaptationPipelineInput,
)
from src.tools import (
    check_python_dependencies,
    execute_plot_script,
    inspect_template_dependencies,
    install_python_dependencies,
    load_selected_template,
    save_adapted_script,
)
from src.tools._utils import error_result, unique_non_empty


def _invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = tool_obj.invoke(payload)
    except Exception as error:  # noqa: BLE001
        return error_result(
            "tool_execution_error", f"{tool_obj.name} 执行失败。", details={"error": str(error)}
        )
    return (
        result
        if isinstance(result, dict)
        else error_result("invalid_tool_result", f"{tool_obj.name} 返回非 dict。")
    )


def _read_json(path: str, label: str) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return error_result("file_not_found", f"{label} 不存在。", details={"path": str(file)})
    try:
        return {"success": True, "data": json.loads(file.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as error:
        return error_result(
            "json_decode_error", f"{label} JSON 格式错误。", details={"error": str(error)}
        )


def _structured_model(model: Any, schema: type):
    if hasattr(model, "with_structured_output"):
        return model.with_structured_output(schema)
    return model


def _invoke_structured_model(
    prompt, model: Any, schema: type, payload: dict[str, Any]
) -> dict[str, Any]:
    try:
        result = (prompt | _structured_model(model, schema)).invoke(payload)
        parsed = schema.model_validate(result)
    except Exception as error:  # noqa: BLE001
        return error_result(
            "model_structured_output_error",
            "模型结构化输出失败。",
            details={"error": str(error)},
            suggestion="请检查模型输出是否符合 Pydantic Schema。",
        )
    return {"success": True, "result": parsed.model_dump()}


def _write_json(path: str, data: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _validate_columns(columns: list[str], allowed_columns: list[str]) -> list[str]:
    return [column for column in columns if column not in set(allowed_columns)]


def _available_dependency_names(report: DependencyReport) -> set[str]:
    names: set[str] = set()
    for item in report.installed_dependencies:
        if item.available:
            names.add(item.import_name)
            if item.package_name:
                names.add(item.package_name)
    return names


def _dependency_names_from_inspection(inspection: dict[str, Any]) -> list[str]:
    return unique_non_empty(
        [
            item["import_name"]
            for item in inspection["required_dependencies"]
            if item.get("import_name")
        ]
    )


def run_template_adaptation_pipeline(
    plan_model: Any,
    code_model: Any,
    data_path: str,
    final_selection_path: str = "workspace/final_template_selection.json",
    requirement_path: str = "workspace/user_requirement.json",
    dataset_context_path: str = "workspace/dataset_context.json",
    catalog_path: str = "docs/template_catalog.yaml",
    workspace_dir: str = "workspace",
    outputs_dir: str = "outputs",
    python_executable: str = "",
    auto_install: bool = True,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """运行阶段五，任一步失败即停止。"""

    try:
        params = TemplateAdaptationPipelineInput(
            final_selection_path=final_selection_path,
            requirement_path=requirement_path,
            dataset_context_path=dataset_context_path,
            catalog_path=catalog_path,
            data_path=data_path,
            workspace_dir=workspace_dir,
            outputs_dir=outputs_dir,
            python_executable=python_executable,
            auto_install=auto_install,
            timeout_seconds=timeout_seconds,
        )
    except ValidationError as error:
        return {
            "success": False,
            "failed_step": "validate_input",
            "error": error_result(
                "validation_error", "阶段五入口参数错误。", details={"errors": error.errors()}
            ),
        }

    requirement_raw = _read_json(params.requirement_path, "用户需求文件")
    if requirement_raw.get("success") is not True:
        return {"success": False, "failed_step": "load_user_requirement", "error": requirement_raw}
    dataset_raw = _read_json(params.dataset_context_path, "数据上下文文件")
    if dataset_raw.get("success") is not True:
        return {"success": False, "failed_step": "load_dataset_context", "error": dataset_raw}
    selection_raw = _read_json(params.final_selection_path, "最终模板选择文件")
    if selection_raw.get("success") is not True:
        return {"success": False, "failed_step": "load_final_selection", "error": selection_raw}

    try:
        requirement = PlotRequirement.model_validate(requirement_raw["data"])
        dataset_context = DatasetContext.model_validate(dataset_raw["data"])
        final_selection = FinalTemplateSelection.model_validate(selection_raw["data"])
    except ValidationError as error:
        return {
            "success": False,
            "failed_step": "validate_inputs",
            "error": error_result(
                "validation_error", "阶段五输入文件校验失败。", details={"errors": error.errors()}
            ),
        }

    selected_result = _invoke_tool(
        load_selected_template,
        {"final_selection_path": params.final_selection_path, "catalog_path": params.catalog_path},
    )
    if selected_result.get("success") is not True:
        return {"success": False, "failed_step": "load_selected_template", "error": selected_result}
    selected = selected_result["selected_template"]
    metadata = selected["metadata"]

    inspection_result = _invoke_tool(
        inspect_template_dependencies,
        {
            "template_code_path": selected["template_code_path"],
            "catalog_dependencies": metadata.get("dependencies", []),
            "project_root": ".",
        },
    )
    if inspection_result.get("success") is not True:
        return {
            "success": False,
            "failed_step": "inspect_template_dependencies",
            "error": inspection_result,
        }
    inspection = inspection_result["dependency_inspection"]
    if inspection["unresolved_dependencies"]:
        return {
            "success": False,
            "failed_step": "inspect_template_dependencies",
            "error": error_result(
                "package_mapping_failed",
                "存在无法映射的模板依赖。",
                details={"unresolved_dependencies": inspection["unresolved_dependencies"]},
                suggestion="请在 docs/package_name_mapping.yaml 中补充明确映射。",
            ),
        }

    dependency_imports = _dependency_names_from_inspection(inspection)
    check_result = _invoke_tool(
        check_python_dependencies,
        {
            "dependencies": dependency_imports,
            "python_executable": params.python_executable or sys.executable,
            "output_path": f"{params.workspace_dir}/dependency_report.json",
        },
    )
    if check_result.get("success") is not True:
        return {"success": False, "failed_step": "check_python_dependencies", "error": check_result}
    dependency_report = DependencyReport.model_validate(check_result["dependency_report"])

    missing_packages = unique_non_empty(
        [item.package_name for item in dependency_report.missing_dependencies if item.package_name]
    )
    install_result_data = DependencyInstallResult(
        success=True, requested_packages=[], installed_packages=[], skipped_packages=[], warnings=[]
    )
    if missing_packages:
        install_result = _invoke_tool(
            install_python_dependencies,
            {
                "packages": missing_packages,
                "python_executable": params.python_executable or sys.executable,
                "auto_install": params.auto_install,
                "timeout_seconds": params.timeout_seconds,
                "output_path": f"{params.workspace_dir}/dependency_install_result.json",
                "requirements_output_path": f"{params.workspace_dir}/requirements.generated.txt",
            },
        )
        if install_result.get("success") is not True:
            return {
                "success": False,
                "failed_step": "install_python_dependencies",
                "error": install_result,
            }
        install_result_data = DependencyInstallResult.model_validate(
            install_result["install_result"]
        )
        recheck = _invoke_tool(
            check_python_dependencies,
            {
                "dependencies": dependency_imports,
                "python_executable": params.python_executable or sys.executable,
                "output_path": f"{params.workspace_dir}/dependency_report.json",
            },
        )
        dependency_report = DependencyReport.model_validate(recheck["dependency_report"])
        if dependency_report.missing_dependencies:
            return {
                "success": False,
                "failed_step": "recheck_dependencies",
                "error": error_result(
                    "dependency_still_missing",
                    "安装后仍有依赖无法导入。",
                    details={
                        "missing": [i.model_dump() for i in dependency_report.missing_dependencies]
                    },
                ),
            }

    available_dependencies = _available_dependency_names(dependency_report)
    plan_result = _invoke_structured_model(
        TEMPLATE_ADAPTATION_PLAN_PROMPT,
        plan_model,
        AdaptationPlan,
        {
            "original_request": requirement.original_request,
            "structured_requirement": requirement.model_dump(),
            "dataset_context": dataset_context.model_dump(),
            "selected_template": final_selection.model_dump(),
            "template_metadata": metadata,
            "template_source_code": selected["template_source_code"],
            "dependency_report": dependency_report.model_dump(),
            "dependency_install_result": install_result_data.model_dump(),
        },
    )
    if plan_result.get("success") is not True:
        return {"success": False, "failed_step": "generate_adaptation_plan", "error": plan_result}
    plan = AdaptationPlan.model_validate(plan_result["result"])
    bad_columns = _validate_columns(
        plan.selected_columns + [m.data_column for m in plan.column_mappings],
        dataset_context.column_names,
    )
    bad_deps = [dep for dep in plan.required_dependencies if dep not in available_dependencies]
    if bad_columns:
        return {
            "success": False,
            "failed_step": "validate_adaptation_plan",
            "error": error_result(
                "user_data_column_missing",
                "适配计划引用了不存在的数据列。",
                details={"columns": bad_columns},
            ),
        }
    if bad_deps:
        return {
            "success": False,
            "failed_step": "validate_adaptation_plan",
            "error": error_result(
                "unchecked_dependency",
                "适配计划引用了未确认依赖。",
                details={"dependencies": bad_deps},
            ),
        }
    if not plan.can_proceed:
        return {
            "success": False,
            "failed_step": "generate_adaptation_plan",
            "error": error_result(
                "adaptation_cannot_proceed", "适配计划判断当前信息不足。", details=plan.model_dump()
            ),
        }
    _write_json(f"{params.workspace_dir}/adaptation_plan.json", plan.model_dump())

    code_result = _invoke_structured_model(
        TEMPLATE_CODE_ADAPTER_PROMPT,
        code_model,
        AdaptationResult,
        {
            "data_path": params.data_path,
            "adaptation_plan": plan.model_dump(),
            "template_source_code": selected["template_source_code"],
            "dependency_report": dependency_report.model_dump(),
        },
    )
    if code_result.get("success") is not True:
        return {"success": False, "failed_step": "generate_adapted_code", "error": code_result}
    adaptation = AdaptationResult.model_validate(code_result["result"])
    bad_columns = _validate_columns(adaptation.data_columns_used, dataset_context.column_names)
    bad_deps = [dep for dep in adaptation.dependencies_used if dep not in available_dependencies]
    if bad_columns:
        return {
            "success": False,
            "failed_step": "validate_adapted_code",
            "error": error_result(
                "user_data_column_missing",
                "适配代码引用了不存在的数据列。",
                details={"columns": bad_columns},
            ),
        }
    if bad_deps:
        return {
            "success": False,
            "failed_step": "validate_adapted_code",
            "error": error_result(
                "unchecked_dependency",
                "适配代码引用了未确认依赖。",
                details={"dependencies": bad_deps},
            ),
        }

    if adaptation.additional_dependencies_requested:
        install_extra = _invoke_tool(
            install_python_dependencies,
            {
                "packages": adaptation.additional_dependencies_requested,
                "python_executable": params.python_executable or sys.executable,
                "auto_install": params.auto_install,
                "timeout_seconds": params.timeout_seconds,
            },
        )
        if install_extra.get("success") is not True:
            return {
                "success": False,
                "failed_step": "install_additional_dependencies",
                "error": install_extra,
            }

    _write_json(f"{params.workspace_dir}/adaptation_result.json", adaptation.model_dump())
    save_result = _invoke_tool(
        save_adapted_script,
        {
            "adapted_code": adaptation.adapted_code,
            "script_path": f"{params.workspace_dir}/adapted_plot.py",
        },
    )
    if save_result.get("success") is not True:
        return {"success": False, "failed_step": "save_adapted_script", "error": save_result}

    execution = _invoke_tool(
        execute_plot_script,
        {
            "script_path": save_result["script_path"],
            "data_path": params.data_path,
            "output_directory": params.outputs_dir,
            "timeout_seconds": params.timeout_seconds,
            "python_executable": params.python_executable or sys.executable,
        },
    )
    if execution.get("success") is not True:
        return {"success": False, "failed_step": "execute_plot_script", "error": execution}

    return {
        "success": True,
        "adaptation_plan": plan.model_dump(),
        "adaptation_result": adaptation.model_dump(),
        "execution_result": execution["execution_result"],
    }

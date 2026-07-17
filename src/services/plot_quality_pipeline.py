"""阶段六：执行、技术检查、视觉检查和有限自动修复。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.prompts import PLOT_CODE_REPAIR_PROMPT, VISUAL_QUALITY_CHECKER_PROMPT
from src.schemas import (
    FinalQualityReport,
    RepairHistoryItem,
    RepairResult,
    TechnicalIssue,
    TechnicalQualityReport,
    VisualQualityReport,
)
from src.tools import (
    collect_plot_warnings,
    execute_plot_script,
    inspect_generated_image,
    install_python_dependencies,
    validate_output_artifacts,
)
from src.tools._utils import error_result


def _read_json(path: str) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: str, data: dict[str, Any]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _structured_model(model: Any, schema: type):
    if hasattr(model, "with_structured_output"):
        return model.with_structured_output(schema)
    return model


def _invoke_model(prompt, model: Any, schema: type, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = (prompt | _structured_model(model, schema)).invoke(payload)
        parsed = schema.model_validate(result)
    except Exception as error:  # noqa: BLE001
        return error_result(
            "model_structured_output_error", "模型结构化输出失败。", details={"error": str(error)}
        )
    return {"success": True, "result": parsed.model_dump()}


def _technical_report(execution: dict[str, Any], output_directory: str) -> TechnicalQualityReport:
    exec_result = execution.get("execution_result", {})
    artifacts = _invoke_tool(
        validate_output_artifacts,
        {
            "generated_files": exec_result.get("generated_files", []),
            "output_directory": output_directory,
        },
    )
    log_report = _invoke_tool(
        collect_plot_warnings,
        {
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", ""),
            "return_code": exec_result.get("return_code"),
        },
    )
    issues = [TechnicalIssue.model_validate(item) for item in artifacts.get("issues", [])]
    issues.extend(TechnicalIssue.model_validate(item) for item in log_report.get("issues", []))

    image_report = {}
    pngs = [file for file in artifacts.get("valid_files", []) if str(file).lower().endswith(".png")]
    if pngs:
        image_report = _invoke_tool(inspect_generated_image, {"image_path": pngs[0]})
        if image_report.get("success") is not True:
            issues.extend(
                TechnicalIssue.model_validate(item) for item in image_report.get("issues", [])
            )

    passed = (
        execution.get("success") is True
        and artifacts.get("success") is True
        and not any(issue.severity == "high" for issue in issues)
    )
    return TechnicalQualityReport(
        passed=passed,
        script_executed=execution.get("success") is True,
        dependencies_available=not any(
            issue.issue_type == "missing_dependency" for issue in issues
        ),
        dependency_issues=[
            issue.message for issue in issues if issue.issue_type == "missing_dependency"
        ],
        image_exists=bool(pngs),
        image_readable=bool(image_report.get("image_readable")),
        image_width=image_report.get("image_width", 0),
        image_height=image_report.get("image_height", 0),
        file_size=image_report.get("file_size", 0),
        pixel_variance=image_report.get("pixel_variance", 0.0),
        possible_blank_image=image_report.get("possible_blank_image", False),
        generated_formats=artifacts.get("generated_formats", []),
        execution_errors=[exec_result.get("stderr", "")] if exec_result.get("stderr") else [],
        warnings=log_report.get("warnings", []),
        issues=issues,
        needs_repair=not passed,
    )


def _save_repair_version(script_path: str, attempt: int) -> str:
    source = Path(script_path)
    target = Path("workspace/repair_versions") / f"adapted_plot_v{attempt}.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return str(target)


def run_plot_quality_pipeline(
    visual_model: Any,
    repair_model: Any,
    script_path: str = "workspace/adapted_plot.py",
    data_path: str = "",
    output_directory: str = "outputs",
    max_repair_attempts: int = 3,
    max_dependency_install_rounds: int = 2,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """执行质量检查和有限自动修复，禁止无限循环。"""

    repair_history: list[RepairHistoryItem] = []
    dependency_rounds = 0
    previous_issues: list[str] | None = None
    current_script = Path(script_path)
    if not current_script.exists():
        return {
            "success": False,
            "error": error_result("script_not_found", "adapted_plot.py 不存在。"),
        }

    for attempt in range(max_repair_attempts + 1):
        execution = _invoke_tool(
            execute_plot_script,
            {
                "script_path": str(current_script),
                "data_path": data_path,
                "output_directory": output_directory,
                "timeout_seconds": timeout_seconds,
            },
        )
        technical = _technical_report(execution, output_directory)
        _write_json("workspace/technical_quality_report.json", technical.model_dump())

        generated_image = ""
        generated_files = execution.get("execution_result", {}).get("generated_files", [])
        for file in generated_files:
            if str(file).lower().endswith(".png"):
                generated_image = file
                break

        visual_payload = {
            "user_requirement": _read_json("workspace/user_requirement.json"),
            "final_template_selection": _read_json("workspace/final_template_selection.json"),
            "adaptation_plan": _read_json("workspace/adaptation_plan.json"),
            "technical_quality_report": technical.model_dump(),
            "template_preview": "",
            "generated_image": generated_image,
        }
        visual_result = _invoke_model(
            VISUAL_QUALITY_CHECKER_PROMPT, visual_model, VisualQualityReport, visual_payload
        )
        if visual_result.get("success") is not True:
            return {"success": False, "failed_step": "visual_quality_check", "error": visual_result}
        visual = VisualQualityReport.model_validate(visual_result["result"])
        _write_json("workspace/visual_quality_report.json", visual.model_dump())

        passed = (
            technical.passed
            and visual.passed
            and not technical.needs_repair
            and not visual.needs_repair
        )
        issue_fingerprint = [issue.message for issue in technical.issues] + visual.issues
        if passed:
            final = FinalQualityReport(
                passed=True,
                repair_attempts=attempt,
                dependency_install_rounds=dependency_rounds,
                final_script=str(current_script),
                final_outputs=generated_files,
                technical_summary="技术检查通过。",
                visual_summary="视觉检查通过。",
            )
            _write_json("workspace/final_quality_report.json", final.model_dump())
            _write_json(
                "workspace/repair_history.json",
                {"items": [item.model_dump() for item in repair_history]},
            )
            return {"success": True, "final_quality_report": final.model_dump()}

        if attempt >= max_repair_attempts:
            final = FinalQualityReport(
                passed=False,
                repair_attempts=attempt,
                dependency_install_rounds=dependency_rounds,
                final_script=str(current_script),
                final_outputs=generated_files,
                technical_summary="技术检查仍有问题。",
                visual_summary="视觉检查仍有问题。",
                remaining_issues=issue_fingerprint,
                warnings=["达到最大修复次数。"],
            )
            _write_json("workspace/final_quality_report.json", final.model_dump())
            return {
                "success": False,
                "error": error_result("max_repair_attempts_reached", "达到最大修复次数。"),
                "final_quality_report": final.model_dump(),
            }

        if previous_issues == issue_fingerprint:
            return {
                "success": False,
                "error": error_result("repeated_issues", "连续两次出现相同问题，停止修复。"),
            }
        previous_issues = issue_fingerprint

        current_code = current_script.read_text(encoding="utf-8")
        repair_payload = {
            "current_code": current_code,
            "user_requirement": _read_json("workspace/user_requirement.json"),
            "adaptation_plan": _read_json("workspace/adaptation_plan.json"),
            "dependency_report": _read_json("workspace/dependency_report.json"),
            "technical_quality_report": technical.model_dump(),
            "visual_quality_report": visual.model_dump(),
            "template_source_code": "",
        }
        repair_result = _invoke_model(
            PLOT_CODE_REPAIR_PROMPT, repair_model, RepairResult, repair_payload
        )
        if repair_result.get("success") is not True:
            return {"success": False, "failed_step": "repair_code", "error": repair_result}
        repair = RepairResult.model_validate(repair_result["result"])
        if not repair.can_retry:
            return {
                "success": False,
                "error": error_result("repair_model_stopped", "修复模型认为不能继续重试。"),
            }
        if repair.repaired_code == current_code:
            return {
                "success": False,
                "error": error_result("repair_no_change", "修复代码没有变化。"),
            }
        if repair.additional_dependencies_requested:
            if dependency_rounds >= max_dependency_install_rounds:
                return {
                    "success": False,
                    "error": error_result(
                        "dependency_install_round_limit", "达到依赖安装次数上限。"
                    ),
                }
            install_result = _invoke_tool(
                install_python_dependencies,
                {"packages": repair.additional_dependencies_requested, "auto_install": True},
            )
            dependency_rounds += 1
            if install_result.get("success") is not True:
                return {
                    "success": False,
                    "failed_step": "install_repair_dependencies",
                    "error": install_result,
                }

        backup = _save_repair_version(str(current_script), attempt + 1)
        current_script.write_text(repair.repaired_code, encoding="utf-8")
        repair_history.append(
            RepairHistoryItem(
                attempt=attempt + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                dependency_changes=repair.additional_dependencies_requested,
                technical_issues=[issue.message for issue in technical.issues],
                visual_issues=visual.issues,
                changes_summary=repair.changes_summary,
                script_path=backup,
                execution_success=execution.get("success") is True,
                technical_passed=technical.passed,
                visual_passed=visual.passed,
            )
        )
        _write_json(
            "workspace/repair_history.json",
            {"items": [item.model_dump() for item in repair_history]},
        )

    return {
        "success": False,
        "error": error_result("max_repair_attempts_reached", "达到最大修复次数。"),
    }

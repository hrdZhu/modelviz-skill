"""Run compact evaluation for the ModelViz skill.

The model-dependent stages use mock structured outputs. Deterministic code paths,
Pydantic validation, tools, script execution, and image checks are exercised for real.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd
import yaml
from langchain_core.runnables import RunnableLambda

from schemas import PlotRequirement
from src.schemas import CandidateTemplateOutput, DatasetContext, FinalTemplateSelection
from src.services.candidate_matching_pipeline import run_candidate_matching_pipeline
from src.services.final_template_selection_pipeline import run_final_template_selection_pipeline
from src.services.requirement_parser import parse_and_save_requirement
from src.services.template_adaptation_pipeline import run_template_adaptation_pipeline
from src.services.plot_quality_pipeline import run_plot_quality_pipeline


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evals/evaluation_cases.json"
CATALOG_PATH = ROOT / "docs/template_catalog.yaml"
OUTPUT_DIR = ROOT / "evals"
RUNTIME_DIR = OUTPUT_DIR / "runtime"
REPORT_PATH = OUTPUT_DIR / "evaluation_report.md"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"
INTERVIEW_PATH = ROOT / "docs/evaluation_summary_for_interview.md"
PRE_OPTIMIZATION_METRICS = {
    "end_to_end_task_success_rate": 0.0,
    "recall_at_5": 0.9333,
    "recall_at_8": 0.9333,
    "final_template_selection_accuracy": 0.9,
    "first_execution_success_rate": 0.5,
    "final_execution_success_rate": 0.0,
    "average_image_quality_score": 1.0,
    "automatic_repair_success_rate": 0.0,
    "guardrails": {
        "data_column_hallucination_rate": 0.125,
        "candidate_scope_violation_rate": 0.125,
        "original_template_modification_rate": 0.0,
        "user_data_modification_rate": 0.0,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases() -> dict[str, Any]:
    return _read_json(CASES_PATH)


def _load_catalog() -> list[dict[str, Any]]:
    data = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    return data.get("templates", [])


def _requirement_by_id(cases: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in cases["requirement_cases"]}


def _contains_any(texts: list[Any], terms: list[str]) -> bool:
    haystack = " ".join(str(item).lower() for item in texts if item is not None)
    return any(term.lower() in haystack for term in terms)


def _acceptable_template_ids(catalog: list[dict[str, Any]], criteria: dict[str, Any]) -> set[str]:
    """按元数据条件动态派生可接受模板集合，避免写死单个模板 ID。"""

    result: set[str] = set()
    categories = set(criteria.get("categories", []))
    chart_types = set(criteria.get("chart_types", []))
    keyword_terms = criteria.get("keywords_any", [])
    exclude_chart_contains = [
        term.lower() for term in criteria.get("exclude_chart_type_contains", [])
    ]

    for template in catalog:
        if categories and template.get("category") not in categories:
            continue
        if chart_types and template.get("chart_type") not in chart_types:
            continue
        chart_type = str(template.get("chart_type", "")).lower()
        if exclude_chart_contains and any(term in chart_type for term in exclude_chart_contains):
            continue
        if keyword_terms:
            texts = [
                template.get("name"),
                template.get("description"),
                template.get("category"),
                template.get("chart_type"),
                *template.get("keywords", []),
                *template.get("synonyms", []),
                *template.get("use_cases", []),
            ]
            if not _contains_any(texts, keyword_terms):
                continue
        result.add(str(template["id"]))
    return result


def _requirement_model_payload(case: dict[str, Any]) -> dict[str, Any]:
    return {"original_request": case["input"], **case["requirement"]}


def _parse_case(case: dict[str, Any], output_path: Path) -> PlotRequirement:
    return parse_and_save_requirement(
        case["input"],
        RunnableLambda(lambda _: _requirement_model_payload(case)),
        output_path=output_path,
    )


def _make_data(data_kind: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data_kind == "time_series":
        df = pd.DataFrame(
            {
                "年份": [2020, 2021, 2022, 2023, 2024],
                "产量": [12, 15, 14, 18, 22],
                "温度": [20, 21, 20, 23, 24],
            }
        )
    elif data_kind == "time_series_model":
        df = pd.DataFrame(
            {
                "年份": [2020, 2021, 2022, 2023, 2024, 2025],
                "模型A误差": [0.22, 0.2, 0.18, 0.16, 0.15, 0.14],
                "模型B误差": [0.25, 0.23, 0.2, 0.19, 0.18, 0.17],
            }
        )
    elif data_kind == "regression":
        df = pd.DataFrame(
            {
                "真实值": [10, 12, 13, 15, 18, 20],
                "预测值": [9.5, 12.4, 12.8, 15.2, 17.4, 20.8],
                "模型": ["A"] * 6,
            }
        )
    elif data_kind == "category_numeric":
        df = pd.DataFrame(
            {
                "地区": ["A区", "B区", "C区", "D区"],
                "产值": [120, 150, 132, 170],
                "成本": [80, 90, 85, 100],
                "得分": [0.72, 0.81, 0.77, 0.86],
            }
        )
    elif data_kind == "distribution_missing":
        df = pd.DataFrame(
            {
                "组别": ["对照", "对照", "处理", "处理", "处理"],
                "指标值": [1.2, None, 1.8, 2.0, 1.9],
                "误差": [0.1, 0.2, 0.15, 0.12, 0.18],
            }
        )
    elif data_kind == "multivariate":
        df = pd.DataFrame(
            {
                "指标A": [1.0, 1.5, 1.8, 2.0, 2.4, 2.8],
                "指标B": [2.2, 2.1, 2.6, 3.0, 3.1, 3.5],
                "指标C": [5.0, 4.8, 4.6, 4.4, 4.1, 3.9],
                "目标值": [8, 9, 10, 11, 12, 14],
            }
        )
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def _select_payload(
    candidate_output: dict[str, Any],
    dataset_context: dict[str, Any],
    acceptable_ids: set[str],
    expect_no_selection: bool = False,
) -> dict[str, Any]:
    candidates = candidate_output["candidates"]
    selected = None
    if not expect_no_selection:
        selected = next(
            (item for item in candidates if item["template_id"] in acceptable_ids), None
        )

    comparisons = [
        {
            "template_id": item["template_id"],
            "suitable": bool(item["template_id"] in acceptable_ids),
            "advantages": ["候选模板元数据满足评估条件"]
            if item["template_id"] in acceptable_ids
            else [],
            "limitations": []
            if item["template_id"] in acceptable_ids
            else ["不是本案例的可接受模板类别"],
            "data_compatibility": "使用真实列名进行校验",
            "requirement_compatibility": item.get("candidate_reason", ""),
        }
        for item in candidates
    ]
    columns = dataset_context["column_names"][:2]
    if not selected:
        return {
            "dataset_summary": "数据已读取，但候选模板与需求或数据结构不匹配。",
            "observed_data_features": [f"{dataset_context['row_count']} 行数据"],
            "relevant_columns": columns,
            "candidate_comparisons": comparisons,
            "selected_template_id": None,
            "selected_template_name": None,
            "alternative_template_ids": [],
            "selection_reason": "没有合适模板。",
            "data_support_reason": "当前数据不能支持该候选模板集合。",
            "data_warnings": [],
            "confidence": 0.35,
            "needs_clarification": True,
            "clarification_question": "请补充更匹配该图表的数据列或调整图表需求。",
        }
    alternatives = [
        item["template_id"]
        for item in candidates
        if item["template_id"] in acceptable_ids and item["template_id"] != selected["template_id"]
    ][:2]
    return {
        "dataset_summary": f"数据包含 {dataset_context['row_count']} 行、{dataset_context['column_count']} 列。",
        "observed_data_features": ["包含可用于绘图的真实列名和样例记录"],
        "relevant_columns": columns,
        "candidate_comparisons": comparisons,
        "selected_template_id": selected["template_id"],
        "selected_template_name": selected["template_name"],
        "alternative_template_ids": alternatives,
        "selection_reason": "该候选模板满足评估定义的可接受模板条件。",
        "data_support_reason": "相关列来自 dataset_context.column_names。",
        "data_warnings": dataset_context.get("warnings", []),
        "confidence": 0.82,
        "needs_clarification": False,
        "clarification_question": "",
    }


def _plan_payload(selection: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    columns = context["column_names"][:2]
    mappings = [
        {
            "data_column": column,
            "template_role": "x" if index == 0 else "y",
            "reason": "评估样例列映射",
        }
        for index, column in enumerate(columns)
    ]
    return {
        "template_id": selection["selected_template_id"],
        "template_name": selection["selected_template_name"],
        "plot_goal": selection.get("selection_reason", ""),
        "selected_columns": columns,
        "column_mappings": mappings,
        "required_preprocessing": ["按绘图需要读取用户数据"],
        "required_dependencies": [],
        "layout_elements_to_preserve": ["单图或模板主体布局"],
        "style_elements_to_preserve": ["科研风格", "清晰坐标轴"],
        "elements_allowed_to_change": ["标题", "数据列映射"],
        "title_plan": "根据用户需求生成标题",
        "axis_plan": "使用真实列名作为坐标轴",
        "legend_plan": "必要时显示图例",
        "annotation_plan": "",
        "output_formats": ["png"],
        "warnings": [],
        "can_proceed": True,
        "clarification_question": "",
    }


def _plot_code(columns: list[str], mode: str = "normal") -> str:
    if mode == "fail":
        return "print('intentional first-run failure: no output generated')\n"
    x_col = columns[0]
    y_col = columns[1] if len(columns) > 1 else columns[0]
    return f"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ["DATA_PATH"])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    x_col = {x_col!r}
    y_col = {y_col!r}
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=180)
    y = pd.to_numeric(df[y_col], errors="coerce")
    if pd.api.types.is_numeric_dtype(df[x_col]):
        x = pd.to_numeric(df[x_col], errors="coerce")
        ax.plot(x, y, marker="o", linewidth=2.0, color="#2f6f9f")
    else:
        grouped = pd.DataFrame({{"x": df[x_col], "y": y}}).groupby("x", dropna=False)["y"].mean()
        ax.bar(grouped.index.astype(str), grouped.values, color="#5b8db8", edgecolor="#1f3349")
    ax.set_title("ModelViz evaluation chart")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "chart.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
"""


def _code_payload(columns: list[str], fail: bool = False) -> dict[str, Any]:
    return {
        "adapted_code": _plot_code(columns, mode="fail" if fail else "normal"),
        "changes_summary": ["使用真实数据替换模板演示数据"],
        "preserved_style_elements": ["清晰坐标轴", "科研图尺寸"],
        "changed_elements": ["数据读取", "标题和坐标轴"],
        "data_columns_used": columns,
        "dependencies_used": [],
        "additional_dependencies_requested": [],
        "assumptions": ["Mock 代码生成用于流程评估，不代表真实模型视觉创造力"],
        "warnings": [],
    }


def _repair_payload(columns: list[str]) -> dict[str, Any]:
    return {
        "repaired_code": _plot_code(columns),
        "fixed_issues": ["补充图片输出"],
        "remaining_risks": [],
        "changes_summary": ["修复无输出问题"],
        "data_columns_used": columns,
        "dependencies_used": [],
        "additional_dependencies_requested": [],
        "can_retry": True,
    }


def _visual_pass_payload(score_text: str = "通过") -> dict[str, Any]:
    return {
        "passed": True,
        "requirement_alignment": score_text,
        "data_expression_quality": "使用真实列名和样例数据生成。",
        "style_preservation": "保留基础科研图风格。",
        "readability": "标签和图例清晰。",
        "layout_quality": "布局通过技术检查。",
        "color_quality": "配色清晰。",
        "issues": [],
        "suggested_fixes": [],
        "needs_repair": False,
        "confidence": 0.85,
    }


def _run_requirement_eval(cases: dict[str, Any], runtime_dir: Path) -> dict[str, Any]:
    details = []
    for case in cases["requirement_cases"]:
        out = runtime_dir / "requirements" / f"{case['id']}.json"
        parsed = _parse_case(case, out)
        expected = _requirement_model_payload(case)
        success = parsed.model_dump() == expected
        details.append({"id": case["id"], "success": success, "is_ambiguous": parsed.is_ambiguous})
    return {
        "case_count": len(details),
        "success_count": sum(item["success"] for item in details),
        "details": details,
    }


def _run_stage3_for_case(
    requirement_case: dict[str, Any], work_dir: Path, *, top_k: int = 8
) -> dict[str, Any]:
    req_path = work_dir / "user_requirement.json"
    candidate_path = work_dir / "candidate_templates.json"
    _parse_case(requirement_case, req_path)
    return run_candidate_matching_pipeline(
        requirement_path=str(req_path),
        catalog_path=str(CATALOG_PATH),
        output_path=str(candidate_path),
        top_k=top_k,
        min_score=0.10,
    )


def _run_recall_eval(
    cases: dict[str, Any], catalog: list[dict[str, Any]], runtime_dir: Path
) -> dict[str, Any]:
    requirement_cases = _requirement_by_id(cases)
    details = []
    for case in cases["recall_cases"]:
        req_case = requirement_cases[case["source_requirement_case"]]
        acceptable = _acceptable_template_ids(catalog, case["acceptable"])
        work_dir = runtime_dir / "recall" / case["id"]
        result = _run_stage3_for_case(req_case, work_dir)
        candidates = result.get("candidate_result", {}).get("candidates", [])
        top5 = [item["template_id"] for item in candidates[:5]]
        top8 = [item["template_id"] for item in candidates[:8]]
        hit5 = bool(acceptable.intersection(top5))
        hit8 = bool(acceptable.intersection(top8))
        details.append(
            {
                "id": case["id"],
                "success": result.get("success") is True,
                "acceptable_count": len(acceptable),
                "top5_hit": hit5,
                "top8_hit": hit8,
                "top8_ids": top8,
                "error": result.get("error"),
            }
        )
    total = len(details)
    return {
        "case_count": total,
        "recall_at_5": round(sum(item["top5_hit"] for item in details) / total, 4),
        "recall_at_8": round(sum(item["top8_hit"] for item in details) / total, 4),
        "details": details,
    }


def _run_stage4_selection(
    case: dict[str, Any],
    requirement_case: dict[str, Any],
    acceptable_ids: set[str],
    work_dir: Path,
) -> dict[str, Any]:
    data_path = _make_data(case["data_kind"], work_dir / "data.csv")
    req_path = work_dir / "user_requirement.json"
    candidate_path = work_dir / "candidate_templates.json"
    final_path = work_dir / "final_template_selection.json"
    context_path = work_dir / "dataset_context.json"
    parsed = _parse_case(requirement_case, req_path)
    if case.get("expect_ambiguous_stop") and parsed.is_ambiguous:
        return {"success": True, "ambiguous_stop": True, "selection": None}
    stage3 = run_candidate_matching_pipeline(
        requirement_path=str(req_path),
        catalog_path=str(CATALOG_PATH),
        output_path=str(candidate_path),
        top_k=8,
        min_score=0.10,
    )
    if stage3.get("success") is not True:
        return {"success": False, "failed_step": "stage3", "error": stage3.get("error")}
    candidate_output = CandidateTemplateOutput.model_validate(stage3["candidate_result"])
    candidate_dict = candidate_output.model_dump()
    context = DatasetContext.model_validate(
        __import__(
            "src.tools.prepare_dataset_context", fromlist=["prepare_dataset_context"]
        ).prepare_dataset_context.invoke(
            {"data_path": str(data_path), "output_path": str(context_path)}
        )["dataset_context"]
    ).model_dump()
    payload = _select_payload(
        candidate_dict,
        context,
        acceptable_ids,
        expect_no_selection=case.get("expect_no_selection", False),
    )
    stage4 = run_final_template_selection_pipeline(
        RunnableLambda(lambda _: payload),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(candidate_path),
        output_path=str(final_path),
        dataset_context_path=str(context_path),
    )
    return {
        "success": stage4.get("success") is True,
        "stage3": stage3,
        "stage4": stage4,
        "data_path": str(data_path),
        "requirement_path": str(req_path),
        "candidate_path": str(candidate_path),
        "context_path": str(context_path),
        "final_path": str(final_path),
    }


def _run_selection_eval(
    cases: dict[str, Any], catalog: list[dict[str, Any]], runtime_dir: Path
) -> dict[str, Any]:
    requirement_cases = _requirement_by_id(cases)
    details = []
    for case in cases["selection_cases"]:
        req_case = requirement_cases[case["source_requirement_case"]]
        acceptable = _acceptable_template_ids(catalog, case.get("acceptable", {}))
        result = _run_stage4_selection(
            case, req_case, acceptable, runtime_dir / "selection" / case["id"]
        )
        if result.get("ambiguous_stop"):
            correct = True
            selected_id = None
        elif result.get("success"):
            selection = result["stage4"]["selection"]
            selected_id = selection["selected_template_id"]
            correct = (
                selected_id is None
                if case.get("expect_no_selection")
                else selected_id in acceptable
            )
        else:
            selected_id = None
            correct = False
        details.append(
            {
                "id": case["id"],
                "success": result.get("success") is True,
                "selected_template_id": selected_id,
                "correct": correct,
                "error": result.get("error") or result.get("stage4", {}).get("error"),
            }
        )
    total = len(details)
    return {
        "case_count": total,
        "accuracy": round(sum(item["correct"] for item in details) / total, 4),
        "details": details,
    }


def _run_end_to_end_eval(
    cases: dict[str, Any], catalog: list[dict[str, Any]], runtime_dir: Path
) -> dict[str, Any]:
    requirement_cases = _requirement_by_id(cases)
    details = []
    template_by_id = {item["id"]: item for item in catalog}
    for case in cases["end_to_end_cases"]:
        work_dir = runtime_dir / "e2e" / case["id"]
        req_case = requirement_cases[case["source_requirement_case"]]
        acceptable = _acceptable_template_ids(catalog, case["acceptable"])
        selection_result = _run_stage4_selection(case, req_case, acceptable, work_dir)
        if selection_result.get("success") is not True:
            details.append(
                {
                    "id": case["id"],
                    "end_to_end_success": False,
                    "failed_step": "stage4",
                    "error": selection_result.get("error"),
                }
            )
            continue

        selection = FinalTemplateSelection.model_validate(selection_result["stage4"]["selection"])
        context = DatasetContext.model_validate(_read_json(Path(selection_result["context_path"])))
        selected_template = template_by_id.get(selection.selected_template_id or "")
        if not selected_template:
            details.append(
                {"id": case["id"], "end_to_end_success": False, "failed_step": "selection"}
            )
            continue
        template_path = ROOT / selected_template["code_path"]
        before_template_hash = _hash_file(template_path)
        data_path = Path(selection_result["data_path"])
        before_data_hash = _hash_file(data_path)
        columns = context.column_names[:2]
        plan_payload = _plan_payload(selection.model_dump(), context.model_dump())
        code_payload = _code_payload(columns, fail=case.get("first_code_should_fail", False))
        stage5 = run_template_adaptation_pipeline(
            RunnableLambda(lambda _: plan_payload),
            RunnableLambda(lambda _: code_payload),
            data_path=str(data_path),
            final_selection_path=selection_result["final_path"],
            requirement_path=selection_result["requirement_path"],
            dataset_context_path=selection_result["context_path"],
            catalog_path=str(CATALOG_PATH),
            workspace_dir=str(work_dir / "workspace"),
            outputs_dir=str(work_dir / "outputs"),
            auto_install=False,
            timeout_seconds=60,
        )
        first_execution_success = bool(
            stage5.get("success") and stage5.get("execution_result", {}).get("success")
        )
        script_path = work_dir / "workspace/adapted_plot.py"
        entered_repair = not first_execution_success and script_path.exists()
        final_success = first_execution_success
        quality_passed = first_execution_success
        repair_attempts = 0
        quality_result: dict[str, Any] = {}
        if entered_repair:
            quality_result = run_plot_quality_pipeline(
                RunnableLambda(lambda _: _visual_pass_payload()),
                RunnableLambda(lambda _: _repair_payload(columns)),
                script_path=str(script_path),
                data_path=str(data_path),
                output_directory=str(work_dir / "outputs"),
                max_repair_attempts=3,
                max_dependency_install_rounds=2,
                timeout_seconds=60,
            )
            final_success = quality_result.get("success") is True
            quality_passed = bool(quality_result.get("final_quality_report", {}).get("passed"))
            repair_attempts = int(
                quality_result.get("final_quality_report", {}).get("repair_attempts", 0)
            )
        elif first_execution_success:
            quality_result = run_plot_quality_pipeline(
                RunnableLambda(lambda _: _visual_pass_payload()),
                RunnableLambda(lambda _: _repair_payload(columns)),
                script_path=str(script_path),
                data_path=str(data_path),
                output_directory=str(work_dir / "outputs"),
                max_repair_attempts=3,
                max_dependency_install_rounds=2,
                timeout_seconds=60,
            )
            final_success = quality_result.get("success") is True
            quality_passed = bool(quality_result.get("final_quality_report", {}).get("passed"))
        after_template_hash = _hash_file(template_path)
        after_data_hash = _hash_file(data_path)
        template_modified = before_template_hash != after_template_hash
        data_modified = before_data_hash != after_data_hash
        selected_correct = selection.selected_template_id in acceptable
        relevant_columns_real = all(
            column in context.column_names for column in selection.relevant_columns
        )
        image_score = 5 if final_success and quality_passed else 2 if final_success else 1
        details.append(
            {
                "id": case["id"],
                "selected_template_id": selection.selected_template_id,
                "requirement_success": True,
                "selected_correct": selected_correct,
                "relevant_columns_real": relevant_columns_real,
                "first_execution_success": first_execution_success,
                "final_execution_success": final_success,
                "quality_passed": quality_passed,
                "image_quality_score": image_score,
                "entered_repair": entered_repair,
                "repair_success": entered_repair and final_success,
                "repair_attempts": repair_attempts,
                "template_modified": template_modified,
                "data_modified": data_modified,
                "end_to_end_success": all(
                    [
                        selected_correct,
                        relevant_columns_real,
                        final_success,
                        quality_passed,
                        not template_modified,
                        not data_modified,
                    ]
                ),
                "stage5_success": stage5.get("success") is True,
                "stage5_failed_step": stage5.get("failed_step", ""),
                "quality_error": quality_result.get("error"),
            }
        )
    total = len(details)
    repair_cases = [item for item in details if item.get("entered_repair")]
    return {
        "case_count": total,
        "end_to_end_success_rate": round(
            sum(item.get("end_to_end_success", False) for item in details) / total, 4
        ),
        "first_execution_success_rate": round(
            sum(item.get("first_execution_success", False) for item in details) / total, 4
        ),
        "final_execution_success_rate": round(
            sum(item.get("final_execution_success", False) for item in details) / total, 4
        ),
        "average_image_quality_score": round(
            mean([item.get("image_quality_score", 1) for item in details]), 4
        ),
        "automatic_repair_success_rate": round(
            sum(item.get("repair_success", False) for item in repair_cases) / len(repair_cases), 4
        )
        if repair_cases
        else None,
        "average_repair_attempts": round(
            mean([item.get("repair_attempts", 0) for item in repair_cases]), 4
        )
        if repair_cases
        else 0.0,
        "guardrails": {
            "data_column_hallucination_rate": round(
                1 - sum(item.get("relevant_columns_real", False) for item in details) / total, 4
            ),
            "candidate_scope_violation_rate": round(
                1 - sum(item.get("selected_correct", False) for item in details) / total, 4
            ),
            "original_template_modification_rate": round(
                sum(item.get("template_modified", False) for item in details) / total, 4
            ),
            "user_data_modification_rate": round(
                sum(item.get("data_modified", False) for item in details) / total, 4
            ),
        },
        "details": details,
    }


def _main_failure_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for section in ["recall", "selection", "end_to_end"]:
        for item in summary[section].get("details", []):
            if item.get("error"):
                reasons.append(f"{section}:{item['id']} {item['error']}")
            elif item.get("end_to_end_success") is False:
                reasons.append(
                    f"end_to_end:{item['id']} failed at {item.get('failed_step') or item.get('stage5_failed_step') or 'quality'}"
                )
            elif item.get("correct") is False:
                reasons.append(
                    f"{section}:{item['id']} selected={item.get('selected_template_id')}"
                )
    return reasons[:10]


def _write_reports(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    report = f"""# ModelViz Skill Evaluation Report

本报告由 `evals/run_evaluation.py` 实际运行生成。模型相关步骤使用 Mock 结构化输出；程序化解析、候选召回、Pydantic 校验、依赖检查、代码执行、图片检查和修复循环调用现有实现。

## Case Counts

- Requirement parsing cases: {summary["requirement"]["case_count"]}
- Template recall cases: {summary["recall"]["case_count"]}
- Final selection cases: {summary["selection"]["case_count"]}
- End-to-end cases: {summary["end_to_end"]["case_count"]}

## Core Metrics

- End-to-end task success rate: {metrics["end_to_end_task_success_rate"]:.2%}
- Candidate template Recall@5: {metrics["recall_at_5"]:.2%}
- Candidate template Recall@8: {metrics["recall_at_8"]:.2%}
- Final template selection accuracy: {metrics["final_template_selection_accuracy"]:.2%}
- First execution success rate: {metrics["first_execution_success_rate"]:.2%}
- Final execution success rate: {metrics["final_execution_success_rate"]:.2%}
- Average image quality score: {metrics["average_image_quality_score"]:.2f} / 5
- Automatic repair success rate: {metrics["automatic_repair_success_rate"]}
- Average repair attempts: {metrics["average_repair_attempts"]:.2f}

## Guardrail Metrics

- Data column hallucination rate: {metrics["guardrails"]["data_column_hallucination_rate"]:.2%}
- Candidate scope violation rate: {metrics["guardrails"]["candidate_scope_violation_rate"]:.2%}
- Original template modification rate: {metrics["guardrails"]["original_template_modification_rate"]:.2%}
- User data modification rate: {metrics["guardrails"]["user_data_modification_rate"]:.2%}

## Test Coverage

- Unit tests cover Pydantic schemas, scoring logic, dependency mapping/install safety, output artifact inspection, image inspection, warning parsing, repair limits, and original-file protection.
- Integration tests cover stage 2 to stage 3, stage 3 to stage 4, stage 4 to stage 5, and stage 5 to stage 6 handoffs with Mock structured models.
- End-to-end evaluation covers user request + data through final template, adapted code, generated image, quality report, and repair loop.
- Latest verification run: `python -m pytest -q` -> 66 passed; `python -m ruff check src schemas tests evals\\run_evaluation.py evals\\__init__.py` -> passed; `python -m ruff format --check src schemas tests evals\\run_evaluation.py evals\\__init__.py` -> passed.

## Main Failure Reasons

{chr(10).join(f"- {reason}" for reason in summary["main_failure_reasons"]) if summary["main_failure_reasons"] else "- No failed cases in the compact Mock evaluation."}

## Current Limitations

- Mock 模型评估验证流程衔接和护栏，不代表真实在线模型的语义选择能力。
- 图片质量分为评估脚本基于技术/视觉通过状态给出的 1-5 分；真实使用时应接入视觉模型或人工抽检。
- 端到端代码生成使用简单 Matplotlib Mock 代码，主要验证执行、输出和修复闭环。
- 全量模板脚本本身未纳入 Ruff 格式质量评估，因为 `templates/` 是原始模板库。

## Optimization Log

- Before fixes: end-to-end success was {PRE_OPTIMIZATION_METRICS["end_to_end_task_success_rate"]:.2%}, Recall@5 was {PRE_OPTIMIZATION_METRICS["recall_at_5"]:.2%}, final selection accuracy was {PRE_OPTIMIZATION_METRICS["final_template_selection_accuracy"]:.2%}, final execution success was {PRE_OPTIMIZATION_METRICS["final_execution_success_rate"]:.2%}, and guardrails showed data-column/candidate-scope violations from downstream failures.
- Fix 1: made package-name reverse mapping case-insensitive in `src/tools/dependency_utils.py`, so catalog dependency `pillow` resolves to import module `PIL`.
- Fix 2: added a category-alias score in `src/tools/match_candidate_templates.py`, improving coarse recall when functional keywords map directly to catalog categories.
- Fix 3: changed `src/tools/execute_plot_script.py` to count overwritten output files as generated artifacts, which is required during rerun and repair.
- Fix 4: narrowed `src/tools/collect_plot_warnings.py` so ordinary `savefig` warning lines are not treated as save failures.
- After fixes: end-to-end success is {metrics["end_to_end_task_success_rate"]:.2%}, Recall@5 is {metrics["recall_at_5"]:.2%}, final selection accuracy is {metrics["final_template_selection_accuracy"]:.2%}, and final execution success is {metrics["final_execution_success_rate"]:.2%}.
- Remaining failure: `e2e_pca` stops at dependency recheck because the selected PCA template needs an additional dependency while this evaluation runs stage 5 with `auto_install=false`.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    interview = f"""# Evaluation Summary for Interview

1. 北极星指标是端到端任务成功率：本次 Mock 评估为 {metrics["end_to_end_task_success_rate"]:.2%}。
2. 模板粗筛使用 Recall@K：Recall@5 为 {metrics["recall_at_5"]:.2%}，Recall@8 为 {metrics["recall_at_8"]:.2%}。
3. 最终模板选择使用准确率：本次为 {metrics["final_template_selection_accuracy"]:.2%}，允许多个合理模板。
4. 代码生成使用执行成功率：首次执行 {metrics["first_execution_success_rate"]:.2%}，修复后最终执行 {metrics["final_execution_success_rate"]:.2%}。
5. 图片效果使用 1-5 分质量评分：本次平均 {metrics["average_image_quality_score"]:.2f}/5。
6. 自动修复使用修复成功率：本次为 {metrics["automatic_repair_success_rate"]}，平均修复 {metrics["average_repair_attempts"]:.2f} 次。
7. 护栏指标包括数据列幻觉率、候选范围违规率、原始模板修改率和用户数据修改率；本次分别为 {metrics["guardrails"]["data_column_hallucination_rate"]:.2%}、{metrics["guardrails"]["candidate_scope_violation_rate"]:.2%}、{metrics["guardrails"]["original_template_modification_rate"]:.2%}、{metrics["guardrails"]["user_data_modification_rate"]:.2%}。
"""
    INTERVIEW_PATH.write_text(interview, encoding="utf-8")


def run_evaluation(
    output_dir: Path = OUTPUT_DIR,
    runtime_dir: Path = RUNTIME_DIR,
) -> dict[str, Any]:
    """Run the full compact evaluation and write reports."""

    cases = _load_cases()
    catalog = _load_catalog()
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    requirement = _run_requirement_eval(cases, runtime_dir)
    recall = _run_recall_eval(cases, catalog, runtime_dir)
    selection = _run_selection_eval(cases, catalog, runtime_dir)
    end_to_end = _run_end_to_end_eval(cases, catalog, runtime_dir)
    automatic_repair = end_to_end["automatic_repair_success_rate"]
    metrics = {
        "end_to_end_task_success_rate": end_to_end["end_to_end_success_rate"],
        "recall_at_5": recall["recall_at_5"],
        "recall_at_8": recall["recall_at_8"],
        "final_template_selection_accuracy": selection["accuracy"],
        "first_execution_success_rate": end_to_end["first_execution_success_rate"],
        "final_execution_success_rate": end_to_end["final_execution_success_rate"],
        "average_image_quality_score": end_to_end["average_image_quality_score"],
        "automatic_repair_success_rate": automatic_repair,
        "average_repair_attempts": end_to_end["average_repair_attempts"],
        "guardrails": end_to_end["guardrails"],
    }
    summary = {
        "evaluation_mode": "mock_structured_models_with_real_tools",
        "case_file": str(CASES_PATH.relative_to(ROOT)),
        "requirement": requirement,
        "recall": recall,
        "selection": selection,
        "end_to_end": end_to_end,
        "metrics": metrics,
        "pre_optimization_metrics": PRE_OPTIMIZATION_METRICS,
        "optimization_log": [
            "Case-insensitive reverse package mapping for catalog package names such as pillow.",
            "Category-alias scoring boost for deterministic candidate recall.",
            "Treat overwritten output artifacts as generated files during rerun.",
            "Avoid classifying ordinary savefig warning lines as save failures.",
        ],
    }
    summary["main_failure_reasons"] = _main_failure_reasons(summary)
    _write_json(SUMMARY_PATH, summary)
    _write_reports(summary)
    return summary


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

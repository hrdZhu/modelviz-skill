import json

import yaml

from schemas import PlotRequirement
from src.schemas import CandidateTemplateOutput
from src.services.candidate_matching_pipeline import run_candidate_matching_pipeline
from src.tools import match_candidate_templates, rank_candidate_templates, save_candidate_templates


def _requirement(**overrides) -> dict:
    data = {
        "original_request": "分析模型预测效果，论文风格，不要雷达图。",
        "goal": "分析模型预测效果",
        "functional_keywords": ["预测评估"],
        "chart_types": [],
        "style_keywords": ["科研风"],
        "use_case": "模型评价",
        "negative_requirements": ["不要雷达图"],
        "explicit_template": False,
        "is_ambiguous": False,
        "clarification_question": "",
    }
    data.update(overrides)
    return PlotRequirement.model_validate(data).model_dump()


def _template(index: int, *, chart_type: str = "regression_fit_scatter", keyword: str = "预测"):
    return {
        "id": f"tpl_{index:02d}",
        "name": f"模板{index}",
        "category": "prediction_evaluation",
        "description": "用于模型预测评估和预测实测比较。",
        "keywords": [keyword, "模型评估", "回归评估"],
        "synonyms": ["预测实测图", "模型评估图"],
        "use_cases": ["模型评价", "论文正文"],
        "negative_keywords": [],
        "chart_type": chart_type,
        "preview": f"templates/tpl_{index:02d}/assets/preview.png",
        "code_path": f"templates/tpl_{index:02d}/plot.py",
    }


def _catalog(count=12):
    templates = [_template(i) for i in range(count)]
    templates.append(
        {
            **_template(99, chart_type="radar_chart"),
            "id": "radar_bad",
            "name": "雷达图模板",
            "synonyms": ["雷达图"],
        }
    )
    return templates


def test_candidate_matching_functional_chartless_and_negative():
    result = match_candidate_templates.invoke(
        {"user_requirement": _requirement(), "templates": _catalog(), "min_score": 0.1}
    )

    assert result["success"] is True
    assert all(item["template_id"] != "radar_bad" for item in result["candidates"])
    assert any("预测评估" in item["matched_keywords"] for item in result["candidates"])


def test_candidate_matching_explicit_chart_type():
    requirement = _requirement(
        original_request="请画预测实测散点图。",
        chart_types=["regression_fit_scatter"],
        explicit_template=True,
        negative_requirements=[],
    )
    result = match_candidate_templates.invoke(
        {"user_requirement": requirement, "templates": _catalog(), "min_score": 0.1}
    )

    assert result["success"] is True
    assert result["candidates"][0]["matched_chart_types"] == ["regression_fit_scatter"]


def test_candidate_matching_style_only_can_return_candidates():
    requirement = _requirement(
        original_request="要高级科研风。",
        goal="",
        functional_keywords=[],
        chart_types=[],
        use_case="",
        negative_requirements=[],
        is_ambiguous=True,
    )
    templates = [
        {
            **_template(1),
            "id": "report",
            "category": "multi_panel_report",
            "name": "论文报告图",
            "description": "适合论文报告展示。",
        }
    ]
    result = match_candidate_templates.invoke(
        {"user_requirement": requirement, "templates": templates, "min_score": 0.01}
    )

    assert result["success"] is True
    assert result["candidates"][0]["matched_styles"] == ["科研风"]


def test_candidate_rank_count_and_stable_order():
    candidates = [
        {
            "template_id": f"tpl_{i}",
            "template_name": str(i),
            "category": "x",
            "score": 0.5,
            "catalog_order": i,
        }
        for i in range(12)
    ]
    first = rank_candidate_templates.invoke({"candidates": candidates, "top_k": 8})
    second = rank_candidate_templates.invoke({"candidates": candidates, "top_k": 8})

    assert first == second
    assert first["success"] is True
    assert first["candidate_result"]["candidate_count"] == 8


def test_candidate_invalid_top_k_and_no_match():
    invalid = run_candidate_matching_pipeline(top_k=4)
    assert invalid["success"] is False
    assert invalid["error"]["error_type"] == "validation_error"

    no_match = match_candidate_templates.invoke(
        {
            "user_requirement": _requirement(functional_keywords=["网络关系"]),
            "templates": _catalog(),
            "min_score": 0.95,
        }
    )
    assert no_match["success"] is False
    assert no_match["error_type"] == "no_qualified_templates"


def test_candidate_save_and_pipeline(tmp_path):
    req_path = tmp_path / "workspace" / "user_requirement.json"
    cat_path = tmp_path / "docs" / "template_catalog.yaml"
    out_path = tmp_path / "workspace" / "candidate_templates.json"
    req_path.parent.mkdir(parents=True)
    cat_path.parent.mkdir(parents=True)
    req_path.write_text(json.dumps(_requirement(), ensure_ascii=False), encoding="utf-8")
    cat_path.write_text(
        yaml.safe_dump({"templates": _catalog()}, allow_unicode=True), encoding="utf-8"
    )

    result = run_candidate_matching_pipeline(
        requirement_path=str(req_path),
        catalog_path=str(cat_path),
        output_path=str(out_path),
        top_k=8,
        min_score=0.1,
    )

    assert result["success"] is True
    saved = CandidateTemplateOutput.model_validate(json.loads(out_path.read_text(encoding="utf-8")))
    assert 5 <= saved.candidate_count <= 10

    saved_direct = save_candidate_templates.invoke(
        {
            "candidate_result": saved.model_dump(),
            "output_path": str(tmp_path / "workspace" / "copy.json"),
        }
    )
    assert saved_direct["success"] is True


def test_candidate_pipeline_file_errors(tmp_path):
    result = run_candidate_matching_pipeline(requirement_path=str(tmp_path / "missing.json"))

    assert result["success"] is False
    assert result["failed_step"] == "load_user_requirement"

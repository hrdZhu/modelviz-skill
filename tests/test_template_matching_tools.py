import json
from pathlib import Path

import pytest
import yaml

from schemas import PlotRequirement
from src.schemas import TemplateRecommendationOutput
from src.services import template_matching_pipeline
from src.services.template_matching_pipeline import run_template_matching_pipeline
from src.tools import (
    load_template_catalog,
    load_user_requirement,
    match_templates,
    rank_templates,
    save_template_recommendations,
)


def _write_requirement(path: Path, **overrides) -> dict:
    data = {
        "original_request": "分析年度变化趋势，论文风格。",
        "goal": "分析年度变化趋势",
        "functional_keywords": ["趋势"],
        "chart_types": [],
        "style_keywords": ["科研风"],
        "use_case": "论文正文",
        "negative_requirements": [],
        "explicit_template": False,
        "is_ambiguous": False,
        "clarification_question": "",
    }
    data.update(overrides)
    requirement = PlotRequirement.model_validate(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(requirement.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return requirement.model_dump()


def _template(
    template_id: str,
    *,
    name: str,
    category: str,
    chart_type: str,
    keywords: list[str],
    synonyms: list[str] | None = None,
    use_cases: list[str] | None = None,
    negative_keywords: list[str] | None = None,
) -> dict:
    return {
        "id": template_id,
        "name": name,
        "category": category,
        "description": f"{name}用于{category}分析。",
        "keywords": keywords,
        "synonyms": synonyms or [],
        "use_cases": use_cases or [],
        "negative_keywords": negative_keywords or [],
        "chart_type": chart_type,
        "preview": f"templates/{template_id}/assets/preview.png",
        "code_path": f"templates/{template_id}/{name}.py",
    }


def _sample_templates() -> list[dict]:
    return [
        _template(
            "trend_line",
            name="平滑趋势曲线",
            category="trend_time_series",
            chart_type="smoothed_line_chart_grid",
            keywords=["趋势", "时间序列", "多组折线"],
            synonyms=["趋势曲线", "年度变化图"],
            use_cases=["论文正文", "展示多个地区随时间变化的平滑趋势"],
            negative_keywords=["空间栅格制图"],
        ),
        _template(
            "radar_compare",
            name="多指标雷达对比图",
            category="comparison_ranking",
            chart_type="radar_chart",
            keywords=["比较", "多指标评价"],
            synonyms=["雷达图", "蛛网图"],
            use_cases=["比较多个模型或方案的多项指标"],
        ),
        _template(
            "shap_report",
            name="SHAP组合分析图",
            category="sensitivity_robustness",
            chart_type="shap_multi_panel_report",
            keywords=["SHAP", "特征重要性", "模型解释", "多面板"],
            synonyms=["SHAP综合图", "特征贡献报告"],
            use_cases=["模型解释", "论文正文"],
        ),
    ]


def _write_catalog(path: Path, templates: list[dict] | None = None) -> list[dict]:
    templates = templates or _sample_templates()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"templates": templates}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return templates


def test_load_user_requirement_success(tmp_path):
    path = tmp_path / "workspace" / "user_requirement.json"
    _write_requirement(path)

    result = load_user_requirement.invoke({"requirement_path": str(path)})

    assert result["success"] is True
    assert result["requirement"]["functional_keywords"] == ["趋势"]


def test_load_user_requirement_missing_and_broken_json(tmp_path):
    missing = load_user_requirement.invoke({"requirement_path": str(tmp_path / "missing.json")})
    assert missing["success"] is False
    assert missing["error_type"] == "file_not_found"

    broken_path = tmp_path / "workspace" / "user_requirement.json"
    broken_path.parent.mkdir(parents=True)
    broken_path.write_text("{bad json", encoding="utf-8")
    broken = load_user_requirement.invoke({"requirement_path": str(broken_path)})
    assert broken["success"] is False
    assert broken["error_type"] == "json_decode_error"


def test_load_template_catalog_success_and_errors(tmp_path):
    catalog_path = tmp_path / "docs" / "template_catalog.yaml"
    _write_catalog(catalog_path)

    loaded = load_template_catalog.invoke({"catalog_path": str(catalog_path)})
    assert loaded["success"] is True
    assert loaded["total_templates"] == 3

    missing = load_template_catalog.invoke({"catalog_path": str(tmp_path / "missing.yaml")})
    assert missing["success"] is False
    assert missing["error_type"] == "file_not_found"

    broken_path = tmp_path / "docs" / "broken.yaml"
    broken_path.write_text("templates: [", encoding="utf-8")
    broken = load_template_catalog.invoke({"catalog_path": str(broken_path)})
    assert broken["success"] is False
    assert broken["error_type"] == "yaml_decode_error"

    empty_path = tmp_path / "docs" / "empty.yaml"
    empty_path.write_text("templates: []", encoding="utf-8")
    empty = load_template_catalog.invoke({"catalog_path": str(empty_path)})
    assert empty["success"] is False
    assert empty["error_type"] == "empty_catalog"

    missing_field_path = tmp_path / "docs" / "missing_field.yaml"
    missing_field_path.write_text(
        yaml.safe_dump({"templates": [{"id": "x"}]}, allow_unicode=True),
        encoding="utf-8",
    )
    missing_field = load_template_catalog.invoke({"catalog_path": str(missing_field_path)})
    assert missing_field["success"] is False
    assert missing_field["error_type"] == "missing_template_fields"


def test_match_functional_keywords_and_no_chart_type():
    requirement = PlotRequirement(
        original_request="分析年度变化趋势。",
        goal="年度变化趋势",
        functional_keywords=["趋势"],
        chart_types=[],
        style_keywords=[],
        use_case="论文正文",
    ).model_dump()

    result = match_templates.invoke(
        {"user_requirement": requirement, "templates": _sample_templates(), "min_score": 0.1}
    )

    assert result["success"] is True
    top_ids = [item["template_id"] for item in result["candidates"]]
    assert "trend_line" in top_ids
    trend = next(item for item in result["candidates"] if item["template_id"] == "trend_line")
    assert trend["matched_keywords"] == ["趋势"]
    assert trend["matched_chart_types"] == []


def test_match_explicit_chart_type_and_style_requirement():
    requirement = PlotRequirement(
        original_request="请画雷达图比较三个方案，科研风。",
        goal="比较三个方案",
        functional_keywords=["比较"],
        chart_types=["radar_chart"],
        style_keywords=["科研风"],
        use_case="",
        explicit_template=True,
    ).model_dump()

    result = match_templates.invoke(
        {"user_requirement": requirement, "templates": _sample_templates(), "min_score": 0.1}
    )

    radar = next(item for item in result["candidates"] if item["template_id"] == "radar_compare")
    assert radar["matched_chart_types"] == ["radar_chart"]
    assert radar["score"] > 0.4


def test_user_negative_requirement_excludes_template():
    requirement = PlotRequirement(
        original_request="分析年度趋势并做比较，不要雷达图。",
        goal="分析年度趋势并做比较",
        functional_keywords=["趋势", "比较"],
        chart_types=[],
        negative_requirements=["不要雷达图"],
    ).model_dump()

    result = match_templates.invoke(
        {"user_requirement": requirement, "templates": _sample_templates(), "min_score": 0.05}
    )

    assert result["success"] is True
    assert "radar_compare" not in [item["template_id"] for item in result["candidates"]]


def test_template_negative_keyword_penalty_can_remove_candidate():
    requirement = PlotRequirement(
        original_request="分析趋势，但这是空间栅格制图任务。",
        goal="趋势",
        functional_keywords=["趋势"],
        chart_types=[],
    ).model_dump()

    result = match_templates.invoke(
        {"user_requirement": requirement, "templates": _sample_templates(), "min_score": 0.1}
    )

    assert result["success"] is False
    assert result["error_type"] == "no_qualified_templates"


def test_no_template_reaches_min_score():
    requirement = PlotRequirement(
        original_request="画一个完全无关的集合关系图。",
        goal="集合关系",
        functional_keywords=["网络关系"],
        chart_types=[],
    ).model_dump()

    result = match_templates.invoke(
        {"user_requirement": requirement, "templates": _sample_templates(), "min_score": 0.9}
    )

    assert result["success"] is False
    assert result["error_type"] == "no_qualified_templates"


def test_rank_templates_stable_order_and_top_k():
    candidates = [
        {
            "template_id": "b",
            "template_name": "B",
            "category": "x",
            "score": 0.8,
            "original_order": 2,
        },
        {
            "template_id": "a",
            "template_name": "A",
            "category": "x",
            "score": 0.8,
            "original_order": 1,
        },
        {
            "template_id": "c",
            "template_name": "C",
            "category": "x",
            "score": 0.6,
            "original_order": 0,
        },
    ]

    result = rank_templates.invoke({"candidates": candidates, "top_k": 2})

    assert result["success"] is True
    assert [item["template_id"] for item in result["ranked_result"]["recommendations"]] == [
        "a",
        "b",
    ]


def test_save_template_recommendations_success_and_failure(tmp_path):
    output = TemplateRecommendationOutput(
        original_request="测试",
        total_templates=1,
        qualified_templates=1,
        recommendations=[
            {
                "template_id": "trend_line",
                "template_name": "平滑趋势曲线",
                "category": "trend_time_series",
                "score": 0.5,
            }
        ],
        warnings=[],
    ).model_dump()
    output_path = tmp_path / "workspace" / "template_recommendations.json"

    saved = save_template_recommendations.invoke(
        {"recommendation_result": output, "output_path": str(output_path)}
    )

    assert saved["success"] is True
    saved_text = output_path.read_text(encoding="utf-8")
    assert "\\u" not in saved_text
    assert TemplateRecommendationOutput.model_validate(json.loads(saved_text))

    blocking_file = tmp_path / "blocked"
    blocking_file.write_text("not a dir", encoding="utf-8")
    failed = save_template_recommendations.invoke(
        {
            "recommendation_result": output,
            "output_path": str(blocking_file / "template_recommendations.json"),
        }
    )
    assert failed["success"] is False
    assert failed["error_type"] == "output_directory_error"


def test_pipeline_runs_tools_in_order_and_saves(tmp_path, monkeypatch):
    requirement_path = tmp_path / "workspace" / "user_requirement.json"
    catalog_path = tmp_path / "docs" / "template_catalog.yaml"
    output_path = tmp_path / "workspace" / "template_recommendations.json"
    _write_requirement(requirement_path)
    _write_catalog(catalog_path)

    calls: list[str] = []
    original_invoke = template_matching_pipeline._invoke_tool

    def recording_invoke(tool_obj, payload):
        calls.append(tool_obj.name)
        return original_invoke(tool_obj, payload)

    monkeypatch.setattr(template_matching_pipeline, "_invoke_tool", recording_invoke)

    result = run_template_matching_pipeline(
        requirement_path=str(requirement_path),
        catalog_path=str(catalog_path),
        top_k=2,
        min_score=0.1,
        output_path=str(output_path),
    )

    assert result["success"] is True
    assert calls == [
        "load_user_requirement",
        "load_template_catalog",
        "match_templates",
        "rank_templates",
        "save_template_recommendations",
    ]
    assert output_path.exists()


@pytest.mark.parametrize(
    ("top_k", "min_score"),
    [(0, 0.1), (5, -0.1), (21, 0.1), (5, 1.1)],
)
def test_pipeline_invalid_top_k_or_min_score(top_k, min_score):
    result = run_template_matching_pipeline(top_k=top_k, min_score=min_score)

    assert result["success"] is False
    assert result["failed_step"] == "validate_pipeline_input"
    assert result["error"]["error_type"] == "validation_error"

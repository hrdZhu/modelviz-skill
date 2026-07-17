import json

import pandas as pd
from langchain_core.runnables import RunnableLambda

from schemas import PlotRequirement
from src.schemas import CandidateTemplateOutput, FinalTemplateSelection
from src.services.final_template_selection_pipeline import run_final_template_selection_pipeline


def _write_requirement(path):
    req = PlotRequirement(
        original_request="比较不同地区产量随年份变化的趋势。",
        goal="比较趋势",
        functional_keywords=["趋势", "比较"],
        chart_types=[],
        style_keywords=["科研风"],
        use_case="论文正文",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(req.model_dump(), ensure_ascii=False), encoding="utf-8")


def _candidate(template_id="trend_line", name="平滑趋势曲线", score=0.8):
    return {
        "template_id": template_id,
        "template_name": name,
        "category": "trend_time_series",
        "score": score,
        "matched_keywords": ["趋势"],
        "matched_synonyms": [],
        "matched_chart_types": [],
        "matched_use_cases": ["论文正文"],
        "matched_styles": ["科研风"],
        "penalties": [],
        "candidate_reason": "功能关键词匹配",
        "preview": "preview.png",
        "code_path": "plot.py",
        "catalog_order": 0,
    }


def _write_candidates(path, candidates=None):
    actual_candidates = (
        [_candidate(), _candidate("radar", "雷达图", 0.4)] if candidates is None else candidates
    )
    output = CandidateTemplateOutput(
        original_request="比较不同地区产量随年份变化的趋势。",
        candidate_count=len(actual_candidates),
        total_templates=10,
        candidates=actual_candidates,
        warnings=[],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output.model_dump(), ensure_ascii=False), encoding="utf-8")


def _write_data(path, rows=5, chinese=True):
    if chinese:
        df = pd.DataFrame(
            {"年份": range(2020, 2020 + rows), "地区": ["A"] * rows, "产量": range(rows)}
        )
    else:
        df = pd.DataFrame({"year": range(rows), "category": ["A"] * rows, "value": range(rows)})
    df.to_csv(path, index=False, encoding="utf-8")


def _model(payload):
    return RunnableLambda(lambda _: payload)


def _selection(**overrides):
    data = {
        "dataset_summary": "数据包含年份、地区和产量。",
        "observed_data_features": ["包含时间字段", "包含类别与数值字段"],
        "relevant_columns": ["年份", "地区", "产量"],
        "candidate_comparisons": [
            {
                "template_id": "trend_line",
                "suitable": True,
                "advantages": ["适合趋势比较"],
                "limitations": [],
                "data_compatibility": "数据包含时间和数值列",
                "requirement_compatibility": "符合趋势比较需求",
            }
        ],
        "selected_template_id": "trend_line",
        "selected_template_name": "平滑趋势曲线",
        "alternative_template_ids": ["radar", "radar"],
        "selection_reason": "最符合趋势分析。",
        "data_support_reason": "年份和产量支持趋势图。",
        "data_warnings": [],
        "confidence": 0.85,
        "needs_clarification": False,
        "clarification_question": "",
    }
    data.update(overrides)
    return data


def test_final_selection_supported_candidate_and_save(tmp_path):
    req_path = tmp_path / "workspace" / "user_requirement.json"
    cand_path = tmp_path / "workspace" / "candidate_templates.json"
    data_path = tmp_path / "data.csv"
    out_path = tmp_path / "workspace" / "final_template_selection.json"
    _write_requirement(req_path)
    _write_candidates(cand_path)
    _write_data(data_path)

    result = run_final_template_selection_pipeline(
        _model(_selection()),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
        output_path=str(out_path),
        dataset_context_path=str(tmp_path / "workspace" / "dataset_context.json"),
    )

    assert result["success"] is True
    saved = FinalTemplateSelection.model_validate(json.loads(out_path.read_text(encoding="utf-8")))
    assert saved.selected_template_id == "trend_line"
    assert saved.alternative_template_ids == ["radar"]


def test_final_selection_chart_requested_but_data_not_supported_clarifies(tmp_path):
    req_path = tmp_path / "workspace" / "user_requirement.json"
    cand_path = tmp_path / "workspace" / "candidate_templates.json"
    data_path = tmp_path / "data.csv"
    _write_requirement(req_path)
    _write_candidates(cand_path)
    _write_data(data_path)

    payload = _selection(
        selected_template_id=None,
        selected_template_name=None,
        alternative_template_ids=[],
        relevant_columns=["年份"],
        needs_clarification=True,
        clarification_question="请提供用于比较的数值列。",
        confidence=0.2,
    )
    result = run_final_template_selection_pipeline(
        _model(payload),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
        output_path=str(tmp_path / "out.json"),
    )

    assert result["success"] is True
    assert result["selection"]["needs_clarification"] is True


def test_final_selection_rejects_fake_column_and_template(tmp_path):
    req_path = tmp_path / "workspace" / "user_requirement.json"
    cand_path = tmp_path / "workspace" / "candidate_templates.json"
    data_path = tmp_path / "data.csv"
    _write_requirement(req_path)
    _write_candidates(cand_path)
    _write_data(data_path)

    fake_column = run_final_template_selection_pipeline(
        _model(_selection(relevant_columns=["不存在列"])),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
    )
    assert fake_column["success"] is False
    assert fake_column["error"]["error_type"] == "invalid_data_column"

    fake_template = run_final_template_selection_pipeline(
        _model(_selection(selected_template_id="bad", selected_template_name="坏模板")),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
    )
    assert fake_template["success"] is False
    assert fake_template["error"]["error_type"] == "invalid_candidate_template"


def test_final_selection_structured_output_error_and_empty_candidates(tmp_path):
    req_path = tmp_path / "workspace" / "user_requirement.json"
    cand_path = tmp_path / "workspace" / "candidate_templates.json"
    data_path = tmp_path / "data.csv"
    _write_requirement(req_path)
    _write_data(data_path)
    _write_candidates(cand_path, candidates=[])

    empty = run_final_template_selection_pipeline(
        _model(_selection()),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
    )
    assert empty["success"] is False
    assert empty["error"]["error_type"] == "empty_candidates"

    _write_candidates(cand_path)
    bad_model = run_final_template_selection_pipeline(
        _model({"selected_template_id": "trend_line"}),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
    )
    assert bad_model["success"] is False
    assert bad_model["error"]["error_type"] == "structured_output_error"


def test_stage3_stage4_integration_with_large_data(tmp_path):
    from tests.test_candidate_matching import _catalog, _requirement
    from src.services.candidate_matching_pipeline import run_candidate_matching_pipeline

    req_path = tmp_path / "workspace" / "user_requirement.json"
    cat_path = tmp_path / "docs" / "template_catalog.yaml"
    data_path = tmp_path / "large.csv"
    cand_path = tmp_path / "workspace" / "candidate_templates.json"
    req_path.parent.mkdir(parents=True)
    cat_path.parent.mkdir(parents=True)
    req_path.write_text(json.dumps(_requirement(), ensure_ascii=False), encoding="utf-8")
    import yaml

    cat_path.write_text(
        yaml.safe_dump({"templates": _catalog()}, allow_unicode=True), encoding="utf-8"
    )
    _write_data(data_path, rows=1200)

    stage3 = run_candidate_matching_pipeline(
        requirement_path=str(req_path),
        catalog_path=str(cat_path),
        output_path=str(cand_path),
        top_k=8,
        min_score=0.1,
    )
    assert stage3["success"] is True

    selected = stage3["candidate_result"]["candidates"][0]
    payload = _selection(
        selected_template_id=selected["template_id"],
        selected_template_name=selected["template_name"],
        alternative_template_ids=[],
    )
    stage4 = run_final_template_selection_pipeline(
        _model(payload),
        data_path=str(data_path),
        requirement_path=str(req_path),
        candidate_path=str(cand_path),
        dataset_context_path=str(tmp_path / "workspace" / "dataset_context.json"),
    )
    assert stage4["success"] is True
    context = json.loads(
        (tmp_path / "workspace" / "dataset_context.json").read_text(encoding="utf-8")
    )
    assert context["sampled"] is True

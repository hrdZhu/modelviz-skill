import json

import yaml
from langchain_core.runnables import RunnableLambda

from schemas import PlotRequirement
from src.schemas import DatasetContext, FinalTemplateSelection
from src.services.template_adaptation_pipeline import run_template_adaptation_pipeline


def _write_stage5_files(tmp_path, template_source: str = "import json\n"):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    template_path = tmp_path / "template.py"
    template_path.write_text(template_source, encoding="utf-8")
    data_path = tmp_path / "data.csv"
    data_path.write_text("年份,产量\n2020,1\n2021,2\n", encoding="utf-8")

    requirement = PlotRequirement(
        original_request="画年份和产量趋势图。",
        goal="趋势",
        functional_keywords=["趋势"],
        chart_types=[],
        style_keywords=["科研风"],
        use_case="论文正文",
    )
    (workspace / "user_requirement.json").write_text(
        json.dumps(requirement.model_dump(), ensure_ascii=False),
        encoding="utf-8",
    )
    context = DatasetContext(
        file_name="data.csv",
        file_type="csv",
        row_count=2,
        column_count=2,
        column_names=["年份", "产量"],
        sample_records=[{"年份": 2020, "产量": 1}],
        head_records=[{"年份": 2020, "产量": 1}],
    )
    (workspace / "dataset_context.json").write_text(
        json.dumps(context.model_dump(), ensure_ascii=False),
        encoding="utf-8",
    )
    selection = FinalTemplateSelection(
        dataset_summary="ok",
        relevant_columns=["年份", "产量"],
        selected_template_id="tpl",
        selected_template_name="趋势模板",
        confidence=0.9,
    )
    (workspace / "final_template_selection.json").write_text(
        json.dumps(selection.model_dump(), ensure_ascii=False),
        encoding="utf-8",
    )
    catalog = {
        "templates": [
            {
                "id": "tpl",
                "name": "趋势模板",
                "category": "trend_time_series",
                "description": "趋势",
                "keywords": ["趋势"],
                "synonyms": [],
                "use_cases": ["论文正文"],
                "negative_keywords": [],
                "chart_type": "line",
                "preview": "",
                "code_path": str(template_path),
                "dependencies": ["json"],
            }
        ]
    }
    catalog_path = tmp_path / "template_catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, allow_unicode=True), encoding="utf-8")
    return workspace, catalog_path, template_path, data_path


def _plan(**overrides):
    data = {
        "template_id": "tpl",
        "template_name": "趋势模板",
        "plot_goal": "趋势",
        "selected_columns": ["年份", "产量"],
        "column_mappings": [
            {"data_column": "年份", "template_role": "x"},
            {"data_column": "产量", "template_role": "y"},
        ],
        "required_preprocessing": [],
        "required_dependencies": [],
        "layout_elements_to_preserve": ["单图布局"],
        "style_elements_to_preserve": ["简洁配色"],
        "elements_allowed_to_change": ["标题"],
        "title_plan": "趋势图",
        "axis_plan": "年份-产量",
        "legend_plan": "",
        "annotation_plan": "",
        "output_formats": ["png"],
        "warnings": [],
        "can_proceed": True,
        "clarification_question": "",
    }
    data.update(overrides)
    return data


def _code(**overrides):
    adapted = """
import base64
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
png = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAIElEQVR4nGP8z8Dwn4ECwESJ5lEDRg0YNWDUgFEDBgA0MAIf5sj2xQAAAABJRU5ErkJggg=='
(output_dir / 'chart.png').write_bytes(base64.b64decode(png))
"""
    data = {
        "adapted_code": adapted,
        "changes_summary": ["替换模拟数据"],
        "preserved_style_elements": ["布局"],
        "changed_elements": ["数据读取"],
        "data_columns_used": ["年份", "产量"],
        "dependencies_used": [],
        "additional_dependencies_requested": [],
        "assumptions": [],
        "warnings": [],
    }
    data.update(overrides)
    return data


def test_template_adaptation_pipeline_success_and_template_unchanged(tmp_path):
    workspace, catalog_path, template_path, data_path = _write_stage5_files(tmp_path)
    original = template_path.read_text(encoding="utf-8")

    result = run_template_adaptation_pipeline(
        RunnableLambda(lambda _: _plan()),
        RunnableLambda(lambda _: _code()),
        data_path=str(data_path),
        final_selection_path=str(workspace / "final_template_selection.json"),
        requirement_path=str(workspace / "user_requirement.json"),
        dataset_context_path=str(workspace / "dataset_context.json"),
        catalog_path=str(catalog_path),
        workspace_dir=str(workspace),
        outputs_dir=str(tmp_path / "outputs"),
    )

    assert result["success"] is True
    assert (workspace / "adapted_plot.py").exists()
    assert template_path.read_text(encoding="utf-8") == original
    assert result["execution_result"]["generated_files"]


def test_template_adaptation_stops_on_dependency_failure_before_model(tmp_path):
    workspace, catalog_path, _, data_path = _write_stage5_files(
        tmp_path, "import fake_missing_dependency\n"
    )
    called = {"plan": False}

    def plan(_):
        called["plan"] = True
        return _plan()

    result = run_template_adaptation_pipeline(
        RunnableLambda(plan),
        RunnableLambda(lambda _: _code()),
        data_path=str(data_path),
        final_selection_path=str(workspace / "final_template_selection.json"),
        requirement_path=str(workspace / "user_requirement.json"),
        dataset_context_path=str(workspace / "dataset_context.json"),
        catalog_path=str(catalog_path),
        workspace_dir=str(workspace),
        auto_install=False,
    )

    assert result["success"] is False
    assert called["plan"] is False


def test_template_adaptation_rejects_fake_column_and_unchecked_dependency(tmp_path):
    workspace, catalog_path, _, data_path = _write_stage5_files(tmp_path)
    fake_column = run_template_adaptation_pipeline(
        RunnableLambda(lambda _: _plan(selected_columns=["不存在列"])),
        RunnableLambda(lambda _: _code()),
        data_path=str(data_path),
        final_selection_path=str(workspace / "final_template_selection.json"),
        requirement_path=str(workspace / "user_requirement.json"),
        dataset_context_path=str(workspace / "dataset_context.json"),
        catalog_path=str(catalog_path),
        workspace_dir=str(workspace),
    )
    assert fake_column["success"] is False
    assert fake_column["error"]["error_type"] == "user_data_column_missing"

    unchecked_dep = run_template_adaptation_pipeline(
        RunnableLambda(lambda _: _plan()),
        RunnableLambda(lambda _: _code(dependencies_used=["newlib"])),
        data_path=str(data_path),
        final_selection_path=str(workspace / "final_template_selection.json"),
        requirement_path=str(workspace / "user_requirement.json"),
        dataset_context_path=str(workspace / "dataset_context.json"),
        catalog_path=str(catalog_path),
        workspace_dir=str(workspace),
    )
    assert unchecked_dep["success"] is False
    assert unchecked_dep["error"]["error_type"] == "unchecked_dependency"

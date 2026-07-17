from pathlib import Path

from langchain_core.runnables import RunnableLambda

from src.services.plot_quality_pipeline import run_plot_quality_pipeline


def _image_script():
    return """
import sys
from pathlib import Path
from PIL import Image, ImageDraw
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
img = Image.new('RGB', (400, 300), 'white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 350, 250], fill='blue')
img.save(out / 'chart.png')
"""


def _no_output_script():
    return "print('no output')\n"


def _visual_sequence(*reports):
    state = {"i": 0}

    def call(_):
        index = min(state["i"], len(reports) - 1)
        state["i"] += 1
        return reports[index]

    return RunnableLambda(call)


def _visual(passed=True, issue=""):
    return {
        "passed": passed,
        "requirement_alignment": "ok",
        "data_expression_quality": "ok",
        "style_preservation": "ok",
        "readability": "ok",
        "layout_quality": "ok",
        "color_quality": "ok",
        "issues": [issue] if issue else [],
        "suggested_fixes": ["fix"] if issue else [],
        "needs_repair": not passed,
        "confidence": 0.9,
    }


def _repair(code, additional=None, can_retry=True):
    return {
        "repaired_code": code,
        "fixed_issues": ["fixed"],
        "remaining_risks": [],
        "changes_summary": ["changed"],
        "data_columns_used": [],
        "dependencies_used": [],
        "additional_dependencies_requested": additional or [],
        "can_retry": can_retry,
    }


def test_plot_quality_pipeline_passes_without_repair(tmp_path):
    script = tmp_path / "adapted_plot.py"
    script.write_text(_image_script(), encoding="utf-8")
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n", encoding="utf-8")

    result = run_plot_quality_pipeline(
        _visual_sequence(_visual(True)),
        RunnableLambda(lambda _: _repair(_image_script())),
        script_path=str(script),
        data_path=str(data),
        output_directory=str(tmp_path / "outputs"),
    )

    assert result["success"] is True
    assert result["final_quality_report"]["passed"] is True


def test_plot_quality_pipeline_repairs_then_passes(tmp_path):
    script = tmp_path / "adapted_plot.py"
    script.write_text(_no_output_script(), encoding="utf-8")
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n", encoding="utf-8")

    result = run_plot_quality_pipeline(
        _visual_sequence(_visual(False, "标签重叠"), _visual(True)),
        RunnableLambda(lambda _: _repair(_image_script())),
        script_path=str(script),
        data_path=str(data),
        output_directory=str(tmp_path / "outputs"),
        max_repair_attempts=2,
    )

    assert result["success"] is True
    assert Path("workspace/repair_history.json").exists()


def test_plot_quality_pipeline_repair_no_change_and_max_attempts(tmp_path):
    script = tmp_path / "adapted_plot.py"
    script.write_text(_no_output_script(), encoding="utf-8")
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n", encoding="utf-8")

    no_change = run_plot_quality_pipeline(
        _visual_sequence(_visual(False, "空白图")),
        RunnableLambda(lambda _: _repair(_no_output_script())),
        script_path=str(script),
        data_path=str(data),
        output_directory=str(tmp_path / "outputs1"),
        max_repair_attempts=1,
    )
    assert no_change["success"] is False
    assert no_change["error"]["error_type"] == "repair_no_change"

    script.write_text(_no_output_script(), encoding="utf-8")
    maxed = run_plot_quality_pipeline(
        _visual_sequence(_visual(False, "问题1"), _visual(False, "问题2")),
        RunnableLambda(lambda _: _repair("print('still no output')\n")),
        script_path=str(script),
        data_path=str(data),
        output_directory=str(tmp_path / "outputs2"),
        max_repair_attempts=1,
    )
    assert maxed["success"] is False
    assert maxed["error"]["error_type"] == "max_repair_attempts_reached"


def test_plot_quality_pipeline_dependency_round_limit(tmp_path):
    script = tmp_path / "adapted_plot.py"
    script.write_text(_no_output_script(), encoding="utf-8")
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n", encoding="utf-8")

    result = run_plot_quality_pipeline(
        _visual_sequence(_visual(False, "缺依赖"), _visual(False, "还缺依赖")),
        RunnableLambda(lambda _: _repair("print('x')\n", additional=["already_installed_demo"])),
        script_path=str(script),
        data_path=str(data),
        output_directory=str(tmp_path / "outputs"),
        max_repair_attempts=2,
        max_dependency_install_rounds=0,
    )

    assert result["success"] is False
    assert result["error"]["error_type"] == "dependency_install_round_limit"

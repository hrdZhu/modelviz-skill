import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_case_set_counts_and_coverage():
    cases = json.loads((ROOT / "evals/evaluation_cases.json").read_text(encoding="utf-8"))

    assert 20 <= len(cases["requirement_cases"]) <= 30
    assert 15 <= len(cases["recall_cases"]) <= 20
    assert 10 <= len(cases["selection_cases"]) <= 15
    assert 8 <= len(cases["end_to_end_cases"]) <= 10
    all_requests = " ".join(case["input"] for case in cases["requirement_cases"])
    for keyword in ["不要", "高级科研图", "PCA", "ROC", "小提琴", "SHAP"]:
        assert keyword in all_requests


def test_evaluation_reports_exist_and_use_actual_metrics():
    summary_path = ROOT / "evals/evaluation_summary.json"
    report_path = ROOT / "evals/evaluation_report.md"
    interview_path = ROOT / "docs/evaluation_summary_for_interview.md"

    assert summary_path.exists()
    assert report_path.exists()
    assert interview_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["evaluation_mode"] == "mock_structured_models_with_real_tools"
    assert "metrics" in summary
    assert "pre_optimization_metrics" in summary
    assert summary["metrics"]["recall_at_5"] >= 0
    assert "End-to-end task success rate" in report_path.read_text(encoding="utf-8")
    assert "北极星指标" in interview_path.read_text(encoding="utf-8")

import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from schemas import PlotRequirement
from src.services.requirement_parser import parse_and_save_requirement


CASES_PATH = Path("examples/requirement_test_cases.json")
STYLE_WORDS = {"高级", "高级科研图", "科研风", "论文风", "学术风", "简洁", "低饱和"}


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _contains_any(text: str, words: list[str]) -> bool:
    return any(word in text for word in words)


def _negated(text: str, term: str) -> bool:
    return f"不要{term}" in text or f"别用{term}" in text or f"不用{term}" in text


def _heuristic_model_output(user_request: str) -> dict:
    """测试用通用解析器，模拟结构化模型输出。

    它只用于自动化测试，不参与生产解析逻辑。
    """

    text = user_request
    functional_keywords: list[str] = []
    chart_types: list[str] = []
    style_keywords: list[str] = []
    negative_requirements: list[str] = []
    use_case = ""

    if _contains_any(text, ["科研风", "论文风", "论文风格", "学术风", "高级", "专业一点"]):
        _append_unique(style_keywords, "科研风")
        # 故意模拟模型把风格词误放入功能词，验证后处理能移除它。
        _append_unique(functional_keywords, "高级")
    if _contains_any(text, ["简洁", "清爽", "干净"]):
        _append_unique(style_keywords, "简洁")
        _append_unique(functional_keywords, "简洁")
    if _contains_any(text, ["低饱和", "柔和配色", "莫兰迪"]):
        _append_unique(style_keywords, "低饱和")
        _append_unique(functional_keywords, "低饱和")

    if _contains_any(text, ["不要雷达图", "别用雷达图", "不用雷达图"]):
        _append_unique(negative_requirements, "不要雷达图")
    if _contains_any(text, ["不要三维", "不要3D", "不用立体", "不要三维效果"]):
        _append_unique(negative_requirements, "不要三维")
    if _contains_any(text, ["不要饼图", "别画饼图", "不想要饼"]):
        _append_unique(negative_requirements, "不要饼图")

    if "雷达图" in text and not _negated(text, "雷达图"):
        _append_unique(chart_types, "radar_chart")
    if _contains_any(text, ["散点图", "回归拟合散点"]):
        _append_unique(chart_types, "regression_fit_scatter")
    if _contains_any(text, ["热图", "热力图"]):
        _append_unique(chart_types, "correlation_heatmap")
    if _contains_any(text, ["多面板图", "组合图", "报告图"]):
        _append_unique(chart_types, "multi_panel_report")

    if _contains_any(text, ["趋势", "随年份", "随时间", "年度变化"]):
        _append_unique(functional_keywords, "趋势")
    if _contains_any(text, ["比较", "对比", "差异", "谁更好", "组间"]):
        _append_unique(functional_keywords, "比较")
    if _contains_any(text, ["显著差异", "显著性", "显著"]):
        _append_unique(functional_keywords, "显著性")
    if _contains_any(text, ["不确定性", "误差", "置信区间"]):
        _append_unique(functional_keywords, "不确定性")
    if _contains_any(text, ["预测值", "真实值", "模型评价", "预测效果"]):
        _append_unique(functional_keywords, "预测评估")
        use_case = "模型评价"
    if _contains_any(text, ["哪个因素最重要", "特征贡献", "SHAP", "变量重要性"]):
        _append_unique(functional_keywords, "模型解释")
    if _contains_any(text, ["PCA", "主成分", "降维"]):
        _append_unique(functional_keywords, "聚类降维")
    if _contains_any(text, ["多面板", "同时展示"]):
        _append_unique(functional_keywords, "多面板")

    if _contains_any(text, ["变量筛选", "指标筛选"]):
        use_case = "变量筛选"
    if _contains_any(text, ["处理组", "组间"]):
        use_case = "组间差异"
    if _contains_any(text, ["论文正文", "论文图", "正文图"]):
        use_case = "论文正文"

    explicit_template = bool(chart_types)
    is_ambiguous = not functional_keywords and not chart_types
    clarification_question = ""
    if is_ambiguous:
        clarification_question = "请说明你想展示的分析目标、数据关系或图表类型。"

    goal = ""
    if functional_keywords:
        goal = "、".join(functional_keywords[:3])

    return {
        "original_request": "",
        "goal": goal,
        "functional_keywords": functional_keywords,
        "chart_types": chart_types,
        "style_keywords": style_keywords,
        "use_case": use_case,
        "negative_requirements": negative_requirements,
        "explicit_template": explicit_template,
        "is_ambiguous": is_ambiguous,
        "clarification_question": clarification_question,
    }


def _fake_model_for(user_request: str):
    return RunnableLambda(lambda _: _heuristic_model_output(user_request))


def _load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _assert_expected_subset(result: PlotRequirement, expected: dict) -> None:
    data = result.model_dump()
    for field, expected_value in expected.items():
        actual_value = data[field]
        if isinstance(expected_value, list):
            if expected_value:
                for item in expected_value:
                    assert item in actual_value
            else:
                assert actual_value == []
        else:
            assert actual_value == expected_value


def _assert_forbidden_absent(result: PlotRequirement, forbidden: dict) -> None:
    data = result.model_dump()
    for field, forbidden_values in forbidden.items():
        actual_value = data[field]
        if isinstance(actual_value, list):
            for item in forbidden_values:
                assert item not in actual_value
        else:
            for item in forbidden_values:
                assert item != actual_value


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["id"])
def test_requirement_parser_cases(case, tmp_path):
    output_path = tmp_path / "workspace" / "user_requirement.json"

    result = parse_and_save_requirement(
        case["input"],
        _fake_model_for(case["input"]),
        output_path=output_path,
    )

    saved_text = output_path.read_text(encoding="utf-8")
    saved_data = json.loads(saved_text)
    saved_requirement = PlotRequirement.model_validate(saved_data)

    assert saved_requirement == result
    assert result.original_request == case["input"]
    assert "\\u" not in saved_text
    assert not (STYLE_WORDS & set(result.functional_keywords))

    _assert_expected_subset(result, case["expected"])
    _assert_forbidden_absent(result, case.get("forbidden", {}))

    if result.is_ambiguous:
        assert result.clarification_question
    else:
        assert result.clarification_question == ""


def test_requirement_test_case_count():
    assert len(_load_cases()) >= 10

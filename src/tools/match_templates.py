"""Tool: deterministic template scoring."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas import PlotRequirement
from src.schemas import MatchTemplatesInput, TemplateEntry, TemplateRecommendation
from src.tools._utils import any_text_matches, error_result, text_matches, unique_non_empty


FUNCTIONAL_WEIGHT = 0.42
CHART_TYPE_WEIGHT = 0.26
SYNONYM_WEIGHT = 0.16
GOAL_WEIGHT = 0.08
USE_CASE_WEIGHT = 0.05
STYLE_WEIGHT = 0.03
NEGATIVE_PENALTY = 0.45

FUNCTIONAL_ALIASES = {
    "趋势": ["时间趋势", "时间序列", "年度变化", "trend_time_series"],
    "比较": ["对比", "分组比较", "多组比较", "comparison_ranking"],
    "排名": ["排序", "重要性排序", "ranking"],
    "分布": ["分布曲线", "箱线图", "小提琴图", "distribution_uncertainty"],
    "不确定性": ["误差", "误差棒", "置信区间", "残差"],
    "显著性": ["显著性标记", "p值", "组间差异", "检验"],
    "相关性": ["相关", "相关系数", "相关性热图", "relationship_correlation"],
    "回归": ["线性回归", "回归拟合", "回归预测", "r2", "rmse"],
    "分类评估": ["roc", "auc", "二分类", "多分类", "classification"],
    "预测评估": ["预测", "预测实测", "回归评估", "模型评估", "prediction_evaluation"],
    "敏感性": ["敏感性分析", "阈值效应", "响应面", "sensitivity_robustness"],
    "模型解释": ["shap", "pdp", "ale", "特征重要性", "特征贡献", "模型解释"],
    "聚类降维": ["pca", "pcoa", "rda", "聚类", "降维", "clustering_reduction"],
    "组成占比": ["组成比例", "百分比", "占比", "贡献比例"],
    "网络关系": ["网络图", "和弦图", "韦恩图", "network_flow"],
    "空间分析": ["栅格", "遥感", "空间", "spatial_geographic"],
    "优化决策": ["参数选择", "最佳参数", "optimization_decision"],
    "多面板": ["组合图", "综合报告", "多图拼版", "multi_panel_report"],
    "三维": ["3d", "三维", "立体"],
    "环形径向": ["环形", "环状", "极坐标", "玫瑰图"],
}


def _contains_negative_requirement(
    requirement: PlotRequirement, template: TemplateEntry
) -> list[str]:
    """检测用户明确排除项和模板是否冲突。"""

    conflicts: list[str] = []
    haystack = [
        template.name,
        template.chart_type,
        template.category,
        template.description,
        *template.keywords,
        *template.synonyms,
    ]
    for negative in requirement.negative_requirements:
        term = (
            negative.replace("不要", "")
            .replace("别用", "")
            .replace("不用", "")
            .replace("效果", "")
            .strip()
        )
        if term and any_text_matches(term, haystack):
            conflicts.append(negative)
    return unique_non_empty(conflicts)


def _recommendation_reason(item: TemplateRecommendation) -> str:
    reasons: list[str] = []
    if item.matched_chart_types:
        reasons.append(f"图表类型匹配：{', '.join(item.matched_chart_types)}")
    if item.matched_keywords:
        reasons.append(f"功能关键词匹配：{', '.join(item.matched_keywords)}")
    if item.matched_synonyms:
        reasons.append(f"原始表达匹配模板同义词：{', '.join(item.matched_synonyms)}")
    if item.matched_use_cases:
        reasons.append(f"使用场景匹配：{', '.join(item.matched_use_cases)}")
    if item.matched_styles:
        reasons.append(f"风格弱匹配：{', '.join(item.matched_styles)}")
    if item.penalties:
        reasons.append(f"扣分项：{', '.join(item.penalties)}")
    return "；".join(reasons) if reasons else "模板与需求存在弱相关。"


def _score_template(
    requirement: PlotRequirement,
    template: TemplateEntry,
    original_order: int,
) -> TemplateRecommendation | None:
    """计算单个模板的归一化得分。"""

    conflicts = _contains_negative_requirement(requirement, template)
    if conflicts:
        # 用户明确排除某类图表时直接排除，不进入候选集。
        return None

    matched_keywords: list[str] = []
    matched_synonyms: list[str] = []
    matched_chart_types: list[str] = []
    matched_use_cases: list[str] = []
    matched_styles: list[str] = []
    penalties: list[str] = []

    keyword_texts = [
        *template.keywords,
        template.category,
        template.description,
        *template.use_cases,
    ]
    for keyword in requirement.functional_keywords:
        variants = [keyword, *FUNCTIONAL_ALIASES.get(keyword, [])]
        if any(any_text_matches(variant, keyword_texts) for variant in variants):
            matched_keywords.append(keyword)

    if requirement.chart_types:
        for chart_type in requirement.chart_types:
            if text_matches(chart_type, template.chart_type) or any_text_matches(
                chart_type,
                [template.name, *template.synonyms],
            ):
                matched_chart_types.append(chart_type)

    source_texts = [requirement.original_request, requirement.goal]
    for synonym in template.synonyms:
        if any_text_matches(synonym, source_texts):
            matched_synonyms.append(synonym)

    goal_targets = [template.description, *template.use_cases]
    if requirement.goal and any_text_matches(requirement.goal, goal_targets):
        matched_use_cases.append(requirement.goal)
    if requirement.use_case and any_text_matches(requirement.use_case, template.use_cases):
        matched_use_cases.append(requirement.use_case)

    # catalog 暂无显式 style 字段，因此只做非常低权重的弱匹配。
    if requirement.style_keywords and (
        template.category == "multi_panel_report"
        or any_text_matches("报告", [template.name, template.description, *template.use_cases])
    ):
        matched_styles.extend(requirement.style_keywords)

    for negative_keyword in template.negative_keywords:
        if any_text_matches(negative_keyword, [requirement.original_request, requirement.goal]):
            penalties.append(f"命中模板负面关键词：{negative_keyword}")

    if penalties:
        # 模板自己声明不适合该需求时直接排除，避免负面命中仍靠其它弱信号进入候选集。
        return None

    functional_score = min(
        1.0, len(unique_non_empty(matched_keywords)) / max(1, len(requirement.functional_keywords))
    )
    chart_score = (
        min(1.0, len(unique_non_empty(matched_chart_types)) / len(requirement.chart_types))
        if requirement.chart_types
        else 0.0
    )
    synonym_score = min(
        1.0, len(unique_non_empty(matched_synonyms)) / max(1, len(template.synonyms))
    )
    goal_score = 1.0 if requirement.goal and requirement.goal in matched_use_cases else 0.0
    use_case_score = (
        1.0 if requirement.use_case and requirement.use_case in matched_use_cases else 0.0
    )
    style_score = 1.0 if matched_styles else 0.0
    penalty_score = min(0.8, len(penalties) * NEGATIVE_PENALTY)

    score = (
        functional_score * FUNCTIONAL_WEIGHT
        + chart_score * CHART_TYPE_WEIGHT
        + synonym_score * SYNONYM_WEIGHT
        + goal_score * GOAL_WEIGHT
        + use_case_score * USE_CASE_WEIGHT
        + style_score * STYLE_WEIGHT
        - penalty_score
    )
    score = max(0.0, min(1.0, round(score, 4)))

    recommendation = TemplateRecommendation(
        template_id=template.id,
        template_name=template.name,
        category=template.category,
        score=score,
        matched_keywords=unique_non_empty(matched_keywords),
        matched_synonyms=unique_non_empty(matched_synonyms),
        matched_chart_types=unique_non_empty(matched_chart_types),
        matched_use_cases=unique_non_empty(matched_use_cases),
        matched_styles=unique_non_empty(matched_styles),
        penalties=unique_non_empty(penalties),
        preview=template.preview,
        code_path=template.code_path,
        original_order=original_order,
    )
    recommendation.recommendation_reason = _recommendation_reason(recommendation)
    return recommendation


@tool(
    "match_templates",
    args_schema=MatchTemplatesInput,
    description=(
        "接收结构化用户需求和模板列表，按确定性规则计算匹配分数，"
        "返回达到最低分数的候选模板；不负责最终排序和保存。"
    ),
)
def match_templates(user_requirement: dict, templates: list[dict], min_score: float = 0.15) -> dict:
    """为每个模板计算匹配分数，并返回候选模板。"""

    try:
        parsed_requirement = PlotRequirement.model_validate(user_requirement)
        parsed_templates = [TemplateEntry.model_validate(item) for item in templates]
        MatchTemplatesInput(
            user_requirement=parsed_requirement.model_dump(),
            templates=[item.model_dump() for item in parsed_templates],
            min_score=min_score,
        )
    except ValidationError as error:
        return error_result(
            "validation_error",
            "match_templates 输入参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查 user_requirement、templates、min_score 是否符合 Schema。",
        )

    candidates: list[TemplateRecommendation] = []
    for index, template in enumerate(parsed_templates):
        candidate = _score_template(parsed_requirement, template, index)
        if candidate and candidate.score >= min_score:
            candidates.append(candidate)

    if not candidates:
        return error_result(
            "no_qualified_templates",
            "没有模板达到最低分数。",
            details={"total_templates": len(parsed_templates), "min_score": min_score},
            suggestion="请降低 min_score，或补充更明确的功能关键词、图表类型或使用场景。",
        )

    return {
        "success": True,
        "original_request": parsed_requirement.original_request,
        "total_templates": len(parsed_templates),
        "qualified_templates": len(candidates),
        "candidates": [item.model_dump() for item in candidates],
        "warnings": [],
    }

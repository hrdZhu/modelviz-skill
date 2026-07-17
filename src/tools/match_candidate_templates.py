"""Tool: 阶段三候选模板粗筛。"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import ValidationError

from schemas import PlotRequirement
from src.schemas import CandidateTemplate, MatchCandidateTemplatesInput, TemplateEntry
from src.tools._utils import any_text_matches, error_result, text_matches, unique_non_empty


FUNCTIONAL_WEIGHT = 0.42
CHART_TYPE_WEIGHT = 0.24
SYNONYM_WEIGHT = 0.15
GOAL_WEIGHT = 0.08
USE_CASE_WEIGHT = 0.08
STYLE_WEIGHT = 0.03
CATEGORY_ALIAS_WEIGHT = 0.06
NEGATIVE_PENALTY = 0.25

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


def _negative_conflicts(requirement: PlotRequirement, template: TemplateEntry) -> list[str]:
    """用户明确排除某类图表时，该模板不得进入候选。"""

    haystack = [
        template.name,
        template.chart_type,
        template.category,
        template.description,
        *template.keywords,
        *template.synonyms,
    ]
    conflicts: list[str] = []
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


def _candidate_reason(candidate: CandidateTemplate) -> str:
    parts: list[str] = []
    if candidate.matched_chart_types:
        parts.append(f"图表类型匹配：{', '.join(candidate.matched_chart_types)}")
    if candidate.matched_keywords:
        parts.append(f"功能关键词匹配：{', '.join(candidate.matched_keywords)}")
    if candidate.matched_synonyms:
        parts.append(f"原始表达匹配：{', '.join(candidate.matched_synonyms)}")
    if candidate.matched_use_cases:
        parts.append(f"使用场景匹配：{', '.join(candidate.matched_use_cases)}")
    if candidate.matched_styles:
        parts.append(f"风格弱匹配：{', '.join(candidate.matched_styles)}")
    if candidate.penalties:
        parts.append(f"扣分项：{', '.join(candidate.penalties)}")
    return "；".join(parts) if parts else "与用户需求存在弱相关。"


def _score_candidate(
    requirement: PlotRequirement,
    template: TemplateEntry,
    catalog_order: int,
) -> CandidateTemplate | None:
    if _negative_conflicts(requirement, template):
        return None

    matched_keywords: list[str] = []
    matched_synonyms: list[str] = []
    matched_chart_types: list[str] = []
    matched_use_cases: list[str] = []
    matched_styles: list[str] = []
    penalties: list[str] = []

    template_texts = [
        *template.keywords,
        template.category,
        template.description,
        *template.use_cases,
    ]
    for keyword in requirement.functional_keywords:
        variants = [keyword, *FUNCTIONAL_ALIASES.get(keyword, [])]
        if any(any_text_matches(variant, template_texts) for variant in variants):
            matched_keywords.append(keyword)

    for chart_type in requirement.chart_types:
        if text_matches(chart_type, template.chart_type) or any_text_matches(
            chart_type, [template.name, *template.synonyms]
        ):
            matched_chart_types.append(chart_type)

    for synonym in template.synonyms:
        if any_text_matches(synonym, [requirement.original_request, requirement.goal]):
            matched_synonyms.append(synonym)

    goal_targets = [template.description, *template.use_cases]
    if requirement.goal and any_text_matches(requirement.goal, goal_targets):
        matched_use_cases.append(requirement.goal)
    if requirement.use_case and any_text_matches(requirement.use_case, template.use_cases):
        matched_use_cases.append(requirement.use_case)

    if requirement.style_keywords and (
        template.category == "multi_panel_report"
        or any_text_matches("报告", [template.name, template.description, *template.use_cases])
    ):
        matched_styles.extend(requirement.style_keywords)

    for negative_keyword in template.negative_keywords:
        if any_text_matches(negative_keyword, [requirement.original_request, requirement.goal]):
            penalties.append(f"命中模板负面关键词：{negative_keyword}")

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
    category_alias_score = (
        1.0
        if any(
            template.category in [keyword, *FUNCTIONAL_ALIASES.get(keyword, [])]
            for keyword in requirement.functional_keywords
        )
        else 0.0
    )
    penalty_score = min(0.7, len(penalties) * NEGATIVE_PENALTY)
    score = (
        functional_score * FUNCTIONAL_WEIGHT
        + chart_score * CHART_TYPE_WEIGHT
        + synonym_score * SYNONYM_WEIGHT
        + goal_score * GOAL_WEIGHT
        + use_case_score * USE_CASE_WEIGHT
        + style_score * STYLE_WEIGHT
        + category_alias_score * CATEGORY_ALIAS_WEIGHT
        - penalty_score
    )
    score = max(0.0, min(1.0, round(score, 4)))

    candidate = CandidateTemplate(
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
        catalog_order=catalog_order,
    )
    candidate.candidate_reason = _candidate_reason(candidate)
    return candidate


@tool(
    "match_candidate_templates",
    args_schema=MatchCandidateTemplatesInput,
    description=(
        "阶段三候选召回工具：接收结构化需求和模板列表，计算每个模板的初步匹配分数，"
        "返回达到 min_score 的候选结果；不读取用户数据，不做最终模板判断。"
    ),
)
def match_candidate_templates(
    user_requirement: dict,
    templates: list[dict],
    min_score: float = 0.10,
) -> dict:
    """根据用户需求对模板目录做确定性粗筛。"""

    try:
        parsed_requirement = PlotRequirement.model_validate(user_requirement)
        parsed_templates = [TemplateEntry.model_validate(item) for item in templates]
        MatchCandidateTemplatesInput(
            user_requirement=parsed_requirement.model_dump(),
            templates=[item.model_dump() for item in parsed_templates],
            min_score=min_score,
        )
    except ValidationError as error:
        return error_result(
            "validation_error",
            "match_candidate_templates 输入参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查用户需求、模板列表和 min_score。",
        )

    candidates: list[CandidateTemplate] = []
    for index, template in enumerate(parsed_templates):
        candidate = _score_candidate(parsed_requirement, template, index)
        if candidate and candidate.score >= min_score:
            candidates.append(candidate)

    if not candidates:
        return error_result(
            "no_qualified_templates",
            "没有模板达到最低候选分数。",
            details={"total_templates": len(parsed_templates), "min_score": min_score},
            suggestion="请降低 min_score，或补充更明确的绘图需求。",
        )

    return {
        "success": True,
        "original_request": parsed_requirement.original_request,
        "total_templates": len(parsed_templates),
        "candidate_count": len(candidates),
        "candidates": [item.model_dump() for item in candidates],
        "warnings": [],
    }

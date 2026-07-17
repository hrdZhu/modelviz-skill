"""Tool: rank candidate templates."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import RankTemplatesInput, TemplateRecommendation, TemplateRecommendationOutput
from src.tools._utils import error_result


@tool(
    "rank_templates",
    args_schema=RankTemplatesInput,
    description=(
        "接收候选模板，按分数从高到低排序并返回 Top-K；分数相同时按原目录顺序和模板 ID 稳定排序。"
    ),
)
def rank_templates(candidates: list[dict], top_k: int = 5) -> dict:
    """排序候选模板并截取 Top-K。"""

    try:
        RankTemplatesInput(candidates=candidates, top_k=top_k)
        parsed_candidates = [TemplateRecommendation.model_validate(item) for item in candidates]
    except ValidationError as error:
        return error_result(
            "validation_error",
            "rank_templates 输入参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请确认 top_k 大于 0 且不超过 20，候选模板符合推荐结果 Schema。",
        )

    sorted_candidates = sorted(
        parsed_candidates,
        key=lambda item: (-item.score, item.original_order, item.template_id),
    )
    recommendations = sorted_candidates[:top_k]
    output = TemplateRecommendationOutput(
        original_request="",
        total_templates=0,
        qualified_templates=len(parsed_candidates),
        recommendations=recommendations,
        warnings=[] if parsed_candidates else ["没有模板达到最低分数。"],
    )
    return {"success": True, "ranked_result": output.model_dump()}

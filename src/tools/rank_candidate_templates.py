"""Tool: 阶段三候选模板稳定排序。"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import CandidateTemplate, CandidateTemplateOutput, RankCandidateTemplatesInput
from src.tools._utils import error_result


@tool(
    "rank_candidate_templates",
    args_schema=RankCandidateTemplatesInput,
    description="阶段三候选排序工具：按分数降序、目录顺序和模板 ID 稳定排序，返回 5 到 10 个候选。",
)
def rank_candidate_templates(candidates: list[dict], top_k: int = 8) -> dict:
    """对候选模板排序并截取 Top-K。"""

    try:
        RankCandidateTemplatesInput(candidates=candidates, top_k=top_k)
        parsed = [CandidateTemplate.model_validate(item) for item in candidates]
    except ValidationError as error:
        return error_result(
            "validation_error",
            "rank_candidate_templates 输入参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请确认 top_k 在 5 到 10 之间，候选结果结构正确。",
        )

    ranked = sorted(parsed, key=lambda item: (-item.score, item.catalog_order, item.template_id))
    output = CandidateTemplateOutput(
        original_request="",
        candidate_count=min(top_k, len(ranked)),
        total_templates=0,
        candidates=ranked[:top_k],
        warnings=[],
    )
    return {"success": True, "candidate_result": output.model_dump()}

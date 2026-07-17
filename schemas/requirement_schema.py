"""
用户绘图需求的结构化表示。
这个 Schema 只描述“用户想画什么”，用于后续模板检索。
它不负责模板匹配、推荐排序或实际绘图。
"""

from typing import Annotated  # 类型增强工具(给类型附加额外信息)
from pydantic import BaseModel, ConfigDict, Field

ShortText = Annotated[str, Field(max_length=120)]
Keyword = Annotated[str, Field(min_length=1, max_length=60)]


class PlotRequirement(BaseModel):
    """从自然语言绘图需求中抽取出的模板检索条件。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    original_request: Annotated[str, Field(min_length=1, max_length=2000)] = Field(
        ...,
        description="用户输入的原始绘图需求，不做改写。",
    )
    goal: ShortText = Field(
        "",
        description="主要绘图或分析目标，例如：比较模型性能、展示时间趋势。",
    )
    functional_keywords: list[Keyword] = Field(
        default_factory=list,
        description="功能关键词，如趋势、比较、误差、敏感性。",
    )
    chart_types: list[Keyword] = Field(
        default_factory=list,
        description="明确提出或可以可靠确定的图表类型；无法确定时保持空列表。",
    )
    style_keywords: list[Keyword] = Field(
        default_factory=list,
        description="风格要求，如科研风、简洁、低饱和。",
    )
    use_case: ShortText = Field(
        "",
        description="使用场景，如论文正文、模型评价、答辩展示。",
    )
    negative_requirements: list[Keyword] = Field(
        default_factory=list,
        description="用户明确排除的图表或效果，如不要三维、不要饼图。",
    )
    explicit_template: bool = Field(
        False,
        description="用户是否明确指定了图表类型或模板名称。",
    )
    is_ambiguous: bool = Field(
        False,
        description="需求是否过于模糊，导致无法形成稳定检索条件。",
    )
    clarification_question: str = Field(
        "",
        max_length=300,
        description="需求模糊时建议追问用户的问题；不模糊时为空字符串。",
    )

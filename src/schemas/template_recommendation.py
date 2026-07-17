"""阶段三模板推荐相关 Pydantic 结构。

这些 Schema 约束工具输入、模板目录条目、候选推荐和最终推荐结果。
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


PathText = Annotated[str, Field(min_length=1, max_length=500)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
TopK = Annotated[int, Field(gt=0, le=20)]


class TemplateMatchingError(BaseModel):
    """统一的工具失败结果，避免向用户暴露 Python 堆栈。"""

    success: bool = False
    error_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    suggestion: str = ""


class LoadUserRequirementInput(BaseModel):
    """读取用户需求文件的工具参数。"""

    requirement_path: PathText = Field(
        "workspace/user_requirement.json",
        description="用户需求 JSON 文件路径，默认读取 workspace/user_requirement.json。",
    )


class LoadTemplateCatalogInput(BaseModel):
    """读取模板目录 YAML 的工具参数。"""

    catalog_path: PathText = Field(
        "docs/template_catalog.yaml",
        description="模板目录 YAML 文件路径，默认读取 docs/template_catalog.yaml。",
    )


class MatchTemplatesInput(BaseModel):
    """模板匹配工具参数。"""

    user_requirement: dict[str, Any] = Field(
        ...,
        description="已通过 PlotRequirement 校验的用户需求结构。",
    )
    templates: list[dict[str, Any]] = Field(
        ...,
        description="已通过 TemplateEntry 校验的模板列表。",
    )
    min_score: Score = Field(
        0.15,
        description="候选模板最低归一化分数，取值范围 0 到 1。",
    )


class RankTemplatesInput(BaseModel):
    """候选模板排序工具参数。"""

    candidates: list[dict[str, Any]] = Field(
        ...,
        description="match_templates 产出的候选模板列表。",
    )
    top_k: TopK = Field(
        5,
        description="返回推荐数量，必须大于 0，最大为 20。",
    )


class SaveTemplateRecommendationsInput(BaseModel):
    """保存推荐结果工具参数。"""

    recommendation_result: dict[str, Any] = Field(
        ...,
        description="已排序的推荐结果，保存前会再次用 TemplateRecommendationOutput 校验。",
    )
    output_path: PathText = Field(
        "workspace/template_recommendations.json",
        description="推荐结果 JSON 输出路径，默认写入 workspace/template_recommendations.json。",
    )


class PipelineInput(BaseModel):
    """阶段三固定流程入口参数。"""

    requirement_path: PathText = Field(
        "workspace/user_requirement.json",
        description="用户需求 JSON 文件路径。",
    )
    catalog_path: PathText = Field(
        "docs/template_catalog.yaml",
        description="模板目录 YAML 文件路径。",
    )
    top_k: TopK = Field(
        5,
        description="最终推荐数量，必须大于 0，最大为 20。",
    )
    min_score: Score = Field(
        0.15,
        description="进入候选集的最低归一化分数，取值范围 0 到 1。",
    )
    output_path: PathText = Field(
        "workspace/template_recommendations.json",
        description="推荐结果 JSON 输出路径。",
    )


class TemplateEntry(BaseModel):
    """template_catalog.yaml 中的单个模板条目。"""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    id: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)]
    category: Annotated[str, Field(min_length=1)]
    description: str = ""
    keywords: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    negative_keywords: list[str] = Field(default_factory=list)
    chart_type: Annotated[str, Field(min_length=1)]
    preview: str = ""
    code_path: str = ""


class TemplateCatalog(BaseModel):
    """已校验的模板目录。"""

    templates: list[TemplateEntry] = Field(min_length=1)


class TemplateRecommendation(BaseModel):
    """单个模板推荐结果。"""

    template_id: str
    template_name: str
    category: str
    score: Score
    matched_keywords: list[str] = Field(default_factory=list)
    matched_synonyms: list[str] = Field(default_factory=list)
    matched_chart_types: list[str] = Field(default_factory=list)
    matched_use_cases: list[str] = Field(default_factory=list)
    matched_styles: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    recommendation_reason: str = ""
    preview: str = ""
    code_path: str = ""
    original_order: int = Field(0, description="原始目录顺序，用于稳定排序。")


class TemplateRecommendationOutput(BaseModel):
    """最终推荐结果文件结构。"""

    original_request: str
    total_templates: int = Field(ge=0)
    qualified_templates: int = Field(ge=0)
    recommendations: list[TemplateRecommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

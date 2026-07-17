"""阶段三候选模板粗筛结构。"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


PathText = Annotated[str, Field(min_length=1, max_length=500)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
CandidateTopK = Annotated[int, Field(ge=5, le=10)]


class ToolError(BaseModel):
    """统一结构化错误结果。"""

    success: bool = False
    error_type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    suggestion: str = ""


class CandidatePipelineInput(BaseModel):
    """阶段三候选召回入口参数。"""

    requirement_path: PathText = Field(
        "workspace/user_requirement.json",
        description="阶段二生成的标准化用户需求 JSON 路径。",
    )
    catalog_path: PathText = Field(
        "docs/template_catalog.yaml",
        description="模板目录 YAML 路径。",
    )
    output_path: PathText = Field(
        "workspace/candidate_templates.json",
        description="候选模板 JSON 输出路径。",
    )
    top_k: CandidateTopK = Field(
        8,
        description="候选模板数量，限制在 5 到 10。",
    )
    min_score: Score = Field(
        0.10,
        description="进入候选集的最低分数，范围 0 到 1。",
    )


class MatchCandidateTemplatesInput(BaseModel):
    """候选匹配工具参数。"""

    user_requirement: dict[str, Any] = Field(..., description="PlotRequirement 字典。")
    templates: list[dict[str, Any]] = Field(..., description="模板目录条目列表。")
    min_score: Score = Field(0.10, description="候选最低分数，范围 0 到 1。")


class RankCandidateTemplatesInput(BaseModel):
    """候选排序工具参数。"""

    candidates: list[dict[str, Any]] = Field(..., description="候选模板列表。")
    top_k: CandidateTopK = Field(8, description="返回 5 到 10 个候选模板。")


class SaveCandidateTemplatesInput(BaseModel):
    """候选保存工具参数。"""

    candidate_result: dict[str, Any] = Field(..., description="CandidateTemplateOutput 字典。")
    output_path: PathText = Field(
        "workspace/candidate_templates.json",
        description="候选模板 JSON 保存路径。",
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
    """已校验模板目录。"""

    templates: list[TemplateEntry] = Field(min_length=1)


class CandidateTemplate(BaseModel):
    """单个阶段三候选模板。"""

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
    candidate_reason: str = ""
    preview: str = ""
    code_path: str = ""
    catalog_order: int = Field(0, ge=0)


class CandidateTemplateOutput(BaseModel):
    """workspace/candidate_templates.json 的结构。"""

    original_request: str
    candidate_count: int = Field(ge=0)
    total_templates: int = Field(ge=0)
    candidates: list[CandidateTemplate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

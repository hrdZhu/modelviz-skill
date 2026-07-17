"""阶段四最终模板选择结构。"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from .candidate_template import PathText


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class CandidateComparison(BaseModel):
    """模型对单个候选模板的简洁比较。"""

    template_id: str
    suitable: bool
    advantages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    data_compatibility: str = ""
    requirement_compatibility: str = ""


class FinalTemplateSelection(BaseModel):
    """阶段四结构化输出。"""

    dataset_summary: str = ""
    observed_data_features: list[str] = Field(default_factory=list)
    relevant_columns: list[str] = Field(default_factory=list)
    candidate_comparisons: list[CandidateComparison] = Field(default_factory=list)
    selected_template_id: str | None = None
    selected_template_name: str | None = None
    alternative_template_ids: list[str] = Field(default_factory=list)
    selection_reason: str = ""
    data_support_reason: str = ""
    data_warnings: list[str] = Field(default_factory=list)
    confidence: Confidence = 0.0
    needs_clarification: bool = False
    clarification_question: str = ""

    @model_validator(mode="after")
    def _check_clarification(self):
        if self.needs_clarification and not self.clarification_question.strip():
            raise ValueError("needs_clarification=true 时必须提供 clarification_question")
        if self.selected_template_id and not self.selected_template_name:
            raise ValueError("选中模板时 selected_template_name 不能为空")
        if not self.selected_template_id and self.selected_template_name:
            raise ValueError("没有选中模板时 selected_template_name 必须为空")
        return self


class FinalSelectionPipelineInput(BaseModel):
    """阶段四 pipeline 参数。"""

    requirement_path: PathText = Field(
        "workspace/user_requirement.json", description="用户需求 JSON。"
    )
    candidate_path: PathText = Field(
        "workspace/candidate_templates.json",
        description="阶段三候选模板 JSON。",
    )
    data_path: PathText = Field(..., description="用户 CSV/XLSX/XLS 数据文件。")
    output_path: PathText = Field(
        "workspace/final_template_selection.json",
        description="最终模板选择 JSON 输出路径。",
    )
    dataset_context_path: PathText = Field(
        "workspace/dataset_context.json",
        description="数据上下文 JSON 输出路径。",
    )
    sheet_name: str | int | None = None
    max_sample_rows: Annotated[int, Field(gt=0, le=100)] = 20


class FinalSelectionValidationContext(BaseModel):
    """普通 Python 二次校验需要的上下文。"""

    candidate_ids_to_names: dict[str, str]
    column_names: list[str]


def validate_final_selection_references(
    selection: FinalTemplateSelection,
    context: FinalSelectionValidationContext,
) -> FinalTemplateSelection:
    """校验模型不能编造候选模板 ID 或数据列。"""

    candidate_ids = set(context.candidate_ids_to_names)
    columns = set(context.column_names)

    if selection.selected_template_id:
        if selection.selected_template_id not in candidate_ids:
            raise ValueError(f"模型返回候选列表外的模板 ID：{selection.selected_template_id}")
        expected_name = context.candidate_ids_to_names[selection.selected_template_id]
        if selection.selected_template_name != expected_name:
            raise ValueError("选中模板名称与模板 ID 不一致")

    unique_alternatives: list[str] = []
    for template_id in selection.alternative_template_ids:
        if template_id not in candidate_ids:
            raise ValueError(f"备选模板 ID 不在候选列表中：{template_id}")
        if template_id != selection.selected_template_id and template_id not in unique_alternatives:
            unique_alternatives.append(template_id)

    unique_columns: list[str] = []
    for column in selection.relevant_columns:
        if column not in columns:
            raise ValueError(f"模型返回不存在的数据列：{column}")
        if column not in unique_columns:
            unique_columns.append(column)

    selection.alternative_template_ids = unique_alternatives
    selection.relevant_columns = unique_columns
    return selection

"""模板适配计划结构。"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from .candidate_template import PathText


class ColumnMapping(BaseModel):
    """真实数据列到模板角色的映射。"""

    data_column: str
    template_role: str
    reason: str = ""


class AdaptationPlan(BaseModel):
    """大模型生成的模板适配计划。"""

    template_id: str
    template_name: str
    plot_goal: str = ""
    selected_columns: list[str] = Field(default_factory=list)
    column_mappings: list[ColumnMapping] = Field(default_factory=list)
    required_preprocessing: list[str] = Field(default_factory=list)
    required_dependencies: list[str] = Field(default_factory=list)
    layout_elements_to_preserve: list[str] = Field(default_factory=list)
    style_elements_to_preserve: list[str] = Field(default_factory=list)
    elements_allowed_to_change: list[str] = Field(default_factory=list)
    title_plan: str = ""
    axis_plan: str = ""
    legend_plan: str = ""
    annotation_plan: str = ""
    output_formats: list[str] = Field(default_factory=lambda: ["png", "svg", "pdf"])
    warnings: list[str] = Field(default_factory=list)
    can_proceed: bool = True
    clarification_question: str = ""

    @model_validator(mode="after")
    def _check_stop_reason(self):
        if not self.can_proceed and not (self.warnings or self.clarification_question.strip()):
            raise ValueError("can_proceed=false 时必须提供 warnings 或 clarification_question")
        return self


class AdaptationPlanPipelineInput(BaseModel):
    """适配计划保存参数。"""

    output_path: PathText = Field(
        "workspace/adaptation_plan.json", description="适配计划保存路径。"
    )


class TemplateAdaptationPipelineInput(BaseModel):
    """阶段五模板适配入口参数。"""

    final_selection_path: PathText = Field(
        "workspace/final_template_selection.json", description="最终模板选择结果。"
    )
    requirement_path: PathText = Field(
        "workspace/user_requirement.json", description="用户需求 JSON。"
    )
    dataset_context_path: PathText = Field(
        "workspace/dataset_context.json", description="数据上下文 JSON。"
    )
    catalog_path: PathText = Field("docs/template_catalog.yaml", description="模板目录 YAML。")
    data_path: PathText = Field(..., description="用户原始数据文件路径。")
    workspace_dir: PathText = Field("workspace", description="阶段五输出目录。")
    outputs_dir: PathText = Field("outputs", description="图表输出目录。")
    python_executable: str = Field("", description="Python 解释器路径；为空时使用当前解释器。")
    auto_install: bool = Field(True, description="是否自动安装缺失依赖。")
    timeout_seconds: Annotated[int, Field(gt=0, le=600)] = 120

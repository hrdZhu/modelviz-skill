"""模板适配代码生成结果。"""

from pydantic import BaseModel, Field, model_validator


class AdaptationResult(BaseModel):
    """大模型返回的完整适配代码和变更说明。"""

    adapted_code: str
    changes_summary: list[str] = Field(default_factory=list)
    preserved_style_elements: list[str] = Field(default_factory=list)
    changed_elements: list[str] = Field(default_factory=list)
    data_columns_used: list[str] = Field(default_factory=list)
    dependencies_used: list[str] = Field(default_factory=list)
    additional_dependencies_requested: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _code_not_empty(self):
        if not self.adapted_code.strip():
            raise ValueError("adapted_code 不得为空")
        return self


class SaveAdaptedScriptInput(BaseModel):
    """保存适配脚本参数。"""

    adapted_code: str = Field(..., description="完整 Python 绘图代码。")
    script_path: str = Field("workspace/adapted_plot.py", description="适配脚本保存路径。")

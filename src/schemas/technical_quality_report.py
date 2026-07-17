"""程序化技术质量检查报告。"""

from pydantic import BaseModel, Field


class TechnicalIssue(BaseModel):
    """技术检查发现的问题。"""

    issue_type: str
    severity: str = "medium"
    message: str
    suggested_action: str = ""


class TechnicalQualityReport(BaseModel):
    """workspace/technical_quality_report.json 结构。"""

    passed: bool = False
    script_executed: bool = False
    dependencies_available: bool = True
    dependency_issues: list[str] = Field(default_factory=list)
    image_exists: bool = False
    image_readable: bool = False
    image_width: int = 0
    image_height: int = 0
    file_size: int = 0
    pixel_variance: float = 0.0
    possible_blank_image: bool = False
    generated_formats: list[str] = Field(default_factory=list)
    execution_errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[TechnicalIssue] = Field(default_factory=list)
    needs_repair: bool = True


class ValidateOutputArtifactsInput(BaseModel):
    generated_files: list[str] = Field(default_factory=list, description="脚本生成的文件列表。")
    output_directory: str = Field("outputs", description="允许输出目录。")


class InspectGeneratedImageInput(BaseModel):
    image_path: str = Field(..., description="待检查 PNG 图片路径。")


class CollectPlotWarningsInput(BaseModel):
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None

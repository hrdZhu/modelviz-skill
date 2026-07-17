"""大语言模型视觉质量检查报告。"""

from typing import Annotated

from pydantic import BaseModel, Field


class VisualQualityReport(BaseModel):
    passed: bool = False
    requirement_alignment: str = ""
    data_expression_quality: str = ""
    style_preservation: str = ""
    readability: str = ""
    layout_quality: str = ""
    color_quality: str = ""
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
    needs_repair: bool = True
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

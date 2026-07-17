"""阶段四提供给模型的数据上下文结构。"""

from typing import Annotated, Any

from pydantic import BaseModel, Field

from .candidate_template import PathText


class DatasetContextInput(BaseModel):
    """数据上下文准备参数。"""

    data_path: PathText = Field(..., description="用户数据文件路径，支持 CSV、XLSX、XLS。")
    output_path: PathText = Field(
        "workspace/dataset_context.json",
        description="数据上下文 JSON 输出路径。",
    )
    sheet_name: str | int | None = Field(
        None,
        description="Excel sheet 名称或索引；为空时读取第一个 sheet。",
    )
    max_sample_rows: Annotated[int, Field(gt=0, le=100)] = Field(
        20,
        description="进入模型上下文的最大样例行数。",
    )
    large_file_row_threshold: Annotated[int, Field(gt=0)] = Field(
        1000,
        description="超过该行数视为较大文件并进行抽样。",
    )


class DatasetContext(BaseModel):
    """只包含事实，不做语义分析或模板推荐。"""

    file_name: str
    file_type: str
    sheet_name: str | int | None = None
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    column_names: list[str] = Field(default_factory=list)
    sample_records: list[dict[str, Any]] = Field(default_factory=list)
    head_records: list[dict[str, Any]] = Field(default_factory=list)
    sampled: bool = False
    sample_strategy: str = ""
    warnings: list[str] = Field(default_factory=list)

"""绘图脚本执行结果结构。"""

from typing import Annotated

from pydantic import BaseModel, Field


class ExecutePlotScriptInput(BaseModel):
    """执行适配脚本参数。"""

    script_path: str = Field("workspace/adapted_plot.py", description="待执行脚本路径。")
    data_path: str = Field(..., description="用户原始数据路径。")
    output_directory: str = Field("outputs", description="图表输出目录。")
    timeout_seconds: Annotated[int, Field(gt=0, le=600)] = 120
    python_executable: str = Field("", description="Python 解释器路径；为空时使用当前解释器。")


class ExecutionResult(BaseModel):
    """workspace/execution_result.json 结构。"""

    success: bool
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    generated_files: list[str] = Field(default_factory=list)
    execution_time: float = 0.0
    error_type: str = ""

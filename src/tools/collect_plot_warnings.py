"""Tool: 从脚本执行输出中收集绘图警告和错误。"""

from __future__ import annotations

import re

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import CollectPlotWarningsInput, TechnicalIssue
from src.tools._utils import error_result


@tool(
    "collect_plot_warnings",
    args_schema=CollectPlotWarningsInput,
    description="分析 stdout/stderr，收集异常、依赖错误、字体、布局、图例和数据转换警告。",
)
def collect_plot_warnings(
    stdout: str = "", stderr: str = "", return_code: int | None = None
) -> dict:
    """把执行日志转换为结构化 issues。"""

    try:
        CollectPlotWarningsInput(stdout=stdout, stderr=stderr, return_code=return_code)
    except ValidationError as error:
        return error_result(
            "validation_error", "日志检查参数错误。", details={"errors": error.errors()}
        )
    text = f"{stdout}\n{stderr}"
    issues: list[TechnicalIssue] = []
    warnings: list[str] = []
    if return_code not in (None, 0):
        issues.append(
            TechnicalIssue(
                issue_type="runtime_error", severity="high", message=f"脚本返回码：{return_code}"
            )
        )
    patterns = {
        "missing_dependency": r"(ModuleNotFoundError|ImportError).*['\"]?([A-Za-z0-9_]+)",
        "font_warning": r"(Glyph .* missing|findfont|Font family)",
        "layout_warning": r"(tight_layout|constrained_layout|Layout)",
        "legend_warning": r"(No artists with labels found|legend)",
        "data_warning": r"(RuntimeWarning|FutureWarning|SettingWithCopyWarning|invalid value)",
        "save_failed": r"(PermissionError|No such file|failed to save|save failed|cannot save)",
    }
    for issue_type, pattern in patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            severity = "high" if issue_type in {"missing_dependency", "save_failed"} else "medium"
            issues.append(
                TechnicalIssue(
                    issue_type=issue_type, severity=severity, message=f"日志命中：{issue_type}"
                )
            )
            warnings.append(issue_type)
    return {
        "success": not any(issue.severity == "high" for issue in issues),
        "warnings": warnings,
        "issues": [i.model_dump() for i in issues],
    }

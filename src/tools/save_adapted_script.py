"""Tool: 保存大模型生成的适配脚本。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import SaveAdaptedScriptInput
from src.tools._utils import error_result


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@tool(
    "save_adapted_script",
    args_schema=SaveAdaptedScriptInput,
    description="保存 workspace/adapted_plot.py，禁止写入 templates/，不覆盖原始模板。",
)
def save_adapted_script(adapted_code: str, script_path: str = "workspace/adapted_plot.py") -> dict:
    """校验并保存完整 Python 代码。"""

    try:
        SaveAdaptedScriptInput(adapted_code=adapted_code, script_path=script_path)
    except ValidationError as error:
        return error_result(
            "validation_error", "适配脚本参数校验失败。", details={"errors": error.errors()}
        )
    if not adapted_code.strip():
        return error_result(
            "empty_code", "适配代码为空。", suggestion="请重新生成完整 Python 代码。"
        )
    path = Path(script_path)
    if _inside(path, Path("templates")):
        return error_result(
            "forbidden_path",
            "禁止把适配代码写入 templates/。",
            details={"script_path": str(path)},
            suggestion="请保存到 workspace/adapted_plot.py。",
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(adapted_code, encoding="utf-8")
    except OSError as error:
        return error_result("save_failed", "适配脚本保存失败。", details={"error": str(error)})
    return {"success": True, "script_path": str(path)}

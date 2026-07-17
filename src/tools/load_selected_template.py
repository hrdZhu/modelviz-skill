"""Tool: 读取最终选择的模板代码和元数据。"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from src.schemas import FinalTemplateSelection, TemplateEntry
from src.tools._utils import error_result


class LoadSelectedTemplateInput(BaseModel):
    final_selection_path: str = Field(
        "workspace/final_template_selection.json", description="最终模板选择结果。"
    )
    catalog_path: str = Field("docs/template_catalog.yaml", description="模板目录 YAML。")


@tool(
    "load_selected_template",
    args_schema=LoadSelectedTemplateInput,
    description="读取最终模板选择结果，查找模板元数据并读取模板源码；不修改原始模板。",
)
def load_selected_template(
    final_selection_path: str = "workspace/final_template_selection.json",
    catalog_path: str = "docs/template_catalog.yaml",
) -> dict:
    """返回选定模板路径、源码和元数据。"""

    try:
        selection_path = Path(final_selection_path)
        catalog_file = Path(catalog_path)
        if not selection_path.exists():
            return error_result(
                "file_not_found", "最终模板选择结果不存在。", details={"path": str(selection_path)}
            )
        if not catalog_file.exists():
            return error_result(
                "file_not_found", "模板目录不存在。", details={"path": str(catalog_file)}
            )
        selection = FinalTemplateSelection.model_validate(
            json.loads(selection_path.read_text(encoding="utf-8"))
        )
        if not selection.selected_template_id:
            return error_result(
                "no_selected_template",
                "最终选择结果没有选中模板。",
                suggestion="请先完成阶段四，并确保 selected_template_id 非空。",
            )
        catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8")) or {}
        templates = catalog.get("templates") or []
        matched = next(
            (item for item in templates if item.get("id") == selection.selected_template_id), None
        )
        if not matched:
            return error_result(
                "template_not_found",
                "模板目录中找不到选定模板。",
                details={"selected_template_id": selection.selected_template_id},
            )
        metadata = TemplateEntry.model_validate(matched)
        code_path = Path(metadata.code_path)
        if not code_path.exists():
            return error_result(
                "template_file_not_found",
                "选定模板源码不存在。",
                details={"code_path": str(code_path)},
            )
        return {
            "success": True,
            "selected_template": {
                "template_id": metadata.id,
                "template_name": metadata.name,
                "metadata": metadata.model_dump(),
                "template_code_path": str(code_path),
                "template_source_code": code_path.read_text(encoding="utf-8"),
                "preview": metadata.preview,
            },
        }
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        return error_result(
            "parse_error",
            "最终选择结果或模板目录格式错误。",
            details={"error": str(error)},
            suggestion="请检查 JSON/YAML 文件格式。",
        )
    except ValidationError as error:
        return error_result(
            "validation_error",
            "选定模板相关文件校验失败。",
            details={"errors": error.errors()},
        )

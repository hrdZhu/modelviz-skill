"""Tool: load and validate template catalog YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import LoadTemplateCatalogInput, TemplateCatalog, TemplateEntry
from src.tools._utils import error_result, path_for_error, validation_error_result


REQUIRED_TEMPLATE_FIELDS = {
    "id",
    "name",
    "category",
    "description",
    "keywords",
    "synonyms",
    "use_cases",
    "negative_keywords",
    "chart_type",
    "preview",
    "code_path",
}


@tool(
    "load_template_catalog",
    args_schema=LoadTemplateCatalogInput,
    description=(
        "读取并验证 docs/template_catalog.yaml。该工具只负责加载模板目录，不负责评分和排序。"
    ),
)
def load_template_catalog(catalog_path: str = "docs/template_catalog.yaml") -> dict:
    """读取模板目录 YAML，检查格式、空列表和必要字段。"""

    path = Path(catalog_path)
    if not path.exists():
        return error_result(
            "file_not_found",
            "模板 YAML 文件不存在。",
            details={"catalog_path": path_for_error(path)},
            suggestion="请确认 docs/template_catalog.yaml 路径正确。",
        )

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        return error_result(
            "yaml_decode_error",
            "模板 YAML 格式错误。",
            details={"catalog_path": path_for_error(path), "error": str(error)},
            suggestion="请检查 YAML 缩进、冒号和列表格式。",
        )

    templates = data.get("templates") if isinstance(data, dict) else None
    if not templates:
        return error_result(
            "empty_catalog",
            "模板目录为空。",
            details={"catalog_path": path_for_error(path)},
            suggestion="请在 template_catalog.yaml 中提供非空 templates 列表。",
        )

    for index, template in enumerate(templates):
        missing = sorted(REQUIRED_TEMPLATE_FIELDS - set(template or {}))
        if missing:
            return error_result(
                "missing_template_fields",
                "模板缺少必要字段。",
                details={"index": index, "missing_fields": missing},
                suggestion="请补齐模板 id、name、category、keywords、chart_type 等必要字段。",
            )

    try:
        catalog = TemplateCatalog(
            templates=[TemplateEntry.model_validate(item) for item in templates]
        )
    except ValidationError as error:
        return validation_error_result(error, suggestion="请按 TemplateEntry Schema 修正模板目录。")

    return {
        "success": True,
        "total_templates": len(catalog.templates),
        "templates": [item.model_dump() for item in catalog.templates],
    }

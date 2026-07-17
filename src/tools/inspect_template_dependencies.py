"""Tool: AST 识别模板依赖。"""

from __future__ import annotations

import ast
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import DependencyInspectionResult, DependencyItem, InspectTemplateDependenciesInput
from src.tools._utils import error_result, unique_non_empty
from src.tools.dependency_utils import (
    is_internal_module,
    is_stdlib_module,
    load_package_mapping,
    resolve_import_and_package,
)


def _imports_from_source(source: str) -> tuple[list[str], list[str]]:
    tree = ast.parse(source)
    imports: list[str] = []
    warnings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if node.module:
                imports.append(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            is_dunder = isinstance(node.func, ast.Name) and node.func.id == "__import__"
            is_importlib = (
                isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
            )
            if is_dunder or is_importlib:
                warnings.append("模板中存在动态导入，无法完全静态识别依赖。")
    return unique_non_empty(imports), unique_non_empty(warnings)


@tool(
    "inspect_template_dependencies",
    args_schema=InspectTemplateDependenciesInput,
    description="读取模板源码并用 AST 识别 import，合并 catalog dependencies，排除标准库和项目内部模块。",
)
def inspect_template_dependencies(
    template_code_path: str,
    catalog_dependencies: list[str] | None = None,
    package_mapping_path: str = "docs/package_name_mapping.yaml",
    project_root: str = ".",
) -> dict:
    """只负责识别依赖，不检查、不安装。"""

    try:
        InspectTemplateDependenciesInput(
            template_code_path=template_code_path,
            catalog_dependencies=catalog_dependencies or [],
            package_mapping_path=package_mapping_path,
            project_root=project_root,
        )
    except ValidationError as error:
        return error_result(
            "validation_error",
            "依赖识别参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查模板代码路径和包名映射路径。",
        )

    path = Path(template_code_path)
    if not path.exists():
        return error_result(
            "template_file_not_found",
            "模板文件不存在。",
            details={"template_code_path": str(path)},
            suggestion="请检查最终模板选择结果和模板目录中的路径。",
        )

    try:
        source_imports, warnings = _imports_from_source(path.read_text(encoding="utf-8"))
    except SyntaxError as error:
        return error_result(
            "template_parse_error",
            "模板源码无法被 Python AST 解析。",
            details={"template_code_path": str(path), "error": str(error)},
            suggestion="请检查模板是否为有效 Python 文件。",
        )

    mapping = load_package_mapping(package_mapping_path)
    required: list[DependencyItem] = []
    unresolved: list[str] = []
    stdlib: list[str] = []
    internal: list[str] = []
    seen: set[str] = set()
    raw_dependencies = unique_non_empty([*source_imports, *(catalog_dependencies or [])])

    for dependency in raw_dependencies:
        import_name, package_name = resolve_import_and_package(dependency, mapping)
        top_import = import_name.split(".")[0]
        if is_stdlib_module(top_import):
            stdlib.append(top_import)
            continue
        if is_internal_module(top_import, project_root):
            internal.append(top_import)
            continue
        if package_name is None:
            unresolved.append(dependency)
            continue
        key = f"{top_import}:{package_name}"
        if key not in seen:
            required.append(
                DependencyItem(
                    import_name=top_import,
                    package_name=package_name,
                    source=["ast" if dependency in source_imports else "catalog"],
                )
            )
            seen.add(key)

    result = DependencyInspectionResult(
        template_code_path=str(path),
        required_dependencies=required,
        unresolved_dependencies=unique_non_empty(unresolved),
        stdlib_dependencies=unique_non_empty(stdlib),
        internal_modules=unique_non_empty(internal),
        warnings=warnings,
    )
    return {"success": True, "dependency_inspection": result.model_dump()}

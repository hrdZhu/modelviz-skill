"""Tool: 检查 Python 依赖是否可导入。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import (
    CheckPythonDependenciesInput,
    DependencyItem,
    DependencyReport,
    InstalledDependency,
)
from src.tools._utils import error_result, unique_non_empty
from src.tools.dependency_utils import (
    dependency_available,
    dependency_version,
    is_stdlib_module,
    load_package_mapping,
    resolve_import_and_package,
)


def build_dependency_report(
    dependencies: list[str],
    python_executable: str = "",
    package_mapping_path: str = "docs/package_name_mapping.yaml",
) -> DependencyReport:
    """复用的依赖检查逻辑。"""

    mapping = load_package_mapping(package_mapping_path)
    installed: list[InstalledDependency] = []
    missing: list[DependencyItem] = []
    unresolved: list[str] = []
    required: list[str] = []
    warnings: list[str] = []

    for dependency in unique_non_empty(dependencies):
        import_name, package_name = resolve_import_and_package(dependency, mapping)
        if is_stdlib_module(import_name):
            continue
        if package_name is None:
            unresolved.append(dependency)
            continue
        available = dependency_available(import_name)
        version = dependency_version(package_name) if available else None
        required.append(package_name)
        installed.append(
            InstalledDependency(
                import_name=import_name,
                package_name=package_name,
                installed_version=version,
                available=available,
            )
        )
        if not available:
            missing.append(
                DependencyItem(import_name=import_name, package_name=package_name, source=["check"])
            )

    if unresolved:
        warnings.append("存在无法确定 pip 包名的依赖，未自动安装。")

    return DependencyReport(
        python_executable=python_executable or sys.executable,
        required_dependencies=unique_non_empty(required),
        installed_dependencies=installed,
        missing_dependencies=missing,
        unresolved_dependencies=unique_non_empty(unresolved),
        version_conflicts=[],
        warnings=warnings,
    )


@tool(
    "check_python_dependencies",
    args_schema=CheckPythonDependenciesInput,
    description="使用 importlib 检查依赖是否可导入，并保存 workspace/dependency_report.json。",
)
def check_python_dependencies(
    dependencies: list[str],
    python_executable: str = "",
    package_mapping_path: str = "docs/package_name_mapping.yaml",
    output_path: str = "workspace/dependency_report.json",
) -> dict:
    """检查当前解释器环境，不解析 pip list 文本。"""

    try:
        CheckPythonDependenciesInput(
            dependencies=dependencies,
            python_executable=python_executable,
            package_mapping_path=package_mapping_path,
            output_path=output_path,
        )
        report = build_dependency_report(dependencies, python_executable, package_mapping_path)
    except ValidationError as error:
        return error_result(
            "validation_error",
            "依赖检查参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查 dependencies 和输出路径。",
        )

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as error:
        return error_result(
            "save_failed",
            "依赖报告保存失败。",
            details={"output_path": str(path), "error": str(error)},
            suggestion="请检查输出目录权限。",
        )
    return {"success": True, "dependency_report": report.model_dump(), "output_path": str(path)}

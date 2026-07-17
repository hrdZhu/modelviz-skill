"""Tool: 安全安装缺失 Python 依赖。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import (
    DependencyInstallResult,
    InstallPythonDependenciesInput,
    PackageInstallResult,
)
from src.tools._utils import error_result, unique_non_empty
from src.tools.check_python_dependencies import build_dependency_report


PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _valid_package(package: str) -> bool:
    return bool(PACKAGE_RE.fullmatch(package)) and not any(
        marker in package for marker in ["://", "/", "\\", "git+", "--"]
    )


@tool(
    "install_python_dependencies",
    args_schema=InstallPythonDependenciesInput,
    description="安全安装确认缺失依赖，禁止 shell=True、URL、本地路径、Git 地址和额外 pip 参数。",
)
def install_python_dependencies(
    packages: list[str] | None = None,
    python_executable: str = "",
    timeout_seconds: int = 120,
    auto_install: bool = True,
    output_path: str = "workspace/dependency_install_result.json",
    requirements_output_path: str = "workspace/requirements.generated.txt",
) -> dict:
    """安装依赖并生成安装报告，不覆盖项目 requirements.txt。"""

    packages = unique_non_empty(packages or [])
    try:
        InstallPythonDependenciesInput(
            packages=packages,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            auto_install=auto_install,
            output_path=output_path,
            requirements_output_path=requirements_output_path,
        )
    except ValidationError as error:
        return error_result(
            "validation_error",
            "依赖安装参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请确认包名合法、数量不超过 20，timeout 在允许范围内。",
        )

    invalid = [package for package in packages if not _valid_package(package)]
    if invalid:
        return error_result(
            "invalid_package_name",
            "存在非法包名，拒绝安装。",
            details={"invalid_packages": invalid},
            suggestion="只允许普通 pip 包名，不允许 URL、本地路径、Git 地址或 pip 参数。",
        )

    executable = python_executable or sys.executable
    skipped: list[str] = []
    installed: list[str] = []
    failed: list[PackageInstallResult] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    warnings: list[str] = []

    if not auto_install:
        skipped = packages
        warnings.append("auto_install=false，未执行 pip install。")
    else:
        for package in packages:
            before = build_dependency_report([package], executable)
            if before.installed_dependencies and before.installed_dependencies[0].available:
                skipped.append(package)
                continue
            try:
                completed = subprocess.run(
                    [executable, "-m", "pip", "install", package],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    shell=False,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                failed.append(
                    PackageInstallResult(package_name=package, stderr=str(error), installed=False)
                )
                warnings.append(f"{package} 安装超时。")
                continue

            stdout_parts.append(completed.stdout)
            stderr_parts.append(completed.stderr)
            after = build_dependency_report([package], executable)
            available = bool(
                after.installed_dependencies and after.installed_dependencies[0].available
            )
            if completed.returncode == 0 and available:
                installed.append(package)
            else:
                failed.append(
                    PackageInstallResult(
                        package_name=package,
                        return_code=completed.returncode,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        installed=False,
                    )
                )

    result = DependencyInstallResult(
        success=not failed,
        requested_packages=packages,
        installed_packages=installed,
        failed_packages=failed,
        skipped_packages=skipped,
        stdout="\n".join(stdout_parts),
        stderr="\n".join(stderr_parts),
        warnings=warnings,
    )
    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        req = Path(requirements_output_path)
        req.parent.mkdir(parents=True, exist_ok=True)
        req.write_text("\n".join(packages) + ("\n" if packages else ""), encoding="utf-8")
    except OSError as error:
        return error_result(
            "save_failed",
            "依赖安装结果保存失败。",
            details={"error": str(error)},
            suggestion="请检查输出目录权限。",
        )
    return {
        "success": result.success,
        "install_result": result.model_dump(),
        "output_path": str(out),
    }

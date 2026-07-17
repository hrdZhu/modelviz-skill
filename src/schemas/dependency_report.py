"""模板依赖识别、检查和安装结果结构。"""

from typing import Annotated

from pydantic import BaseModel, Field

from .candidate_template import PathText


PackageName = Annotated[str, Field(min_length=1, max_length=120)]


class InspectTemplateDependenciesInput(BaseModel):
    """依赖识别工具参数。"""

    template_code_path: PathText = Field(..., description="选定模板 Python 源码路径。")
    catalog_dependencies: list[str] = Field(
        default_factory=list, description="模板元数据中的 dependencies。"
    )
    package_mapping_path: PathText = Field(
        "docs/package_name_mapping.yaml",
        description="import 名到 pip 包名的映射表。",
    )
    project_root: PathText = Field(".", description="项目根目录，用于排除内部模块。")


class DependencyItem(BaseModel):
    """单个依赖项。"""

    import_name: str
    package_name: str | None = None
    source: list[str] = Field(default_factory=list)


class DependencyInspectionResult(BaseModel):
    """模板依赖识别结果。"""

    success: bool = True
    template_code_path: str
    required_dependencies: list[DependencyItem] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    stdlib_dependencies: list[str] = Field(default_factory=list)
    internal_modules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CheckPythonDependenciesInput(BaseModel):
    """依赖检查工具参数。"""

    dependencies: list[str] = Field(..., description="需要检查的 import 名或包名。")
    python_executable: str = Field("", description="Python 解释器路径；为空时使用当前解释器。")
    package_mapping_path: PathText = Field(
        "docs/package_name_mapping.yaml", description="包名映射表路径。"
    )
    output_path: PathText = Field(
        "workspace/dependency_report.json", description="依赖报告保存路径。"
    )


class InstalledDependency(BaseModel):
    """已检查依赖状态。"""

    import_name: str
    package_name: str | None = None
    installed_version: str | None = None
    available: bool = False


class DependencyReport(BaseModel):
    """workspace/dependency_report.json 结构。"""

    success: bool = True
    python_executable: str
    required_dependencies: list[str] = Field(default_factory=list)
    installed_dependencies: list[InstalledDependency] = Field(default_factory=list)
    missing_dependencies: list[DependencyItem] = Field(default_factory=list)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    version_conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class InstallPythonDependenciesInput(BaseModel):
    """依赖安装工具参数。"""

    packages: list[PackageName] = Field(
        default_factory=list, max_length=20, description="待安装 pip 包名。"
    )
    python_executable: str = Field("", description="Python 解释器路径；为空时使用当前解释器。")
    timeout_seconds: Annotated[int, Field(gt=0, le=600)] = Field(
        120, description="单包安装超时时间。"
    )
    auto_install: bool = Field(True, description="是否允许自动安装。")
    output_path: PathText = Field(
        "workspace/dependency_install_result.json",
        description="安装结果 JSON 保存路径。",
    )
    requirements_output_path: PathText = Field(
        "workspace/requirements.generated.txt",
        description="本次模板依赖清单输出路径，不覆盖项目 requirements.txt。",
    )


class PackageInstallResult(BaseModel):
    """单个包安装结果。"""

    package_name: str
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    installed: bool = False


class DependencyInstallResult(BaseModel):
    """依赖安装完整结果。"""

    success: bool
    requested_packages: list[str] = Field(default_factory=list)
    installed_packages: list[str] = Field(default_factory=list)
    failed_packages: list[PackageInstallResult] = Field(default_factory=list)
    skipped_packages: list[str] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    warnings: list[str] = Field(default_factory=list)

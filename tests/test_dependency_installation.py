import subprocess
import importlib
from types import SimpleNamespace

from src.tools import check_python_dependencies, install_python_dependencies
from src.tools.dependency_utils import resolve_import_and_package


def test_resolve_package_name_case_insensitive_reverse_mapping():
    import_name, package_name = resolve_import_and_package("pillow", {"PIL": "Pillow"})

    assert import_name == "PIL"
    assert package_name == "Pillow"


def test_check_python_dependencies_mapping_and_missing(monkeypatch, tmp_path):
    module = importlib.import_module("src.tools.check_python_dependencies")
    monkeypatch.setattr(module, "dependency_available", lambda name: name == "sklearn")
    monkeypatch.setattr(module, "dependency_version", lambda package: "1.0")

    result = check_python_dependencies.invoke(
        {
            "dependencies": ["sklearn", "fake_missing"],
            "output_path": str(tmp_path / "dependency_report.json"),
        }
    )

    report = result["dependency_report"]
    assert result["success"] is True
    assert report["installed_dependencies"][0]["package_name"] == "scikit-learn"
    assert report["missing_dependencies"][0]["import_name"] == "fake_missing"


def test_install_python_dependencies_rejects_illegal_package():
    result = install_python_dependencies.invoke({"packages": ["https://example.com/pkg.whl"]})

    assert result["success"] is False
    assert result["error_type"] == "invalid_package_name"


def test_install_python_dependencies_success_timeout_failure_and_no_shell(monkeypatch, tmp_path):
    calls = []
    availability = {"demo_pkg": False}

    def fake_report(
        dependencies, python_executable="", package_mapping_path="docs/package_name_mapping.yaml"
    ):
        from src.schemas import DependencyReport, InstalledDependency

        package = dependencies[0]
        return DependencyReport(
            python_executable="python",
            required_dependencies=[package],
            installed_dependencies=[
                InstalledDependency(
                    import_name=package, package_name=package, available=availability[package]
                )
            ],
            missing_dependencies=[]
            if availability[package]
            else [{"import_name": package, "package_name": package}],
        )

    def fake_run(args, capture_output, text, timeout, shell, check):
        calls.append({"args": args, "shell": shell})
        availability["demo_pkg"] = True
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    module = importlib.import_module("src.tools.install_python_dependencies")
    monkeypatch.setattr(module, "build_dependency_report", fake_report)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = install_python_dependencies.invoke(
        {
            "packages": ["demo_pkg"],
            "output_path": str(tmp_path / "install.json"),
            "requirements_output_path": str(tmp_path / "requirements.generated.txt"),
        }
    )

    assert result["success"] is True
    assert calls[0]["shell"] is False
    assert (tmp_path / "requirements.generated.txt").read_text(encoding="utf-8") == "demo_pkg\n"

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pip", timeout=1)

    availability["demo_pkg"] = False
    monkeypatch.setattr(module.subprocess, "run", timeout_run)
    failed = install_python_dependencies.invoke(
        {
            "packages": ["demo_pkg"],
            "timeout_seconds": 1,
            "output_path": str(tmp_path / "install2.json"),
            "requirements_output_path": str(tmp_path / "requirements2.generated.txt"),
        }
    )
    assert failed["success"] is False
    assert failed["install_result"]["failed_packages"]

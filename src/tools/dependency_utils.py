"""依赖识别和包名映射辅助函数。"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import yaml


def load_package_mapping(path: str = "docs/package_name_mapping.yaml") -> dict[str, str]:
    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}
    data = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    return {str(key): str(value) for key, value in data.items()}


def reverse_package_mapping(mapping: dict[str, str]) -> dict[str, str]:
    return {value: key for key, value in mapping.items()}


def is_stdlib_module(import_name: str) -> bool:
    top = import_name.split(".")[0]
    return top in getattr(sys, "stdlib_module_names", set()) or top in sys.builtin_module_names


def is_internal_module(import_name: str, project_root: str = ".") -> bool:
    top = import_name.split(".")[0]
    root = Path(project_root)
    return (root / f"{top}.py").exists() or (root / top / "__init__.py").exists()


def resolve_import_and_package(name: str, mapping: dict[str, str]) -> tuple[str, str | None]:
    reverse = reverse_package_mapping(mapping)
    if name in mapping:
        return name, mapping[name]
    if name in reverse:
        return reverse[name], name
    lower_name = name.lower()
    for import_name, package_name in mapping.items():
        if package_name.lower() == lower_name:
            return import_name, package_name
    if "-" in name:
        return name, None
    return name, name


def dependency_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def dependency_version(package_name: str | None) -> str | None:
    if not package_name:
        return None
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

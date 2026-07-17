from src.tools import inspect_template_dependencies


def test_inspect_template_dependencies_ast_mapping_internal_and_dynamic(tmp_path):
    (tmp_path / "internal_mod.py").write_text("x = 1\n", encoding="utf-8")
    template = tmp_path / "template.py"
    template.write_text(
        "\n".join(
            [
                "import os",
                "import numpy as np",
                "import pandas",
                "from matplotlib import pyplot",
                "from sklearn.preprocessing import StandardScaler",
                "from .local import helper",
                "import internal_mod",
                "__import__('seaborn')",
            ]
        ),
        encoding="utf-8",
    )

    result = inspect_template_dependencies.invoke(
        {
            "template_code_path": str(template),
            "catalog_dependencies": ["Pillow", "numpy"],
            "project_root": str(tmp_path),
        }
    )

    assert result["success"] is True
    inspection = result["dependency_inspection"]
    packages = {item["package_name"] for item in inspection["required_dependencies"]}
    assert {"numpy", "pandas", "matplotlib", "scikit-learn", "Pillow"} <= packages
    assert "os" in inspection["stdlib_dependencies"]
    assert "internal_mod" in inspection["internal_modules"]
    assert inspection["warnings"]


def test_inspect_template_dependencies_standard_library_only(tmp_path):
    template = tmp_path / "template.py"
    template.write_text("import os\nfrom pathlib import Path\n", encoding="utf-8")

    result = inspect_template_dependencies.invoke({"template_code_path": str(template)})

    assert result["success"] is True
    assert result["dependency_inspection"]["required_dependencies"] == []

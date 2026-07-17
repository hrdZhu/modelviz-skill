import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"


def _readme_text() -> str:
    return README_PATH.read_text(encoding="utf-8")


def test_readme_exists_and_matches_skill_project_positioning():
    text = _readme_text()
    assert "ModelViz Skill" in text
    assert "SKILL.md" in text
    assert "Codex" in text
    assert "Claude Code" in text
    assert "不是普通命令行绘图应用" in text
    assert "不建议预先安装所有模板可能用到的依赖" in text


def test_readme_referenced_project_paths_exist():
    text = _readme_text()
    referenced_paths = [
        "SKILL.md",
        "prompts/",
        "templates/",
        "docs/",
        "src/prompts/",
        "src/tools/",
        "src/services/",
        "tests/",
        "evals/",
        "examples/",
        "workspace/",
        "outputs/",
        "requirements.txt",
        "docs/template_catalog.yaml",
        "docs/package_name_mapping.yaml",
        "evals/evaluation_report.md",
        "evals/evaluation_summary.json",
        "templates/01_CLU_clustering_reduction/01_CLU_004/assets/preview.png",
        "templates/06_NET_network_flow/06_NET_001/assets/preview.png",
        "templates/12_TRD_trend_time_series/12_TRD_003/output/preview.png",
        "templates/12_TRD_trend_time_series/12_TRD_004/assets/preview.png",
        "templates/01_CLU_clustering_reduction/01_CLU_003/assets/preview.png",
        "templates/01_CLU_clustering_reduction/01_CLU_001/output/preview.png",
        "templates/01_CLU_clustering_reduction/01_CLU_002/assets/preview.png",
    ]
    missing_from_text = [path for path in referenced_paths if path not in text]
    assert missing_from_text == []

    missing_on_disk = [path for path in referenced_paths if not (ROOT / path).exists()]
    assert missing_on_disk == []

    assert text.count("<img src=") == 7


def test_readme_does_not_contain_unsupported_install_or_cli_claims():
    text = _readme_text()
    forbidden_patterns = [
        r"git clone",
        r"pip install -r requirements\.txt",
        r"modelviz\s+run",
        r"python\s+main\.py",
        r"/modelviz",
        r"[A-Za-z]:\\",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_template_catalog_code_and_preview_paths_exist():
    catalog = yaml.safe_load((ROOT / "docs/template_catalog.yaml").read_text(encoding="utf-8"))
    templates = catalog.get("templates", [])
    assert len(templates) == 89

    missing_paths = []
    for item in templates:
        for field in ("code_path", "preview"):
            value = item.get(field)
            if value and not (ROOT / value).exists():
                missing_paths.append((item.get("id"), field, value))
    assert missing_paths == []


def test_runtime_directories_have_gitkeep_and_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for directory in ("workspace", "outputs"):
        assert (ROOT / directory / ".gitkeep").exists()
        assert f"{directory}/*" in gitignore
        assert f"!{directory}/.gitkeep" in gitignore

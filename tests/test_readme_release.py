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
    assert "面向数学建模竞赛和科研报告的科学可视化 Skill" in text
    assert "近百个正式科学可视化模板" in text
    assert "上百个预览与模板资产" in text
    assert "覆盖聚类降维、相关性、分布不确定性、预测评估、敏感性分析、网络关系、空间分析和时序趋势" in text
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
        "templates/01_CLU_clustering_reduction/01_CLU_001/output/preview.png",
        "templates/01_CLU_clustering_reduction/01_CLU_002/assets/preview.png",
        "templates/10_SEN_sensitivity_robustness/10_SEN_005/output/preview.png",
        "templates/10_SEN_sensitivity_robustness/10_SEN_011/output/preview.png",
        "templates/09_REL_relationship_correlation/09_REL_005/output/preview.png",
        "templates/09_REL_relationship_correlation/09_REL_013/output/preview.png",
        "templates/06_NET_network_flow/06_NET_001/assets/preview.png",
        "templates/03_EVL_composite_evaluation/03_EVL_002/output/preview.png",
        "templates/01_CLU_clustering_reduction/01_CLU_004/assets/preview.png",
        "templates/02_CMP_comparison_ranking/02_CMP_001/output/preview.png",
        "templates/01_CLU_clustering_reduction/01_CLU_003/assets/preview.png",
        "templates/02_CMP_comparison_ranking/02_CMP_012/output/preview.png",
        "templates/04_DIS_distribution_uncertainty/04_DIS_006/output/preview.png",
        "templates/05_MPN_multi_panel_report/05_MPN_004/output/preview.png",
        "templates/09_REL_relationship_correlation/09_REL_016/output/preview.png",
        "templates/09_REL_relationship_correlation/09_REL_017/output/preview.png",
        "templates/09_REL_relationship_correlation/09_REL_019/output/preview.png",
        "templates/10_SEN_sensitivity_robustness/10_SEN_006/output/preview.png",
        "templates/10_SEN_sensitivity_robustness/10_SEN_015/output/preview.png",
        "templates/11_SPA_spatial_geographic/11_SPA_001/output/preview.png",
    ]
    missing_from_text = [path for path in referenced_paths if path not in text]
    assert missing_from_text == []

    missing_on_disk = [path for path in referenced_paths if not (ROOT / path).exists()]
    assert missing_on_disk == []

    assert text.count("<img src=") == 20


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

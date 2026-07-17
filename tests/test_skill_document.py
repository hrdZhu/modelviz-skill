import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = ROOT / "SKILL.md"


REQUIRED_PATHS = [
    "docs/template_catalog.yaml",
    "docs/template_index.csv",
    "docs/requirement_vocabulary.yaml",
    "docs/package_name_mapping.yaml",
    "docs/stage5_stage6_skill_workflow.md",
    "prompts/template_adaptation.md",
    "prompts/visual_quality_check.md",
    "prompts/code_repair.md",
    "schemas/requirement_schema.py",
    "src/prompts/requirement_parser.py",
    "src/prompts/final_template_selector.py",
    "src/prompts/template_adaptation_plan.py",
    "src/prompts/template_code_adapter.py",
    "src/prompts/visual_quality_checker.py",
    "src/prompts/plot_code_repair.py",
    "src/services/requirement_parser.py",
    "src/services/candidate_matching_pipeline.py",
    "src/services/final_template_selection_pipeline.py",
    "src/services/template_adaptation_pipeline.py",
    "src/services/plot_quality_pipeline.py",
    "src/tools/load_requirement.py",
    "src/tools/load_catalog.py",
    "src/tools/match_candidate_templates.py",
    "src/tools/rank_candidate_templates.py",
    "src/tools/save_candidate_templates.py",
    "src/tools/prepare_dataset_context.py",
    "src/tools/load_selected_template.py",
    "src/tools/inspect_template_dependencies.py",
    "src/tools/check_python_dependencies.py",
    "src/tools/install_python_dependencies.py",
    "src/tools/save_adapted_script.py",
    "src/tools/execute_plot_script.py",
    "src/tools/validate_output_artifacts.py",
    "src/tools/inspect_generated_image.py",
    "src/tools/collect_plot_warnings.py",
]


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def test_skill_frontmatter_and_no_absolute_paths():
    text = _skill_text()
    assert text.startswith("---\n")
    frontmatter = text.split("---", 2)[1]
    metadata = yaml.safe_load(frontmatter)
    assert set(metadata) == {"name", "description"}
    assert metadata["name"] == "modelviz-skill"
    assert "Scientific visualization" in metadata["description"]

    assert not re.search(r"[A-Za-z]:\\", text)
    assert "run_template_matching_pipeline)" not in text
    assert "template_recommendations.json" not in text


def test_skill_referenced_paths_exist():
    text = _skill_text()
    missing_from_document = [path for path in REQUIRED_PATHS if path not in text]
    assert missing_from_document == []

    missing_on_disk = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    assert missing_on_disk == []


def test_skill_service_entries_are_importable():
    from src.services import (
        parse_and_save_requirement,
        run_candidate_matching_pipeline,
        run_final_template_selection_pipeline,
        run_plot_quality_pipeline,
        run_template_adaptation_pipeline,
    )

    assert callable(parse_and_save_requirement)
    assert callable(run_candidate_matching_pipeline)
    assert callable(run_final_template_selection_pipeline)
    assert callable(run_template_adaptation_pipeline)
    assert callable(run_plot_quality_pipeline)


def test_skill_tool_names_are_real():
    from src.tools import (
        STAGE3_CANDIDATE_TOOLS,
        check_python_dependencies,
        collect_plot_warnings,
        execute_plot_script,
        inspect_generated_image,
        inspect_template_dependencies,
        install_python_dependencies,
        load_selected_template,
        prepare_dataset_context,
        save_adapted_script,
        validate_output_artifacts,
    )

    text = _skill_text()
    stage3_names = [tool.name for tool in STAGE3_CANDIDATE_TOOLS]
    assert stage3_names == [
        "load_user_requirement",
        "load_template_catalog",
        "match_candidate_templates",
        "rank_candidate_templates",
        "save_candidate_templates",
    ]
    checked_tools = [
        *STAGE3_CANDIDATE_TOOLS,
        prepare_dataset_context,
        load_selected_template,
        inspect_template_dependencies,
        check_python_dependencies,
        install_python_dependencies,
        save_adapted_script,
        execute_plot_script,
        validate_output_artifacts,
        inspect_generated_image,
        collect_plot_warnings,
    ]
    assert all(tool.name in text for tool in checked_tools)


def test_skill_prompt_variables_match_real_templates():
    from src.prompts import (
        FINAL_TEMPLATE_SELECTOR_PROMPT,
        PLOT_CODE_REPAIR_PROMPT,
        REQUIREMENT_PARSER_PROMPT,
        TEMPLATE_ADAPTATION_PLAN_PROMPT,
        TEMPLATE_CODE_ADAPTER_PROMPT,
        VISUAL_QUALITY_CHECKER_PROMPT,
    )

    assert {
        "user_request",
        "functional_vocabulary",
        "chart_type_vocabulary",
        "style_vocabulary",
        "use_case_vocabulary",
    } <= set(REQUIREMENT_PARSER_PROMPT.input_variables)
    assert {
        "original_request",
        "structured_requirement",
        "dataset_context",
        "candidate_templates",
    } <= set(FINAL_TEMPLATE_SELECTOR_PROMPT.input_variables)
    assert {
        "template_source_code",
        "dependency_report",
        "dependency_install_result",
    } <= set(TEMPLATE_ADAPTATION_PLAN_PROMPT.input_variables)
    assert {"data_path", "adaptation_plan"} <= set(TEMPLATE_CODE_ADAPTER_PROMPT.input_variables)
    assert {"generated_image", "technical_quality_report"} <= set(
        VISUAL_QUALITY_CHECKER_PROMPT.input_variables
    )
    assert {"current_code", "visual_quality_report"} <= set(PLOT_CODE_REPAIR_PROMPT.input_variables)


def test_skill_matches_current_limits_and_data_support():
    text = _skill_text()
    assert "csv" in text
    assert "xlsx" in text
    assert "xls" in text
    assert "top_k=8" in text
    assert "5-10" in text
    assert "max_repair_attempts=3" in text
    assert "max_dependency_install_rounds=2" in text
    assert "89 templates" in text

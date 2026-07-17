"""LangChain tools for deterministic template matching."""

from .check_python_dependencies import check_python_dependencies
from .collect_plot_warnings import collect_plot_warnings
from .inspect_template_dependencies import inspect_template_dependencies
from .inspect_generated_image import inspect_generated_image
from .install_python_dependencies import install_python_dependencies
from .execute_plot_script import execute_plot_script
from .load_catalog import load_template_catalog
from .load_requirement import load_user_requirement
from .load_selected_template import load_selected_template
from .match_candidate_templates import match_candidate_templates
from .match_templates import match_templates
from .prepare_dataset_context import prepare_dataset_context
from .rank_candidate_templates import rank_candidate_templates
from .rank_templates import rank_templates
from .save_candidate_templates import save_candidate_templates
from .save_adapted_script import save_adapted_script
from .save_recommendations import save_template_recommendations
from .validate_output_artifacts import validate_output_artifacts

STAGE3_CANDIDATE_TOOLS = [
    load_user_requirement,
    load_template_catalog,
    match_candidate_templates,
    rank_candidate_templates,
    save_candidate_templates,
]

CORE_TEMPLATE_MATCHING_TOOLS = [
    load_user_requirement,
    load_template_catalog,
    match_templates,
    rank_templates,
    save_template_recommendations,
]

__all__ = [
    "CORE_TEMPLATE_MATCHING_TOOLS",
    "STAGE3_CANDIDATE_TOOLS",
    "check_python_dependencies",
    "collect_plot_warnings",
    "inspect_template_dependencies",
    "inspect_generated_image",
    "install_python_dependencies",
    "execute_plot_script",
    "load_selected_template",
    "load_template_catalog",
    "load_user_requirement",
    "match_candidate_templates",
    "match_templates",
    "prepare_dataset_context",
    "rank_candidate_templates",
    "rank_templates",
    "save_candidate_templates",
    "save_adapted_script",
    "save_template_recommendations",
    "validate_output_artifacts",
]

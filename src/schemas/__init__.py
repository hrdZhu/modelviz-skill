"""Schemas for stage three template recommendation."""

from .adaptation_plan import AdaptationPlan, ColumnMapping, TemplateAdaptationPipelineInput
from .adaptation_result import AdaptationResult, SaveAdaptedScriptInput
from .candidate_template import (
    CandidatePipelineInput,
    CandidateTemplate,
    CandidateTemplateOutput,
    MatchCandidateTemplatesInput,
    RankCandidateTemplatesInput,
    SaveCandidateTemplatesInput,
    TemplateCatalog,
    TemplateEntry,
    ToolError,
)
from .dataset_context import DatasetContext, DatasetContextInput
from .dependency_report import (
    CheckPythonDependenciesInput,
    DependencyInspectionResult,
    DependencyInstallResult,
    DependencyItem,
    DependencyReport,
    InspectTemplateDependenciesInput,
    InstallPythonDependenciesInput,
    InstalledDependency,
    PackageInstallResult,
)
from .execution_result import ExecutePlotScriptInput, ExecutionResult
from .final_template_selection import (
    CandidateComparison,
    FinalSelectionPipelineInput,
    FinalSelectionValidationContext,
    FinalTemplateSelection,
    validate_final_selection_references,
)
from .repair_result import FinalQualityReport, RepairHistoryItem, RepairResult
from .technical_quality_report import (
    CollectPlotWarningsInput,
    InspectGeneratedImageInput,
    TechnicalIssue,
    TechnicalQualityReport,
    ValidateOutputArtifactsInput,
)
from .template_recommendation import (
    LoadTemplateCatalogInput,
    LoadUserRequirementInput,
    MatchTemplatesInput,
    PipelineInput,
    RankTemplatesInput,
    SaveTemplateRecommendationsInput,
    TemplateMatchingError,
    TemplateRecommendation,
    TemplateRecommendationOutput,
)
from .visual_quality_report import VisualQualityReport

__all__ = [
    "AdaptationPlan",
    "AdaptationResult",
    "CandidateComparison",
    "CandidatePipelineInput",
    "CandidateTemplate",
    "CandidateTemplateOutput",
    "CheckPythonDependenciesInput",
    "CollectPlotWarningsInput",
    "ColumnMapping",
    "DatasetContext",
    "DatasetContextInput",
    "DependencyInspectionResult",
    "DependencyInstallResult",
    "DependencyItem",
    "DependencyReport",
    "ExecutePlotScriptInput",
    "ExecutionResult",
    "FinalSelectionPipelineInput",
    "FinalSelectionValidationContext",
    "FinalQualityReport",
    "FinalTemplateSelection",
    "InspectGeneratedImageInput",
    "InspectTemplateDependenciesInput",
    "InstallPythonDependenciesInput",
    "InstalledDependency",
    "LoadTemplateCatalogInput",
    "LoadUserRequirementInput",
    "MatchCandidateTemplatesInput",
    "MatchTemplatesInput",
    "PipelineInput",
    "PackageInstallResult",
    "RepairHistoryItem",
    "RepairResult",
    "SaveAdaptedScriptInput",
    "RankCandidateTemplatesInput",
    "RankTemplatesInput",
    "SaveCandidateTemplatesInput",
    "SaveTemplateRecommendationsInput",
    "TemplateCatalog",
    "TemplateAdaptationPipelineInput",
    "TemplateEntry",
    "TemplateMatchingError",
    "TemplateRecommendation",
    "TemplateRecommendationOutput",
    "ToolError",
    "TechnicalIssue",
    "TechnicalQualityReport",
    "ValidateOutputArtifactsInput",
    "VisualQualityReport",
    "validate_final_selection_references",
]

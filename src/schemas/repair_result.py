"""自动修复结果和最终质量报告。"""

from pydantic import BaseModel, Field, model_validator


class RepairResult(BaseModel):
    repaired_code: str
    fixed_issues: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)
    changes_summary: list[str] = Field(default_factory=list)
    data_columns_used: list[str] = Field(default_factory=list)
    dependencies_used: list[str] = Field(default_factory=list)
    additional_dependencies_requested: list[str] = Field(default_factory=list)
    can_retry: bool = True

    @model_validator(mode="after")
    def _code_not_empty(self):
        if not self.repaired_code.strip():
            raise ValueError("repaired_code 不得为空")
        return self


class RepairHistoryItem(BaseModel):
    attempt: int
    timestamp: str
    dependency_changes: list[str] = Field(default_factory=list)
    technical_issues: list[str] = Field(default_factory=list)
    visual_issues: list[str] = Field(default_factory=list)
    changes_summary: list[str] = Field(default_factory=list)
    script_path: str = ""
    execution_success: bool = False
    technical_passed: bool = False
    visual_passed: bool = False


class FinalQualityReport(BaseModel):
    passed: bool
    repair_attempts: int = 0
    dependency_install_rounds: int = 0
    final_dependencies: list[str] = Field(default_factory=list)
    final_script: str = ""
    final_outputs: list[str] = Field(default_factory=list)
    technical_summary: str = ""
    visual_summary: str = ""
    remaining_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

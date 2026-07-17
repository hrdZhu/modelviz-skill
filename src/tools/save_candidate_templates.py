"""Tool: 保存阶段三候选模板。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import CandidateTemplateOutput, SaveCandidateTemplatesInput
from src.tools._utils import error_result


@tool(
    "save_candidate_templates",
    args_schema=SaveCandidateTemplatesInput,
    description="保存阶段三候选模板到 workspace/candidate_templates.json，保存前执行 Pydantic 校验。",
)
def save_candidate_templates(
    candidate_result: dict,
    output_path: str = "workspace/candidate_templates.json",
) -> dict:
    """保存候选模板 JSON，中文不转义，不混入日志。"""

    try:
        SaveCandidateTemplatesInput(candidate_result=candidate_result, output_path=output_path)
        output = CandidateTemplateOutput.model_validate(candidate_result)
    except ValidationError as error:
        return error_result(
            "validation_error",
            "候选模板保存前校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查 candidate_result 是否符合 CandidateTemplateOutput。",
        )

    path = Path(output_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return error_result(
            "output_directory_error",
            "候选模板输出目录无法创建。",
            details={"output_path": str(path), "error": str(error)},
            suggestion="请检查输出路径和目录权限。",
        )

    try:
        path.write_text(
            json.dumps(output.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        return error_result(
            "save_failed",
            "候选模板保存失败。",
            details={"output_path": str(path), "error": str(error)},
            suggestion="请检查文件是否被占用或磁盘权限是否足够。",
        )

    return {"success": True, "output_path": str(path), "candidate_result": output.model_dump()}

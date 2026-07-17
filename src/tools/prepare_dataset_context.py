"""Tool: 准备提供给模型的数据上下文。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from pydantic import ValidationError

from src.schemas import DatasetContext, DatasetContextInput
from src.tools._utils import error_result


def _clean_value(value: Any) -> Any:
    """把 pandas/numpy 值转换为 JSON 可序列化值。"""

    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({str(key): _clean_value(value) for key, value in row.items()})
    return records


def _read_dataframe(
    path: Path, sheet_name: str | int | None
) -> tuple[pd.DataFrame, str | int | None]:
    file_type = path.suffix.lower().lstrip(".")
    if file_type == "csv":
        return pd.read_csv(path), None
    if file_type in {"xlsx", "xls"}:
        sheet = 0 if sheet_name is None else sheet_name
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except ValueError as error:
            raise KeyError(f"Excel Sheet 不存在：{sheet_name}") from error
        return df, sheet_name
    raise ValueError(f"不支持的数据格式：{path.suffix}")


@tool(
    "prepare_dataset_context",
    args_schema=DatasetContextInput,
    description=(
        "读取 CSV/XLSX/XLS 用户数据并准备模型上下文。"
        "只返回列名、行列数量和样例记录，不判断适合图表、不推荐模板。"
    ),
)
def prepare_dataset_context(
    data_path: str,
    output_path: str = "workspace/dataset_context.json",
    sheet_name: str | int | None = None,
    max_sample_rows: int = 20,
    large_file_row_threshold: int = 1000,
) -> dict:
    """安全读取用户数据样例，保存 dataset_context.json。"""

    try:
        DatasetContextInput(
            data_path=data_path,
            output_path=output_path,
            sheet_name=sheet_name,
            max_sample_rows=max_sample_rows,
            large_file_row_threshold=large_file_row_threshold,
        )
    except ValidationError as error:
        return error_result(
            "validation_error",
            "数据上下文参数校验失败。",
            details={"errors": error.errors()},
            suggestion="请检查 data_path、sheet_name 和 max_sample_rows。",
        )

    path = Path(data_path)
    if not path.exists():
        return error_result(
            "data_file_not_found",
            "用户数据文件不存在。",
            details={"data_path": str(path)},
            suggestion="请提供有效的 CSV、XLSX 或 XLS 文件路径。",
        )

    file_type = path.suffix.lower().lstrip(".")
    if file_type not in {"csv", "xlsx", "xls"}:
        return error_result(
            "unsupported_data_format",
            "数据格式不支持。",
            details={"data_path": str(path), "file_type": file_type},
            suggestion="请使用 CSV、XLSX 或 XLS 文件。",
        )

    try:
        df, resolved_sheet = _read_dataframe(path, sheet_name)
    except KeyError as error:
        return error_result(
            "excel_sheet_not_found",
            "Excel Sheet 不存在。",
            details={"data_path": str(path), "sheet_name": sheet_name, "error": str(error)},
            suggestion="请确认 sheet_name 是否正确，或留空读取第一个 sheet。",
        )
    except Exception as error:  # noqa: BLE001
        return error_result(
            "data_read_error",
            "数据无法读取。",
            details={"data_path": str(path), "error": str(error)},
            suggestion="请检查文件是否损坏、是否被占用或格式是否正确。",
        )

    if df.empty or len(df.columns) == 0:
        return error_result(
            "empty_dataset",
            "数据为空。",
            details={"data_path": str(path)},
            suggestion="请提供包含表头和数据行的文件。",
        )

    row_count = int(len(df))
    column_names = [str(column) for column in df.columns]
    df = df.rename(columns={column: str(column) for column in df.columns})
    head_rows = min(max_sample_rows, row_count)
    sampled = row_count > large_file_row_threshold
    if sampled:
        sample_df = df.sample(n=min(max_sample_rows, row_count), random_state=42)
        sample_strategy = (
            f"数据行数 {row_count} 超过阈值 {large_file_row_threshold}，"
            f"提供前 {head_rows} 行和固定随机样例。"
        )
    else:
        sample_df = df.head(head_rows)
        sample_strategy = f"数据行数 {row_count} 未超过阈值，提供前 {head_rows} 行作为样例。"

    context = DatasetContext(
        file_name=path.name,
        file_type=file_type,
        sheet_name=resolved_sheet,
        row_count=row_count,
        column_count=int(len(column_names)),
        column_names=column_names,
        sample_records=_records(sample_df),
        head_records=_records(df.head(head_rows)),
        sampled=sampled,
        sample_strategy=sample_strategy,
        warnings=["仅提供抽样上下文，模型不应声称看过全部数据。"] if sampled else [],
    )

    output = Path(output_path)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(context.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        return error_result(
            "save_failed",
            "数据上下文保存失败。",
            details={"output_path": str(output), "error": str(error)},
            suggestion="请检查输出目录权限。",
        )

    return {"success": True, "dataset_context": context.model_dump(), "output_path": str(output)}

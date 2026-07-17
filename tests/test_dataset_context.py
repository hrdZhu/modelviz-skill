import json

import pandas as pd

from src.schemas import DatasetContext
from src.tools import prepare_dataset_context


def test_prepare_dataset_context_csv_with_chinese_columns_and_time(tmp_path):
    data_path = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "年份": ["2020", "2021"],
            "地区": ["A", "B"],
            "产量": [10.5, None],
        }
    ).to_csv(data_path, index=False, encoding="utf-8")

    result = prepare_dataset_context.invoke(
        {"data_path": str(data_path), "output_path": str(tmp_path / "ctx.json")}
    )

    assert result["success"] is True
    context = DatasetContext.model_validate(result["dataset_context"])
    assert context.column_names == ["年份", "地区", "产量"]
    assert context.sample_records[1]["产量"] is None
    assert json.loads((tmp_path / "ctx.json").read_text(encoding="utf-8"))["column_names"] == [
        "年份",
        "地区",
        "产量",
    ]


def test_prepare_dataset_context_large_file_sampling(tmp_path):
    data_path = tmp_path / "large.csv"
    pd.DataFrame({"类别": ["A"] * 120, "数值": list(range(120))}).to_csv(data_path, index=False)

    result = prepare_dataset_context.invoke(
        {
            "data_path": str(data_path),
            "output_path": str(tmp_path / "ctx.json"),
            "max_sample_rows": 10,
            "large_file_row_threshold": 50,
        }
    )

    context = DatasetContext.model_validate(result["dataset_context"])
    assert context.sampled is True
    assert context.row_count == 120
    assert len(context.sample_records) == 10


def test_prepare_dataset_context_excel_sheet_missing_and_unsupported(tmp_path):
    xlsx_path = tmp_path / "data.xlsx"
    pd.DataFrame({"x": [1]}).to_excel(xlsx_path, index=False, sheet_name="Sheet1")

    missing_sheet = prepare_dataset_context.invoke(
        {"data_path": str(xlsx_path), "sheet_name": "Missing"}
    )
    assert missing_sheet["success"] is False
    assert missing_sheet["error_type"] == "excel_sheet_not_found"

    unsupported = tmp_path / "data.txt"
    unsupported.write_text("x", encoding="utf-8")
    result = prepare_dataset_context.invoke({"data_path": str(unsupported)})
    assert result["success"] is False
    assert result["error_type"] == "unsupported_data_format"

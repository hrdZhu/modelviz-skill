"""Tool: 检查生成 PNG 图片是否损坏或接近空白。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from langchain_core.tools import tool
from PIL import Image
from pydantic import ValidationError

from src.schemas import InspectGeneratedImageInput, TechnicalIssue
from src.tools._utils import error_result


@tool(
    "inspect_generated_image",
    args_schema=InspectGeneratedImageInput,
    description="读取 PNG 图片，检查尺寸、像素方差、透明度和可能空白问题。",
)
def inspect_generated_image(image_path: str) -> dict:
    """结合多个指标判断图片是否可能为空白。"""

    try:
        InspectGeneratedImageInput(image_path=image_path)
    except ValidationError as error:
        return error_result(
            "validation_error", "图片检查参数错误。", details={"errors": error.errors()}
        )
    path = Path(image_path)
    if not path.exists():
        return error_result("image_not_found", "图片不存在。", details={"image_path": str(path)})

    issues: list[TechnicalIssue] = []
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            arr = np.asarray(image.convert("RGBA"))
    except Exception as error:  # noqa: BLE001
        return error_result(
            "image_corrupted", "图片损坏或无法打开。", details={"error": str(error)}
        )

    rgb = arr[..., :3].astype(float)
    alpha = arr[..., 3].astype(float)
    variance = float(np.var(rgb))
    alpha_mean = float(np.mean(alpha))
    white_ratio = float(np.mean(np.all(rgb > 245, axis=2)))
    possible_blank = variance < 5.0 or white_ratio > 0.98 or alpha_mean < 5.0
    if width < 300 or height < 200:
        issues.append(TechnicalIssue(issue_type="small_image", message="图片尺寸偏小。"))
    if possible_blank:
        issues.append(
            TechnicalIssue(
                issue_type="possible_blank_image",
                severity="high",
                message="图片可能为空白或近似全透明。",
            )
        )

    return {
        "success": not possible_blank,
        "image_readable": True,
        "image_width": width,
        "image_height": height,
        "file_size": path.stat().st_size,
        "pixel_variance": round(variance, 4),
        "possible_blank_image": possible_blank,
        "issues": [issue.model_dump() for issue in issues],
    }

"""Standardized plotting template; the immutable source is in ``../original``."""

# Standardized editable copy; immutable source retained in ../original.
import os as _os
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent
DATA_DIR = TEMPLATE_DIR / "data"
OUTPUT_DIR = TEMPLATE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Force non-interactive rendering for reproducible template execution.
_os.environ["MPLBACKEND"] = "Agg"

from PIL import Image, ImageEnhance

original_image_path = str(DATA_DIR / "shap_grid_combined.png")  # 输入图像来自模板数据目录
# =========================================================================================
# ======================================饱和度调整===============================
# =========================================================================================
# 加载原始图像
img = Image.open(original_image_path)
# 创建一个颜色增强器
enhancer = ImageEnhance.Color(img)
saturation_factor = 1.5  # 定义一个饱和度增强因子变量，1.5 表示增加50%的饱和度
# 应用增强
img_saturated = enhancer.enhance(saturation_factor)
# 保存路径
saturated_image_path = str(OUTPUT_DIR / "饱和度增强1.png")
# 保存增强后的图像
img_saturated.save(saturated_image_path)

# =========================================================================================
# ======================================对比度调整===============================
# =========================================================================================
img = Image.open(original_image_path)  # 重新打开原始图像文件
enhancer_contrast = ImageEnhance.Contrast(img)  # 基于原始图像创建一个对比度增强器对象
img_contrasted = enhancer_contrast.enhance(1.5)  # 增加对比度
img_contrasted.save(str(OUTPUT_DIR / "增加对比度1.png"))
# =========================================================================================
# ======================================锐度调整===============================
# =========================================================================================
img = Image.open(original_image_path)  # 再次打开
# 调整锐度
enhancer_sharpness = ImageEnhance.Sharpness(img)
img_sharp = enhancer_sharpness.enhance(1.5)  # 增加锐度
img_sharp.save(str(OUTPUT_DIR / "调整锐度1.png"))
# =========================================================================================
# ======================================亮度调整===============================
# =========================================================================================
# 加载
img = Image.open(original_image_path)
# 调整亮度
enhancer_brightness = ImageEnhance.Brightness(img)
img_bright = enhancer_brightness.enhance(1.1)  # 增强亮度
img_bright.save(str(OUTPUT_DIR / "调整亮度1.png"))
# =========================================================================================
# ======================================综合调整调整===============================
# =========================================================================================
img = Image.open(original_image_path)  # 再次打开原始图像
# 增强对比度
enhancer_contrast = ImageEnhance.Contrast(img)
img = enhancer_contrast.enhance(1.15)
# 增强饱和度
enhancer_color = ImageEnhance.Color(img)
img = enhancer_color.enhance(1.3)
# 增强锐度
enhancer_sharpness = ImageEnhance.Sharpness(img)
img = enhancer_sharpness.enhance(1.2)
enhanced_path = str(OUTPUT_DIR / "final_grid_fully_enhanced1.png")
img.save(enhanced_path, dpi=(108, 108))

# Stage 0 compact output normalization: keep only output/preview.png and assets/preview.png.
def _stage0_finalize_preview():
    from pathlib import Path as _Stage0Path
    import shutil as _stage0_shutil

    template_dir = _Stage0Path(__file__).resolve().parent
    output_dir = template_dir / "output"
    assets_dir = template_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "preview.png"

    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    candidates = [
        p for p in output_dir.rglob("*")
        if p.is_file() and p.name != "preview.png" and p.suffix.lower() in image_suffixes
    ]
    if not preview_path.exists():
        if not candidates:
            raise RuntimeError("No image output was generated for output/preview.png")
        candidates.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
        source_image = candidates[0]
        if source_image.suffix.lower() == ".png":
            _stage0_shutil.copy2(source_image, preview_path)
        else:
            from PIL import Image as _Stage0Image

            with _Stage0Image.open(source_image) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")
                img.save(preview_path)

    import logging as _stage0_logging
    _stage0_logging.shutdown()

    for path in sorted(output_dir.rglob("*"), reverse=True):
        if path.is_file() and path.resolve() != preview_path.resolve():
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    assets_preview = assets_dir / "preview.png"
    _stage0_shutil.copy2(preview_path, assets_preview)
    for path in sorted(assets_dir.rglob("*"), reverse=True):
        if path.is_file() and path.resolve() != assets_preview.resolve():
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    allowed_root_dirs = {"data", "assets", "output"}
    for child in template_dir.iterdir():
        if child.is_dir() and child.name not in allowed_root_dirs:
            _stage0_shutil.rmtree(child, ignore_errors=True)
        elif child.is_file() and child.suffix.lower() != ".py":
            try:
                child.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    _stage0_finalize_preview()

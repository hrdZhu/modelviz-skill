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

# =========================================================================================
# ====================================== 1. 库的导入 =========================================
# =========================================================================================
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2. 颜色库设置=======================================
# =========================================================================================
color_schemes = {
    1: {
        "forest_light": "#a2d2a2",
        "forest_dark": "#1b7921",
        "openland_light": "#e0d493",
        "openland_dark": "#b8a32f",
        "delta_light": "#d3c9c9",
        "delta_dark": "#8b7d7d",
    },
    2: {
        "forest_light": "#a6cee3",
        "forest_dark": "#1f78b4",
        "openland_light": "#fdbf6f",
        "openland_dark": "#ff7f00",
        "delta_light": "#cccccc",
        "delta_dark": "#555555",
    },
    3: {
        "forest_light": "#cab2d6",
        "forest_dark": "#6a3d9a",
        "openland_light": "#b2df8a",
        "openland_dark": "#33a02c",
        "delta_light": "#d9d9d9",
        "delta_dark": "#636363",
    },
    4: {
        "forest_light": "#fbb4ae",
        "forest_dark": "#e41a1c",
        "openland_light": "#b3cde3",
        "openland_dark": "#377eb8",
        "delta_light": "#ccebc5",
        "delta_dark": "#4daf4a",
    },
    5: {
        "forest_light": "#fccde5",
        "forest_dark": "#e7298a",
        "openland_light": "#fed9a6",
        "openland_dark": "#a65628",
        "delta_light": "#d9d9d9",
        "delta_dark": "#525252",
    },
    6: {
        "forest_light": "#b3e2cd",
        "forest_dark": "#0868ac",
        "openland_light": "#fdcdac",
        "openland_dark": "#e66101",
        "delta_light": "#d1e5f0",
        "delta_dark": "#5e3c99",
    },
    7: {
        "forest_light": "#d9d9d9",
        "forest_dark": "#252525",
        "openland_light": "#fdd0a2",
        "openland_dark": "#d95f0e",
        "delta_light": "#c7e9c0",
        "delta_dark": "#238b45",
    },
    8: {
        "forest_light": "#ccebc5",
        "forest_dark": "#006d2c",
        "openland_light": "#fbb4ae",
        "openland_dark": "#b2182b",
        "delta_light": "#dddddd",
        "delta_dark": "#777777",
    },
    9: {
        "forest_light": "#e5f5e0",
        "forest_dark": "#31a354",
        "openland_light": "#fff2ae",
        "openland_dark": "#b15928",
        "delta_light": "#e0e0e0",
        "delta_dark": "#6a0d60",
    },
    10: {
        "forest_light": "#deebf7",
        "forest_dark": "#08519c",
        "openland_light": "#fee0d2",
        "openland_dark": "#cb181d",
        "delta_light": "#d9d9d9",
        "delta_dark": "#525252",
    },
    11: {
        "forest_light": "#a2d2a2",
        "forest_dark": "#1b7921",
        "openland_light": "#fdbf6f",
        "openland_dark": "#ff7f00",
        "delta_light": "#d3c9c9",
        "delta_dark": "#8b7d7d",
    },
    12: {
        "forest_light": "#a6cee3",
        "forest_dark": "#1f78b4",
        "openland_light": "#b2df8a",
        "openland_dark": "#33a02c",
        "delta_light": "#cccccc",
        "delta_dark": "#555555",
    },
    13: {
        "forest_light": "#cab2d6",
        "forest_dark": "#6a3d9a",
        "openland_light": "#e0d493",
        "openland_dark": "#b8a32f",
        "delta_light": "#d9d9d9",
        "delta_dark": "#636363",
    },
    14: {
        "forest_light": "#fbb4ae",
        "forest_dark": "#e41a1c",
        "openland_light": "#fed9a6",
        "openland_dark": "#a65628",
        "delta_light": "#ccebc5",
        "delta_dark": "#4daf4a",
    },
    15: {
        "forest_light": "#fccde5",
        "forest_dark": "#e7298a",
        "openland_light": "#b3cde3",
        "openland_dark": "#377eb8",
        "delta_light": "#d9d9d9",
        "delta_dark": "#525252",
    },
    16: {
        "forest_light": "#b3e2cd",
        "forest_dark": "#0868ac",
        "openland_light": "#fbb4ae",
        "openland_dark": "#b2182b",
        "delta_light": "#d1e5f0",
        "delta_dark": "#5e3c99",
    },
    17: {
        "forest_light": "#d9d9d9",
        "forest_dark": "#252525",
        "openland_light": "#b2df8a",
        "openland_dark": "#33a02c",
        "delta_light": "#c7e9c0",
        "delta_dark": "#238b45",
    },
    18: {
        "forest_light": "#ccebc5",
        "forest_dark": "#006d2c",
        "openland_light": "#fdbf6f",
        "openland_dark": "#ff7f00",
        "delta_light": "#dddddd",
        "delta_dark": "#777777",
    },
    19: {
        "forest_light": "#e5f5e0",
        "forest_dark": "#31a354",
        "openland_light": "#b3cde3",
        "openland_dark": "#377eb8",
        "delta_light": "#e0e0e0",
        "delta_dark": "#6a0d60",
    },
    20: {
        "forest_light": "#deebf7",
        "forest_dark": "#08519c",
        "openland_light": "#fed9a6",
        "openland_dark": "#a65628",
        "delta_light": "#d9d9d9",
        "delta_dark": "#525252",
    },
}
selected_scheme = 1  # 选择要使用的配色方案
# 选择配色方案
selected_colors = color_schemes[selected_scheme]


# =========================================================================================
# ======================================3. 绘图函数====================================
# =========================================================================================
def create_plot(years, forest_data, openland_data, delta_alpha_data, colors):
    fig, ax1 = plt.subplots(figsize=(10, 6))  # 创建一个画布和一组坐标轴
    ax2 = ax1.twinx()  # 创建第二个Y轴与ax1共享同一个X轴
    # 绘制Forest数据折线
    ax1.plot(
        years,  # 指定X轴数据
        forest_data,  # 指定Y轴数据
        marker="o",  # 设置数据点的标记样式
        markersize=5,  # 设置标记点的大小
        linestyle="-",  # 设置线条的样式
        color=colors["forest_light"],  # 设置线条的颜色为
        markerfacecolor=colors["forest_dark"],  # 设置标记点内部的填充颜色
        markeredgecolor=colors["forest_dark"],
    )  # 设置标记点边缘的颜色
    # 计算Forest数据的线性回归，获取斜率、截距和 p 值
    slope_f, intercept_f, _, p_value_f, _ = stats.linregress(years, forest_data)
    ax1.plot(
        years,  # X轴数据
        slope_f * years + intercept_f,  # Y轴数据
        linestyle="--",  # 线条的样式
        linewidth=3,  # 线条的宽度
        color=colors["forest_dark"],
    )  # 颜色

    ax1.text(
        2013,  # X坐标
        0.12,  # Y坐标
        f"Forest α$_w$ Trend = {slope_f * 10:.3f} unitless/decade p = {p_value_f:.2f}",  # 文本内容
        color=colors["forest_dark"],  # 文本的颜色
        fontsize=14,  # 字体大小
        ha="center",  # 水平对齐方式
        va="center",
    )  # 垂直对齐方式

    # 绘制Openland数据折线图
    ax1.plot(
        years,
        openland_data,
        marker="o",
        markersize=5,
        linestyle="-",
        color=colors["openland_light"],
        markerfacecolor=colors["openland_dark"],
        markeredgecolor=colors["openland_dark"],
    )
    # 计算并绘制Openland数据趋势线
    slope_o, intercept_o, _, p_value_o, _ = stats.linregress(years, openland_data)
    ax1.plot(
        years,
        slope_o * years + intercept_o,
        linestyle="--",
        linewidth=3,
        color=colors["openland_dark"],
    )
    # 添加对应的注释
    ax1.text(
        2013,
        0.28,
        f"Openland α$_w$ Trend = {slope_o * 10:.3f} unitless/decade   p = {p_value_o:.2f}",
        color=colors["openland_dark"],
        fontsize=14,
        ha="center",
        va="center",
    )

    # 绘制上面的柱状图
    ax2.bar(years, delta_alpha_data, color=colors["delta_light"], alpha=0.8, width=0.8)
    # 计算并绘制趋势线
    slope_d, intercept_d, _, p_value_d, _ = stats.linregress(years, delta_alpha_data)
    ax2.plot(
        years,
        slope_d * years + intercept_d,
        linestyle="--",
        linewidth=3,
        color=colors["delta_dark"],
    )
    # 添加文本注释
    ax2.text(
        2013,
        -0.1,
        f"Δα$_w$ Trend = {slope_d * 10:.3f} unitless/decade   p = {p_value_d:.2f}",
        color=colors["delta_dark"],
        fontsize=14,
        ha="center",
        va="center",
    )

    ax1.set_xlabel("Year", fontsize=18)  # x轴标题
    ax1.set_ylabel("α$_w$ (unitless)", fontsize=18, color="black")  # 左侧y轴标题
    ax2.set_ylabel(
        "Δα$_w$ (unitless)", fontsize=18, color=colors["delta_dark"], rotation=270, labelpad=20
    )  # 右侧y轴标题

    ax1.set_xlim(2003, 2024)  # X轴的显示范围
    ax1.set_xticks([2005, 2010, 2015, 2020])
    ax1.set_ylim(0.1, 0.5)  # 左侧y轴范围
    ax2.set_ylim(-0.4, 0.0)  # 右侧y轴范围
    ax1.tick_params(axis="y", labelsize=14, labelcolor="black")  # 左侧y轴刻度标注设置
    ax1.tick_params(axis="x", labelsize=14)  # x轴刻度标注设置
    ax2.tick_params(axis="y", labelsize=14, labelcolor=colors["delta_dark"])  # 右侧y轴刻度标注设置

    ax1.grid(axis="y", linestyle="-", linewidth=0.5, color="gray", alpha=0.3)  # 网格线，
    # 隐藏坐标轴边框
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    # 添加小标题
    ax1.text(-0.08, 1.02, "a", transform=ax1.transAxes, fontsize=24, fontweight="bold")
    plt.tight_layout()  # 自动调整图表布局

    plt.savefig(str(OUTPUT_DIR / f"{selected_scheme}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"{selected_scheme}.pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# ======================================4.数据的加载与处理=========================================
# =========================================================================================

df = pd.read_excel(str(DATA_DIR / "simulated_land_data.xlsx"))  # 读取数据
# 提取数据
years = df["Year"].values
forest_data = df["Forest_alpha_w"].values
openland_data = df["Openland_alpha_w"].values
delta_alpha_data = df["Delta_alpha_w"].values
# =========================================================================================
# ======================================5.绘图=========================================
# =========================================================================================
create_plot(
    years,  # x轴数据
    forest_data,  # 第一个折线
    openland_data,  # 第二个折线
    delta_alpha_data,  # 柱状图数据
    selected_colors,  # 颜色方案
)

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

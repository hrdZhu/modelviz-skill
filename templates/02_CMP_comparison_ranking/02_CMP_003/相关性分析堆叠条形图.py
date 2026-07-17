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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Times New Roman"
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ======================================2.设置颜色库 =========================================
# =========================================================================================
color_schemes = {
    1: {
        "Elevation": "#C8C8A9",
        "Latitude": "#92C5DE",
        "Longitude": "#F7D282",
        "Slope": "#FBAED2",
        "Aspect": "#CBA7D2",
        "Surface Relief": "#A8E6CF",
        "Surface Roughness": "#7AC5CD",
    },
    2: {
        "Elevation": "#A98D72",
        "Latitude": "#C8B9A5",
        "Longitude": "#E4D8C7",
        "Slope": "#B99B6B",
        "Aspect": "#8A6E58",
        "Surface Relief": "#D3B88C",
        "Surface Roughness": "#6F5B49",
    },
    3: {
        "Elevation": "#003B46",
        "Latitude": "#07575B",
        "Longitude": "#66A5AD",
        "Slope": "#C4DFE6",
        "Aspect": "#007580",
        "Surface Relief": "#8EC0C4",
        "Surface Roughness": "#2E8B57",
    },
    4: {
        "Elevation": "#FF6B6B",
        "Latitude": "#FFD166",
        "Longitude": "#F2A65A",
        "Slope": "#EF798A",
        "Aspect": "#B23A48",
        "Surface Relief": "#F9C784",
        "Surface Roughness": "#FC814A",
    },
    5: {
        "Elevation": "#588157",
        "Latitude": "#84A98C",
        "Longitude": "#A3B18A",
        "Slope": "#3A5A40",
        "Aspect": "#344E41",
        "Surface Relief": "#DAD7CD",
        "Surface Roughness": "#52796F",
    },
    6: {
        "Elevation": "#E07A5F",
        "Latitude": "#3D405B",
        "Longitude": "#81B29A",
        "Slope": "#F2CC8F",
        "Aspect": "#E85D04",
        "Surface Relief": "#F4F1DE",
        "Surface Roughness": "#B8AD9A",
    },
    7: {
        "Elevation": "#9CAFB7",
        "Latitude": "#B9B7BD",
        "Longitude": "#DAD2BC",
        "Slope": "#EAE0D5",
        "Aspect": "#C6B9B3",
        "Surface Relief": "#A29F98",
        "Surface Roughness": "#878583",
    },
    8: {
        "Elevation": "#FFADAD",
        "Latitude": "#FFD6A5",
        "Longitude": "#FDFFB6",
        "Slope": "#CAFFBF",
        "Aspect": "#9BF6FF",
        "Surface Relief": "#A0C4FF",
        "Surface Roughness": "#BDB2FF",
    },
    9: {
        "Elevation": "#0D1B2A",
        "Latitude": "#1B263B",
        "Longitude": "#415A77",
        "Slope": "#778DA9",
        "Aspect": "#A9BCD0",
        "Surface Relief": "#E0E1DD",
        "Surface Roughness": "#6B7A8F",
    },
    10: {
        "Elevation": "#005F73",
        "Latitude": "#0A9396",
        "Longitude": "#94D2BD",
        "Slope": "#E9D8A6",
        "Aspect": "#EE9B00",
        "Surface Relief": "#CA6702",
        "Surface Roughness": "#BB3E03",
    },
    11: {
        "Elevation": "#31363F",
        "Latitude": "#434A54",
        "Longitude": "#65737E",
        "Slope": "#AAB2B5",
        "Aspect": "#ACBDBA",
        "Surface Relief": "#C0C5C1",
        "Surface Roughness": "#D1D5D8",
    },
    12: {
        "Elevation": "#3C1874",
        "Latitude": "#654E92",
        "Longitude": "#9B72AA",
        "Slope": "#C895C3",
        "Aspect": "#E0B1D4",
        "Surface Relief": "#F5D5E8",
        "Surface Roughness": "#BEAEE2",
    },
    13: {
        "Elevation": "#8D021F",
        "Latitude": "#C83F49",
        "Longitude": "#D88C9A",
        "Slope": "#F2D0A9",
        "Aspect": "#F1E3D3",
        "Surface Relief": "#99C1B9",
        "Surface Roughness": "#545E56",
    },
    14: {
        "Elevation": "#FADADD",
        "Latitude": "#FCEECF",
        "Longitude": "#E4F4D4",
        "Slope": "#D4F0F0",
        "Aspect": "#D4E2F0",
        "Surface Relief": "#E0D4F0",
        "Surface Roughness": "#F0D4DF",
    },
    15: {
        "Elevation": "#4C0000",
        "Latitude": "#7B0000",
        "Longitude": "#A4133C",
        "Slope": "#C9184A",
        "Aspect": "#FF4D6D",
        "Surface Relief": "#FF758F",
        "Surface Roughness": "#FFB3C1",
    },
    16: {
        "Elevation": "#ABC8E2",
        "Latitude": "#88AED0",
        "Longitude": "#6495ED",
        "Slope": "#4169E1",
        "Aspect": "#1E3A8A",
        "Surface Relief": "#001A57",
        "Surface Roughness": "#000D2E",
    },
    17: {
        "Elevation": "#BF5700",
        "Latitude": "#E69F00",
        "Longitude": "#F0E442",
        "Slope": "#D55E00",
        "Aspect": "#CC79A7",
        "Surface Relief": "#AB4E52",
        "Surface Roughness": "#8B4513",
    },
    18: {
        "Elevation": "#2C3E50",
        "Latitude": "#34495E",
        "Longitude": "#7F8C8D",
        "Slope": "#95A5A6",
        "Aspect": "#BDC3C7",
        "Surface Relief": "#ECF0F1",
        "Surface Roughness": "#5D6D7E",
    },
    19: {
        "Elevation": "#E6A0B2",
        "Latitude": "#DDA0DD",
        "Longitude": "#BA55D3",
        "Slope": "#9932CC",
        "Aspect": "#8A2BE2",
        "Surface Relief": "#9370DB",
        "Surface Roughness": "#B19CD9",
    },
    20: {
        "Elevation": "#800020",
        "Latitude": "#A52A2A",
        "Longitude": "#D2691E",
        "Slope": "#F4A460",
        "Aspect": "#FFDAB9",
        "Surface Relief": "#FFE4C4",
        "Surface Roughness": "#F5DEB3",
    },
}
selected_scheme = 20  # 设置要使用的配色方案
selected_colors = color_schemes[selected_scheme]  # 获取配色方案


# =========================================================================================
# ======================================3.绘图函数 =========================================
# =========================================================================================
def create_and_save_plot(data_dict, order, colors, climate_labels, season_labels):
    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 8))

    # X轴位置
    x_positions = np.arange(len(climate_labels) * len(season_labels))

    # 绘制渐变矩形的函数
    def gradient_bar(ax, x, y, width, height, cmap):
        gradient = np.linspace(0, 1, 256).reshape(1, -1)  # 创建一个从0到1的线性渐变数组
        ax.imshow(
            gradient,  # 要显示的图像数据
            aspect="auto",  # 图像的宽高比
            cmap=cmap,  # 颜色映射
            extent=[
                x,
                x + width,
                y,
                y + height,
            ],  # 在坐标轴上的位置和范围，[起点, 终点, 起点, 终点]
            zorder=1,
        )

    # 绘制柱子旁边的阴影让柱子看上去像是悬浮在图面上
    shadow_offset = 0.03  # 阴影的水平偏移量
    shadow_width = 0.6  # 阴影的宽度
    shadow_alpha = 0.5  # 阴影的透明度
    shadow_cmap = LinearSegmentedColormap.from_list(
        "shadow_gradient",  # 渐变色的名字
        [(0, 0, 0, shadow_alpha), (0, 0, 0, 0)],  # 定义渐变色
    )

    bottom_shadow = np.zeros(len(x_positions))  # 用于记录每个柱子阴影的起始高度
    for factor in order:  # 遍历绘图
        values = data_dict[factor]  # 获取当前因子的数据值
        for i, (x_pos, height) in enumerate(zip(x_positions, values)):  # 遍历每个柱子的位置和高度
            if height > 0:  # 如果高度大于0，才绘制
                # 调用前面定义的函数来绘制渐变阴影
                gradient_bar(
                    ax,  # 指定要在哪个子图上进行绘制
                    x=x_pos + shadow_offset,  # X轴起始位置
                    y=bottom_shadow[i],  # Y轴起始位置
                    width=shadow_width,  # 渐变矩形的宽度
                    height=height,  # 渐变矩形的高度
                    cmap=shadow_cmap,  # 颜色映射
                )
        bottom_shadow += values  # 更新阴影的底部位置，为下一个因子的阴影做准备

    bottom_main = np.zeros(len(x_positions))  # 记录每个主数据柱子的起始高度
    for factor in order:  # 遍历因子绘图
        values = data_dict[factor]  # 获取当前因子的数据值
        # 绘制条形图
        ax.bar(
            x_positions,  # X轴位置
            values,  # 高度
            bottom=bottom_main,  # Y轴起始位置
            label=factor,  # 该组条形的名称，用于图例显示
            color=colors[factor],  # 填充颜色
            width=0.7,  # 宽度
            edgecolor="white",  # 边框颜色
            linewidth=3,  # 边框的线宽
            zorder=2,
        )
        bottom_main += values  # 更新主数据柱子的底部位置，为下一个因子做准备

    ax.grid(axis="y", linestyle="-", color="gray", alpha=0.6, zorder=0)  # 添加Y轴方向的水平网格线

    ax.legend(
        ncol=4, loc="lower center", bbox_to_anchor=(0.35, 1.05), frameon=False, fontsize=14
    )  # 设置图例
    ax.set_ylabel("Terrain Factor Dimensionality", fontsize=14)  # y轴标题
    ax.set_ylim(0, 7)  # y轴范围
    ax.set_yticks(np.arange(0, 8, 1))  # y轴刻度标注
    ax.set_xticks(x_positions)  # 设置X轴主刻度的位置
    ax.set_xticklabels(
        season_labels * len(climate_labels), fontsize=12, rotation=90
    )  # 设置X轴主刻度的标注

    ax.tick_params(axis="x", which="major", length=0)  # 去掉x轴主刻度线
    ax.tick_params(axis="y", length=0)  # 去掉y轴主刻度线

    ax.set_xlabel("")  # 不设置x轴标题
    # 遍历分组类别
    for i, climate in enumerate(climate_labels):
        pos = i * len(season_labels) + (len(season_labels) - 1) / 2.0  # 每个分组标注的中心位置
        ax.text(pos, -0.8, climate, ha="center", va="top", fontsize=14)  # 添加标注
    plt.subplots_adjust(bottom=0.15)  # 调整子图的底部边距，放置显示不全

    separator_positions = [
        i * len(season_labels) - 0.5 for i in range(0, len(climate_labels) + 1)
    ]  # 计算分隔线的位置
    ax.set_xticks(separator_positions, minor=True)  # 将这些位置设置为X轴的次刻度
    ax.tick_params(
        axis="x", which="minor", direction="out", length=60, width=1, color="gray"
    )  # 设置分隔线的样式

    # 去掉顶部、右侧和左侧的图框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.text(
        -0.05, 1.0, "f", transform=ax.transAxes, fontsize=20, fontweight="bold", va="top"
    )  # 添加小标题
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"{selected_scheme}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"{selected_scheme}.pdf"), bbox_inches="tight")


# =========================================================================================
# ====================================4.执行部分========================================
# =========================================================================================
if __name__ == "__main__":
    excel_filename = str(DATA_DIR / "data.xlsx")  # 原始数据的路径

    df = pd.read_excel(excel_filename, index_col=[0, 1])  # 读取数据
    print(df.head())

    legend_order = list(df.columns)  # 图例、堆叠图使用
    print(legend_order)
    climates = list(df.index.get_level_values(0).unique())  # x轴第一层
    print(climates)
    seasons = list(df.index.get_level_values(1).unique())  # x轴第二层
    print(seasons)
    data_from_excel = {col: df[col].to_numpy() for col in df.columns}  # 绘图数据
    print(data_from_excel)
    # 调用绘图函数
    create_and_save_plot(
        data_dict=data_from_excel,  # 绘图数据
        order=legend_order,  # 图例和堆叠的名称
        colors=selected_colors,  # 配色
        climate_labels=climates,  # x轴第一层
        season_labels=seasons,  # x轴第二层
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

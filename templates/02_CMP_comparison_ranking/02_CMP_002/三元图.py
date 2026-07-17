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
import matplotlib.pyplot as plt
import pandas as pd
import ternary

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
color_palettes = {
    1: {
        "cmap": "RdYlGn",
        "grid_h": "green",
        "grid_l": "red",
        "grid_r": "orange",
        "label_b": "darkgreen",
        "label_l": "darkred",
        "label_r": "orange",
    },
    2: {
        "cmap": "viridis",
        "grid_h": "blue",
        "grid_l": "purple",
        "grid_r": "teal",
        "label_b": "darkblue",
        "label_l": "purple",
        "label_r": "teal",
    },
    3: {
        "cmap": "inferno",
        "grid_h": "#E34234",
        "grid_l": "#FF8C00",
        "grid_r": "#FFD700",
        "label_b": "darkred",
        "label_l": "darkorange",
        "label_r": "goldenrod",
    },
    4: {
        "cmap": "YlGnBu",
        "grid_h": "blue",
        "grid_l": "green",
        "grid_r": "purple",
        "label_b": "darkblue",
        "label_l": "darkgreen",
        "label_r": "purple",
    },
    5: {
        "cmap": "Greys",
        "grid_h": "gray",
        "grid_l": "gray",
        "grid_r": "gray",
        "label_b": "black",
        "label_l": "black",
        "label_r": "black",
    },
    6: {
        "cmap": "plasma",
        "grid_h": "#6A00A8",
        "grid_l": "#E65C00",
        "grid_r": "#F0E442",
        "label_b": "#6A00A8",
        "label_l": "#E65C00",
        "label_r": "#B58D10",
    },
    7: {
        "cmap": "magma",
        "grid_h": "#3B0F70",
        "grid_l": "#B40426",
        "grid_r": "#F0E442",
        "label_b": "#3B0F70",
        "label_l": "#B40426",
        "label_r": "#B58D10",
    },
    8: {
        "cmap": "cividis",
        "grid_h": "#00224E",
        "grid_l": "#005F5F",
        "grid_r": "#FFDD40",
        "label_b": "#00224E",
        "label_l": "#005F5F",
        "label_r": "#B58D10",
    },
    9: {
        "cmap": "coolwarm",
        "grid_h": "blue",
        "grid_l": "red",
        "grid_r": "darkblue",
        "label_b": "blue",
        "label_l": "red",
        "label_r": "darkblue",
    },
    10: {
        "cmap": "bwr",
        "grid_h": "blue",
        "grid_l": "red",
        "grid_r": "darkgray",
        "label_b": "darkblue",
        "label_l": "darkred",
        "label_r": "black",
    },
    11: {
        "cmap": "PuOr",
        "grid_h": "purple",
        "grid_l": "orange",
        "grid_r": "saddlebrown",
        "label_b": "purple",
        "label_l": "orange",
        "label_r": "saddlebrown",
    },
    12: {
        "cmap": "Blues",
        "grid_h": "darkblue",
        "grid_l": "mediumblue",
        "grid_r": "royalblue",
        "label_b": "darkblue",
        "label_l": "mediumblue",
        "label_r": "royalblue",
    },
    13: {
        "cmap": "Greens",
        "grid_h": "darkgreen",
        "grid_l": "green",
        "grid_r": "seagreen",
        "label_b": "darkgreen",
        "label_l": "green",
        "label_r": "seagreen",
    },
    14: {
        "cmap": "Reds",
        "grid_h": "darkred",
        "grid_l": "firebrick",
        "grid_r": "red",
        "label_b": "darkred",
        "label_l": "firebrick",
        "label_r": "red",
    },
    15: {
        "cmap": "BuPu",
        "grid_h": "blue",
        "grid_l": "purple",
        "grid_r": "darkmagenta",
        "label_b": "blue",
        "label_l": "purple",
        "label_r": "darkmagenta",
    },
    16: {
        "cmap": "GnBu",
        "grid_h": "green",
        "grid_l": "blue",
        "grid_r": "darkblue",
        "label_b": "green",
        "label_l": "blue",
        "label_r": "darkblue",
    },
    17: {
        "cmap": "Spectral",
        "grid_h": "darkred",
        "grid_l": "darkblue",
        "grid_r": "darkgreen",
        "label_b": "darkred",
        "label_l": "darkblue",
        "label_r": "darkgreen",
    },
    18: {
        "cmap": "ocean",
        "grid_h": "green",
        "grid_l": "blue",
        "grid_r": "darkblue",
        "label_b": "green",
        "label_l": "blue",
        "label_r": "darkblue",
    },
    19: {
        "cmap": "terrain",
        "grid_h": "green",
        "grid_l": "sienna",
        "grid_r": "blue",
        "label_b": "darkgreen",
        "label_l": "saddlebrown",
        "label_r": "darkblue",
    },
    20: {
        "cmap": "twilight_shifted",
        "grid_h": "indigo",
        "grid_l": "darkblue",
        "grid_r": "darkred",
        "label_b": "indigo",
        "label_l": "darkblue",
        "label_r": "darkred",
    },
}
selected_palette = 20  # 选择的颜色方案
color = color_palettes[selected_palette]  # 获取配色


# =========================================================================================
# ====================================== 3.绘图函数 =========================================
# =========================================================================================
def plot_ternary_diagram(df, theme):
    selected_cmap = plt.get_cmap(theme["cmap"])  # 加载对应的颜色映射
    fig, ax = plt.subplots(figsize=(12, 10))  # 创建画布
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=100)  # 在指定的坐标轴上初始化一个三元图坐标系

    points = df[
        ["Ecology Quality", "Supply-Demand Equity", "Activity Quality"]
    ].values  # 提取三元图的三个坐标数据
    sizes = df["Carbon Sequestration(t)"]  # 提取用于定义散点大小的数据
    colors = df["Carbon Emissions(t)"]  # 提取用于定义散点颜色的数据
    # 调用三元图坐标系的scatter方法绘制散点图
    tax.scatter(
        points,  # 散点的位置坐标数据
        s=sizes / 100,  # 散点的大小
        c=colors,  # 散点的颜色
        cmap=selected_cmap,  # 颜色映射
        vmin=5000,  # 颜色映射的最小值
        vmax=20000,  # 颜色映射的最大值
        alpha=0.8,  # 透明度
        edgecolors="gray",  # 散点的边缘颜色
        linewidth=0.5,  # 散点边缘线的宽度
    )
    # 设置三元图的标题
    tax.set_title("15-Minute Walking Neighbourhood", fontsize=18, weight="bold", pad=40)
    # tax.boundary(lw=2, axes_colors={'l': theme['grid_h'], 'r': 'black', 'b': 'black'}) # 绘制三元图的边界线
    # 绘制三元图的网格线
    tax.gridlines(
        multiple=20,  # 网格线的间隔
        linestyle="-.",  # 网格线的样式
        horizontal_kwargs={"color": theme["grid_h"], "alpha": 0.7, "lw": 2},  # 设置水平网格线
        left_kwargs={"color": theme["grid_l"], "alpha": 0.7, "lw": 2},  # 设置左侧网格线
        right_kwargs={"color": theme["grid_r"], "alpha": 0.7, "lw": 2},  # 设置右侧网格线
    )
    # 设置刻度线
    tax.ticks(axis="lbr", multiple=20, offset=0.02, fontsize=20)
    # tax.clear_matplotlib_ticks() # 清除 matplotlib 默认生成的（非三元图的）刻度线
    tax.get_axes().axis("off")  # 去掉常规的坐标轴的显示
    tax.left_axis_label(
        "Activity Quality", fontsize=18, color=theme["label_l"], offset=0.12
    )  # 设置左侧坐标轴的标题
    tax.right_axis_label(
        "Efficiency Equity", fontsize=18, color=theme["label_r"], offset=0.12
    )  # 设置右侧坐标轴的标题
    tax.bottom_axis_label(
        "Ecology Quality", fontsize=18, color=theme["label_b"], offset=0.06
    )  # 设置底部坐标轴的标题
    cax = fig.add_axes([0.945, 0.2, 0.03, 0.35])  # 添加一个新的坐标轴用于放置颜色条，[左, 下,宽,高]
    sm = plt.cm.ScalarMappable(
        cmap=selected_cmap, norm=plt.Normalize(vmin=5000, vmax=20000)
    )  # 创建一个可映射标量对象，使用与散点图相同的颜色映射和归一化范围
    cbar = fig.colorbar(sm, cax=cax)  # 绘制颜色条
    # 设置颜色条的标题
    cbar.set_label("Carbon Emissions(t)", fontsize=16, rotation=270, labelpad=20)
    cbar.set_ticks(
        [5000, 10000, 15000, 20000],
    )  # 设置颜色条上显示的刻度值
    cbar.ax.tick_params(labelsize=16)
    # 创建一个列表，用于存放图例的元素
    legend_elements = [
        plt.scatter(
            [],  # X坐标
            [],  # Y坐标
            s=5000 / 100,  # 散点的大小
            facecolors="none",  # 空心圆
            edgecolors="black",  # 边缘颜色
            linewidth=0.5,  # 边缘线的宽度
            label="5000",
        ),  # 点的文本标注
        plt.scatter(
            [],
            [],
            s=10000 / 100,
            facecolors="none",
            edgecolors="black",
            linewidth=0.5,
            label="10000",
        ),
        plt.scatter(
            [],
            [],
            s=15000 / 100,
            facecolors="none",
            edgecolors="black",
            linewidth=0.5,
            label="15000",
        ),
    ]
    # 创建图例
    legend = ax.legend(
        handles=legend_elements,  # 图例中要显示的元素
        title="Carbon Sequestration(t)",  # 图例的标题（
        loc="upper right",  # 图例的位置
        bbox_to_anchor=(1.25, 0.95),  # 详细位置
        labelspacing=2.0,  # 图例垂直间距
        borderpad=1.2,  # 图例边框与内部元素的间距
        title_fontsize=16,  # 图例标题的字体大小
        fontsize=16,  # 图例标签的字体大小
        frameon=False,  # 不显示图例的边框
    )
    ax.add_artist(legend)  # 将创建的图例添加到坐标轴
    plt.savefig(str(OUTPUT_DIR / f"{selected_palette}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"{selected_palette}.pdf"), bbox_inches="tight")


# =========================================================================================
# ======================================4.数据读取以及绘图=========================================
# =========================================================================================
# 读取原始数据
df = pd.read_excel(str(DATA_DIR / "simulated_data.xlsx"))
# 调用绘图函数
plot_ternary_diagram(
    df=df,  # 数据
    theme=color,  # 颜色方案
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

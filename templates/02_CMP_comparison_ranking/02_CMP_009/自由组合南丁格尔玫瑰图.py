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

# 导入 numpy 库，通常用于科学计算，尤其擅长处理数组和矩阵
# 导入 matplotlib 库的主模块，用于更底层的配置
import matplotlib

# 导入 mpatches 模块，用于创建图例中的颜色块（Patch）
import matplotlib.patches as mpatches

# 导入 matplotlib.pyplot 库，这是 Python 中最常用的绘图库
import matplotlib.pyplot as plt
import numpy as np


def draw_final_chart(ax, chart_data):
    """
    :param ax: 要在上面绘图的子图对象 (Axes)。
    :param chart_data: 包含该图表所有数据的字典。
    """
    # --- 从数据字典中解包数据 ---
    # 获取图表标题
    title = chart_data["title"]
    # 获取数据值列表（百分比）
    values = chart_data["values"]
    # 获取颜色列表
    colors = chart_data["colors"]
    # 获取图例标签列表
    legend_labels = chart_data["legend_labels"]
    # 获取变量（数据点）的数量
    num_vars = len(values)
    # 计算所有数据值的总和，用于后续计算每个值所占的比例
    sum_of_values = sum(values)
    # 根据每个数据值占总和的比例，计算其在极坐标图（饼图）中应占的角度（弧度）
    widths = [(v / sum_of_values) * 2 * np.pi for v in values]
    # 设置一个基础长度，决定了最内圈的半径
    base_length = 3.0
    # 设置一个固定的递增量，使得每个数据环的半径比前一个大
    fixed_increment = 0.4
    # 计算每个数据环的总长度（半径）
    total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
    # 设置外圈彩色数据环的宽度
    colored_ring_width = 2.0
    # 计算内圈灰色底座的高度（半径），等于总长度减去外圈宽度
    inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
    # 定义内圈底座交替使用的颜色
    inner_colors = ["#EAEAEA", "#FFFFFF"] * (num_vars // 2 + 1)
    # 截取所需数量的内圈颜色
    inner_colors = inner_colors[:num_vars]
    # 设置一个偏移量，让整个图表稍微旋转，使第一个扇区的起始位置看起来更像在“一点钟”方向
    one_oclock_offset = np.pi / 21
    # 计算每个扇区（bar）的起始角度（theta）。np.cumsum是累积求和，用于确定每个扇区的位置
    thetas = np.cumsum([0] + widths[:-1]) - one_oclock_offset
    # 1. 绘制内圈底座
    ax.bar(
        x=thetas,
        height=inner_heights,
        width=widths,
        color=inner_colors,
        align="edge",
        edgecolor="white",
        linewidth=1.5,
    )
    # 2. 绘制外圈数据环
    ax.bar(
        x=thetas,
        height=[colored_ring_width] * num_vars,
        width=widths,
        bottom=inner_heights,
        color=colors,
        align="edge",
        edgecolor="white",
        linewidth=1.5,
    )
    # 遍历每一个数据点，为其添加文本标签
    for i in range(num_vars):
        # 计算标签所在位置的角度（弧度），即扇区中间的角度
        label_angle_rad = thetas[i] + widths[i] / 2
        # 计算标签所在位置的半径，比数据环稍远一些
        label_radius = total_lengths[i] + 1.2
        # 在计算好的位置添加文本，显示百分比值，并设置对齐方式、旋转角度和字体大小
        ax.text(
            label_angle_rad,
            label_radius,
            f"{values[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=18,
        )
    # 在子图的指定位置添加标题
    ax.text(0.87, 0.95, title, transform=ax.transAxes, fontsize=18, ha="left", va="center")
    # 美化和配置
    # 隐藏Y轴的刻度标签
    ax.set_yticklabels([])
    # 隐藏X轴（角度轴）的刻度标签
    ax.set_xticklabels([])
    # 隐藏极坐标图最外层的边框线
    ax.spines["polar"].set_visible(False)
    # 关闭极坐标图的网格线
    ax.grid(False)
    # 设置极坐标的0度位置在正北方（'N'）
    ax.set_theta_zero_location("N")
    # 设置极坐标的角度方向为顺时针（-1）
    ax.set_theta_direction(-1)
    # 设置Y轴的范围，使其刚好能容纳所有环和标签，减少多余的空白
    ax.set_ylim(0, max(total_lengths) + 0.1)
    # 添加图例
    # 使用 mpatches.Patch 创建图例的句柄（颜色块和对应的标签）
    legend_handles = [
        mpatches.Patch(color=colors[i], label=legend_labels[i]) for i in range(num_vars)
    ]
    # 在子图上显示图例
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.85, 0.5),
        frameon=False,
        fontsize=12,
        labelspacing=0.05,
    )


# --- 程序的【主逻辑】开始 ---
# --- 1. 全局设置 ---
# 设置matplotlib的后端，'TkAgg'是一个常用的、与Tkinter GUI工具包集成的后端
matplotlib.use("Agg")
# 设置全局字体为 'serif'（衬线字体）
plt.rcParams["font.family"] = "serif"
# 指定 'serif' 字体族具体使用 'Times New Roman'
plt.rcParams["font.serif"] = "Times New Roman"
# --- 2. 准备数据 ---
# 定义所有图表共享的颜色和图例标签
common_colors = [
    "#C42121",
    "#E84D5B",
    "#F27676",
    "#FF9980",
    "#FFB479",
    "#FFE083",
    "#FFFF99",
    "#ACD6FF",
    "#79ABDC",
    "#4682B4",
    "#9AD4D4",
    "#60B4B4",
    "#008B8B",
    "#7D7DFF",
    "#6A5ACD",
]
# 颜色顺序和图例标签顺序需要和您的数据值顺序一一对应
common_legend_labels = [
    "A1-8",
    "B2-10",
    "B1-7",
    "C1-1",
    "A1-1",
    "A1-3",
    "A1-5",
    "B2-6",
    "B2-15",
    "C1-6",
    "A3-1",
    "B2-8",
    "A1-11",
    "A1-2",
    "A1-3",
]
# 【【【 在这里自由控制图表数量和数据 】】】
# 通过增删列表中的字典项，来决定最终绘制几张图。
all_data = [
    {
        "title": "BTH",
        "values": [10.0, 9.5, 8.0, 7.9, 7.6, 7.2, 7.2, 7.1, 6.8, 5.6, 5.0, 4.9, 4.8, 4.3, 4.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "PRD",
        "values": [8.0, 8.5, 9.0, 7.5, 7.8, 6.2, 8.2, 9.1, 6.4, 5.9, 5.1, 4.2, 5.8, 6.3, 7.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "YRD",
        "values": [11.0, 9.1, 8.4, 7.2, 7.1, 7.8, 6.9, 7.0, 6.1, 5.4, 5.8, 4.1, 4.2, 4.9, 5.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "CY",
        "values": [9.0, 9.1, 8.2, 8.9, 7.0, 7.1, 7.8, 7.1, 6.2, 5.1, 5.2, 4.3, 4.9, 4.1, 5.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "SP",
        "values": [12.0, 8.5, 8.1, 7.3, 7.0, 7.1, 6.8, 7.5, 6.0, 5.0, 5.1, 4.8, 4.2, 4.1, 4.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "BTH",
        "values": [10.0, 9.5, 8.0, 7.9, 7.6, 7.2, 7.2, 7.1, 6.8, 5.6, 5.0, 4.9, 4.8, 4.3, 4.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "PRD",
        "values": [8.0, 8.5, 9.0, 7.5, 7.8, 6.2, 8.2, 9.1, 6.4, 5.9, 5.1, 4.2, 5.8, 6.3, 7.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "YRD",
        "values": [11.0, 9.1, 8.4, 7.2, 7.1, 7.8, 6.9, 7.0, 6.1, 5.4, 5.8, 4.1, 4.2, 4.9, 5.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    {
        "title": "YRD",
        "values": [11.0, 9.1, 8.4, 7.2, 7.1, 7.8, 6.9, 7.0, 6.1, 5.4, 5.8, 4.1, 4.2, 4.9, 5.0],
        "colors": common_colors,
        "legend_labels": common_legend_labels,
    },
    # 您可以继续在这里添加新的字典来增加图表...
    # {
    #     'title': "New Chart",
    #     'values': [ ... ],
    #     'colors': common_colors,
    #     'legend_labels': common_legend_labels
    # }
]

# --- 3. 动态创建网格并循环绘图 ---
# 获取要绘制的图表总数
num_charts = len(all_data)
# 仅在有数据时才执行绘图操作
if num_charts > 0:
    # 自动计算网格的行数和列数，使其尽可能接近正方形
    ncols = int(np.ceil(np.sqrt(num_charts)))
    nrows = int(np.ceil(num_charts / ncols))
    # 根据计算出的行列数和图表数量动态创建子图网格和画布大小
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5, nrows * 5), subplot_kw={"projection": "polar"}
    )
    # 不论子图网格是几维的，都将其展平为一维数组，方便遍历
    # 使用 np.atleast_1d 确保在只有一个图（num_charts=1）时也能正常工作
    axes = np.atleast_1d(axes).flatten()
    # 循环遍历数据和子图，进行绘图
    for i, ax in enumerate(axes):
        # 检查当前子图索引是否在数据范围内
        if i < num_charts:
            # 传入当前子图 ax 和对应的数据 all_data[i] 进行绘制
            draw_final_chart(ax, all_data[i])
        else:
            # 如果存在多余的子图（例如需要画7张图，但创建了3x3=9个位置），则隐藏多余的
            ax.set_visible(False)
    # --- 4. 调整布局和保存原始图像 ---
    # 自动调整子图参数以适应画布
    fig.tight_layout()
    # 如果 tight_layout 效果不佳，可以尝试使用此行手动调整子图的水平和垂直间距
    plt.subplots_adjust(hspace=-0.25, wspace=0.2)
    # 定义原始图片的保存路径
    original_image_path = str(OUTPUT_DIR / "final_grid_flexible.png")
    # 将最终生成的图像保存到指定路径，dpi设置分辨率，bbox_inches='tight'裁剪掉多余空白
    plt.savefig(original_image_path, dpi=108, bbox_inches="tight")
    # --- 5. 显示图表 ---
    # 在屏幕上显示matplotlib图表窗口
    plt.close("all")  # Interactive display removed; assets were exported above.
else:
    print("数据列表 'all_data' 为空，没有图表可以绘制。")

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

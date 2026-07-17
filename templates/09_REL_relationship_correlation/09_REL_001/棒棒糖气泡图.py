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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
COLOR_LIBRARY = {
    1: {
        "pleasure": "#0571B0",
        "safety": "#92C5DE",
        "comfort": "#F4A582",
        "bg_pleasure": "#D6E6F0",
        "bg_safety": "#E6F0F6",
        "bg_comfort": "#FEECE2",
        "size_legend": ["#92C5DE", "#F4A582", "#6BAED6", "#0571B0"],
    },
    2: {
        "pleasure": "#1f77b4",
        "safety": "#ff7f0e",
        "comfort": "#2ca02c",
        "bg_pleasure": "#DDF0FF",
        "bg_safety": "#FFF2E5",
        "bg_comfort": "#E0F5E0",
        "size_legend": ["#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"],
    },
    3: {
        "pleasure": "#e41a1c",
        "safety": "#377eb8",
        "comfort": "#4daf4a",
        "bg_pleasure": "#FDE0E0",
        "bg_safety": "#E0E8F6",
        "bg_comfort": "#E6F5E6",
        "size_legend": ["#377eb8", "#4daf4a", "#984ea3", "#e41a1c"],
    },
    4: {
        "pleasure": "#1f78b4",
        "safety": "#33a02c",
        "comfort": "#e31a1c",
        "bg_pleasure": "#a6cee3",
        "bg_safety": "#b2df8a",
        "bg_comfort": "#fb9a99",
        "size_legend": ["#33a02c", "#e31a1c", "#fdbf6f", "#1f78b4"],
    },
    5: {
        "pleasure": "#d73027",
        "safety": "#fc8d59",
        "comfort": "#fee090",
        "bg_pleasure": "#FADAD9",
        "bg_safety": "#FEE7DB",
        "bg_comfort": "#FFF9E7",
        "size_legend": ["#fc8d59", "#fee090", "#fE6E2e", "#d73027"],
    },
    6: {
        "pleasure": "#02818a",
        "safety": "#3690c0",
        "comfort": "#a6bddb",
        "bg_pleasure": "#D9F2F4",
        "bg_safety": "#E1ECF4",
        "bg_comfort": "#ECEFF5",
        "size_legend": ["#3690c0", "#a6bddb", "#67a9cf", "#02818a"],
    },
    7: {
        "pleasure": "#006d2c",
        "safety": "#31a354",
        "comfort": "#74c476",
        "bg_pleasure": "#E0F0E4",
        "bg_safety": "#E3F3E8",
        "bg_comfort": "#EFF7EF",
        "size_legend": ["#31a354", "#74c476", "#a1d99b", "#006d2c"],
    },
    8: {
        "pleasure": "#d7191c",
        "safety": "#abdda4",
        "comfort": "#2b83ba",
        "bg_pleasure": "#FAD8D8",
        "bg_safety": "#EDF8EA",
        "bg_comfort": "#DDE6F0",
        "size_legend": ["#abdda4", "#2b83ba", "#fdae61", "#d7191c"],
    },
    9: {
        "pleasure": "#4878d0",
        "safety": "#ee854a",
        "comfort": "#6acc64",
        "bg_pleasure": "#E4EBF9",
        "bg_safety": "#FBEDE5",
        "bg_comfort": "#EAF7E9",
        "size_legend": ["#ee854a", "#6acc64", "#d65f5f", "#4878d0"],
    },
    10: {
        "pleasure": "#003f5c",
        "safety": "#367f9d",
        "comfort": "#74bfd8",
        "bg_pleasure": "#E0E9ED",
        "bg_safety": "#E4ECF0",
        "bg_comfort": "#EEF6F9",
        "size_legend": ["#367f9d", "#74bfd8", "#a4dff0", "#003f5c"],
    },
    11: {
        "pleasure": "#5E4A4C",
        "safety": "#a2b0c3",
        "comfort": "#e6a0c4",
        "bg_pleasure": "#E9E5E5",
        "bg_safety": "#ECEFF3",
        "bg_comfort": "#FBEFF7",
        "size_legend": ["#a2b0c3", "#e6a0c4", "#c6dbc0", "#5E4A4C"],
    },
    12: {
        "pleasure": "#5f442b",
        "safety": "#a67c52",
        "comfort": "#c7ad92",
        "bg_pleasure": "#E9E4E1",
        "bg_safety": "#F0EBE6",
        "bg_comfort": "#F6F2ED",
        "size_legend": ["#a67c52", "#c7ad92", "#826a51", "#5f442b"],
    },
    13: {
        "pleasure": "#386cb0",
        "safety": "#fdc086",
        "comfort": "#7fc97f",
        "bg_pleasure": "#E1EAF6",
        "bg_safety": "#FFF5EB",
        "bg_comfort": "#EBF7EB",
        "size_legend": ["#fdc086", "#7fc97f", "#beaed4", "#386cb0"],
    },
    14: {
        "pleasure": "#1b9e77",
        "safety": "#d95f02",
        "comfort": "#7570b3",
        "bg_pleasure": "#DEF2EB",
        "bg_safety": "#FBE9DE",
        "bg_comfort": "#EAE9F3",
        "size_legend": ["#d95f02", "#7570b3", "#e7298a", "#1b9e77"],
    },
    15: {
        "pleasure": "#2c7bb6",
        "safety": "#abd9e9",
        "comfort": "#fdae61",
        "bg_pleasure": "#DFEBF6",
        "bg_safety": "#EFF8FA",
        "bg_comfort": "#FEF2E9",
        "size_legend": ["#abd9e9", "#fdae61", "#fEbc74", "#2c7bb6"],
    },
    16: {
        "pleasure": "#252525",
        "safety": "#636363",
        "comfort": "#969696",
        "bg_pleasure": "#E3E3E3",
        "bg_safety": "#E9E9E9",
        "bg_comfort": "#EEEEEE",
        "size_legend": ["#636363", "#969696", "#cccccc", "#252525"],
    },
    17: {
        "pleasure": "#542788",
        "safety": "#807dba",
        "comfort": "#b2abd2",
        "bg_pleasure": "#E7E4ED",
        "bg_safety": "#EBECF3",
        "bg_comfort": "#F2F1F7",
        "size_legend": ["#807dba", "#b2abd2", "#d8daeb", "#542788"],
    },
    18: {
        "pleasure": "#8c510a",
        "safety": "#01665e",
        "comfort": "#bf812d",
        "bg_pleasure": "#EFE7E0",
        "bg_safety": "#DFF0EF",
        "bg_comfort": "#F5EBE1",
        "size_legend": ["#01665e", "#bf812d", "#f6e8c3", "#8c510a"],
    },
    19: {
        "pleasure": "#c51b7d",
        "safety": "#4d9221",
        "comfort": "#e9a3c9",
        "bg_pleasure": "#F7E2EE",
        "bg_safety": "#E6EFE2",
        "bg_comfort": "#FBEFF8",
        "size_legend": ["#4d9221", "#e9a3c9", "#a1d76a", "#c51b7d"],
    },
    20: {
        "pleasure": "#a1c9f4",
        "safety": "#ffb482",
        "comfort": "#8de5a1",
        "bg_pleasure": "#EEF5FD",
        "bg_safety": "#FFF3E8",
        "bg_comfort": "#EBFBEE",
        "size_legend": ["#ffb482", "#8de5a1", "#ff9f9b", "#a1c9f4"],
    },
}
SELECTED_SCHEME = 1  # 选择配色方案
palette = COLOR_LIBRARY.get(SELECTED_SCHEME, COLOR_LIBRARY[1])  # 获取配色方案


# =========================================================================================
# ====================================== 3.绘图函数=========================================
# =========================================================================================
def create_bubble_plot(
    y_labels,  # Y轴标签
    x_values,  # X轴数值
    colors,  # 气泡的颜色
    category_legend_items,  # 类别图例项
    size_legend_labels,  # 尺寸图例的标签
    size_legend_radii,  # 尺寸图例的半径
    size_legend_colors,  # 尺寸图例的颜色
    bg_color_comfort,  # 背景色
    bg_color_safety,  # 背景色
    bg_color_pleasure,  # 背景色
    title="Standardized regression coefficients",  # 标题
):
    y_pos = np.arange(len(y_labels))  # 生成Y轴的位置
    sizes = [v * 1500 for v in x_values]  # 根据X轴数值, 计算每个气泡的面积
    fig, ax = plt.subplots(figsize=(12, 8))  # 创建一个新的图表和一个坐标轴
    x_start = 0  # 背景矩形的起始X坐标
    plot_width = 0.85  # 背景矩形的宽度
    # 绘制的背景矩形
    rect3 = patches.Rectangle(
        (x_start, -0.5), (plot_width), 3.0, color=bg_color_pleasure, zorder=1, alpha=0.6
    )
    ax.add_patch(rect3)  # 将该矩形添加到坐标轴上
    # 绘制的背景矩形
    rect2 = patches.Rectangle(
        (x_start, 2.5), (plot_width), 3.0, color=bg_color_safety, zorder=1, alpha=0.6
    )
    ax.add_patch(rect2)  # 将该矩形添加到坐标轴上
    # 绘制的背景矩形
    rect1 = patches.Rectangle(
        (x_start, 5.5), (plot_width), 3.0, color=bg_color_comfort, zorder=1, alpha=0.6
    )
    ax.add_patch(rect1)  # 将该矩形添加到坐标轴上

    # 在 x=0 的位置绘制一条垂直的黑线
    ax.axvline(x=0, color="black", linewidth=1.5, zorder=2)

    # 绘制连接虚线和数值标签
    for i in range(len(y_pos)):  # 遍历每一个数据点
        value = x_values[i]  # 获取当前数据点的值
        y = y_pos[i]  # 获取当前数据点的Y轴位置，就是行
        color = colors[i]  # 获取当前数据点的颜色
        x_center = value / 2  # 计算中点, 用于标注数值
        text_gap = 0.035  # 设置标注数值左右的间隙大小
        # 在中点位置添加数值标注
        ax.text(
            x_center,
            y,
            f"{value:.3f}",
            va="center",
            ha="center",
            fontsize=12,
            color=color,
            alpha=0.8,
            zorder=4,
        )
        # 绘制从数值左侧虚线
        ax.hlines(
            y=y,
            xmin=0,
            xmax=x_center - text_gap,
            color=color,
            linestyles="dashed",
            lw=1.5,
            zorder=3,
            alpha=0.7,
        )
        # 绘制数值右侧虚线
        ax.hlines(
            y=y,
            xmin=x_center + text_gap,
            xmax=value,
            color=color,
            linestyles="dashed",
            lw=1.5,
            zorder=3,
            alpha=0.7,
        )
    # 绘制气泡
    ax.scatter(
        x_values, y_pos, s=sizes, c=colors, alpha=0.9, zorder=5, edgecolors="white", linewidth=2
    )
    # 添加矩形区域的标注
    ax.text(
        0.65, 7, "Model 1\nOCV", ha="left", va="center", fontsize=16, fontweight="bold", zorder=10
    )
    ax.text(
        0.65, 4, "Model 2\nOSV", ha="left", va="center", fontsize=16, fontweight="bold", zorder=10
    )
    ax.text(
        0.65, 1, "Model 3\nOPV", ha="left", va="center", fontsize=16, fontweight="bold", zorder=10
    )

    ax.set_yticks(y_pos)  # 设置Y轴的刻度位置
    ax.set_yticklabels([])  # 去掉Y轴的刻度数字
    ax.tick_params(axis="y", length=0)  # 去掉Y轴的刻度线
    ax.set_ylabel("Independent variable", fontsize=16, fontweight="bold", labelpad=30)  # Y轴的标题

    for i in range(len(y_pos)):  # 遍历Y轴上的每个位置
        ax.text(
            -0.03,
            y_pos[i],
            y_labels[i],
            va="center",
            ha="right",
            fontsize=14,
            fontweight="bold",
            color=colors[i],
        )  # 手动添加Y轴标签文本

    x_tick_values = [0.0, 0.2, 0.4, 0.6, 0.8]  # 定义X轴上要显示的刻度值
    ax.set_xticks(x_tick_values)  # 设置X轴的刻度位置
    ax.set_xticklabels([f"{val:.1f}" for val in x_tick_values])  # 设置X轴刻度数值的显示格式
    ax.set_xlabel("Coefficient Value", fontsize=16, fontweight="bold", labelpad=15)  # X轴标题
    ax.tick_params(axis="x", labelsize=16, length=4, width=1.5)  # X轴刻度标注的样式
    ax.set_xlim(-0.12, 0.85)  # X轴的显示范围
    ax.set_ylim(-0.5, 8.5)  # Y轴的显示范围
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)  # 设置标题
    # 设置图框
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)  # 可见
        ax.spines[spine].set_color("black")  # 黑色
        ax.spines[spine].set_linewidth(1.5)  # 线宽

    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)  # 调整子图画布中的位置

    # --- 绘制图例 ---
    ax_legend = fig.add_axes([0.8, 0.1, 0.2, 0.8])  # 在右侧添加一个新的坐标轴用于绘制图例
    ax_legend.set_axis_off()  # 去掉这个新坐标轴的所有坐标轴元素 (线、刻度、标签)
    ax_legend.set_xlim(0, 1)  # 设置图例坐标轴的X范围
    ax_legend.set_ylim(0, 1)  # 设置图例坐标轴的Y范围
    ax_legend.set_aspect("equal")  # 设置图例坐标轴的X和Y轴等比例

    x_center = 0.5  # 定义尺寸图例的中心X坐标，就是那个半圆形图例
    y_bottom_edge = 0.5  # 定义尺寸图例的底部Y坐标

    for i in range(len(size_legend_radii)):  # 遍历所有尺寸图例的半径
        r = size_legend_radii[i]  # 当前半径
        color = size_legend_colors[i]  # 当前颜色
        y_center_current = y_bottom_edge + r  # 当前半圆的Y轴圆心

        wedge = patches.Wedge(  # 创建一个半圆形
            (x_center, y_center_current),  # 圆心坐标
            r,  # 半径
            90,  # 起始角度
            270,  # 结束角度
            facecolor=color,  # 填充色
            transform=ax_legend.transData,  # 数据坐标系
        )
        ax_legend.add_patch(wedge)  # 添加到图例坐标轴

    # 绘制指示线和文本
    for i in range(len(size_legend_radii)):
        r = size_legend_radii[i]  # 当前半径
        label = size_legend_labels[i]  # 当前标签
        y_line = y_bottom_edge + 2 * r  # 计算指示线的Y坐标
        ax_legend.plot(
            [x_center, x_center + 0.02], [y_line, y_line], color="black", lw=1
        )  # 绘制一条短的水平线
        ax_legend.text(
            x_center + 0.04, y_line, label, va="center", ha="left", fontsize=10
        )  # 添加文本

    # 绘制类别图例，就是那个圆加虚线
    y_base = 0.35  # 类别图例的起始Y坐标，最上面
    y_step = 0.1  # 每个类别图例项之间的高度差
    x_text = 0.6  # 文本的X坐标
    x_marker = 0.4  # 圆点的X坐标
    x_line1_start = 0.2  # 左侧虚线的起始X坐标
    x_line1_end = 0.3  # 左侧虚线的结束X坐标
    x_line2_start = 0.3  # 右侧虚线的起始X坐标
    x_line2_end = x_text - 0.02  # 右侧虚线的结束X坐标
    for i, (label, color) in enumerate(reversed(category_legend_items)):
        y = y_base - i * y_step  # 当前图例项的Y坐标
        ax_legend.plot(
            [x_line1_start, x_line1_end], [y, y], linestyle="--", color=color, lw=2
        )  # 绘制左侧虚线
        ax_legend.plot(x_marker, y, "o", markersize=15, color=color)  # 绘制中间的圆点标记
        ax_legend.plot(
            [x_line2_start, x_line2_end], [y, y], linestyle="--", color=color, lw=2
        )  # 绘制右侧虚线
        ax_legend.text(x_text, y, label, va="center", ha="left", fontsize=16)  # 添加文本
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"{SELECTED_SCHEME}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"{SELECTED_SCHEME}.pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# ====================================== 4.数据准备=========================================
# =========================================================================================
plot_y_labels = ["NPV", "CPV", "SPV", "PSV", "DSV", "ASV", "VCV", "ACV", "TCV"]  # Y轴
plot_x_values = [0.587, 0.133, 0.218, 0.375, 0.294, 0.265, 0.304, 0.216, 0.503]  # X轴数值

# =========================================================================================
# ====================================== 5.绘图参数设置=========================================
# =========================================================================================
plot_size_labels = ["0.9", "0.6", "0.35", "0.1"]  # 尺寸图例的文本标注
# 图例标签对应的实际数据值
legend_data_values = np.array([0.9, 0.6, 0.35, 0.1])
# 图例中的半圆的大小
s_values_in_main_plot = legend_data_values * 1500
r_visual_in_points = np.sqrt(s_values_in_main_plot / np.pi)  # 根据面积反推半径
# 计算图例坐标轴的实际宽度
legend_ax_width_points = (12 * 0.2) * 72.0
# 将points单位的半径转换回图例坐标轴的数据坐标系下的半径
plot_size_radii = r_visual_in_points / legend_ax_width_points
# 提取图例、气泡颜色
color_pleasure = palette["pleasure"]
color_safety = palette["safety"]
color_comfort = palette["comfort"]
# 提取背景色
bg_pleasure = palette["bg_pleasure"]
bg_safety = palette["bg_safety"]
bg_comfort = palette["bg_comfort"]
# 尺寸图例颜色
plot_size_colors = palette["size_legend"]
# 生成气泡颜色列表
plot_colors = [color_pleasure] * 3 + [color_safety] * 3 + [color_comfort] * 3
# 生成类别图例的名称、颜色
plot_category_items = [
    ("Comfort", color_comfort),
    ("Safety", color_safety),
    ("Pleasure", color_pleasure),
]
# =========================================================================================
# ====================================== 6.执行绘图=========================================
# =========================================================================================
create_bubble_plot(
    y_labels=plot_y_labels,  # Y轴标签
    x_values=plot_x_values,  # X轴数值
    colors=plot_colors,  # 气泡颜色列表
    category_legend_items=plot_category_items,  # 类别图例项
    size_legend_labels=plot_size_labels,  # 尺寸图例标签
    size_legend_radii=plot_size_radii,  # 尺寸图例半径
    size_legend_colors=plot_size_colors,  # 尺寸图例颜色
    bg_color_comfort=bg_comfort,  # 背景色
    bg_color_safety=bg_safety,  # 背景色
    bg_color_pleasure=bg_pleasure,  # 背景色
    title="Standardized regression coefficients",  # 标题
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

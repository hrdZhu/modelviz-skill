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
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ("RdPu", "GnBu"),
    2: ("Oranges", "Blues"),
    3: ("Reds", "Greens"),
    4: ("Purples", "Greens"),
    5: ("YlOrBr", "PuBu"),
    6: ("Reds", "Blues"),
    7: ("OrRd", "PuBuGn"),
    8: ("PuRd", "YlGn"),
    9: ("BuPu", "YlGnBu"),
    10: ("YlOrRd", "BuGn"),
    11: ("Greys", "Reds"),
    12: ("autumn", "winter"),
    13: ("pink", "bone"),
    14: ("hot", "cool"),
    15: ("RdPu", "PuBu"),
    16: ("YlOrBr", "Greens"),
    17: ("Wistia", "YlGnBu_r"),
    18: ("plasma", "viridis"),
    19: ("summer", "copper"),
    20: ("afmhot", "ocean"),
}
SELECTED_SCHEME = 1


# =========================================================================================
# ====================================== 3.绘图函数=========================================
# =========================================================================================
def plot_and_save_heatmap(
    regions, data_pluvial, data_drought, idx_highlight_row, idx_highlight_col, scheme_index=1
):
    # 获取区域数量，用于确定网格大小
    n = len(regions)
    # 定义颜色的分级数量
    n_steps = 10
    # 获取配色
    cmap_d_name, cmap_p_name = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])
    # 加载下三角的颜色映射
    cmap_drought = matplotlib.colormaps[cmap_d_name].resampled(n_steps)
    # 加载上三角的颜色映射
    cmap_pluvial = matplotlib.colormaps[cmap_p_name].resampled(n_steps)
    # 创建一个图形
    fig = plt.figure(figsize=(12, 9))

    # 在指定位置添加主图，左,下,宽,高
    ax_main = fig.add_axes([0.12, 0.15, 0.65, 0.75])
    # X轴范围
    ax_main.set_xlim(0, n)
    # Y轴范围
    ax_main.set_ylim(0, n)
    # 设置纵横比
    ax_main.set_aspect("equal")
    # 反转Y轴
    ax_main.invert_yaxis()
    # 绘图
    for row in range(n):
        for col in range(n):
            # 只绘制下三角部分
            if row >= col:
                # 获取对应位置的数据
                val_tl = data_pluvial[row, col]
                # 根据数值从颜色映射中获取对应的颜色
                color_tl = cmap_pluvial(val_tl)
                # 上三角形
                triangle_tl = patches.Polygon(
                    [(col, row), (col + 1, row), (col, row + 1)],  # 顶点坐标
                    closed=True,  # 闭合图形
                    color=color_tl,  # 填充颜色
                    ec="white",  # 边框颜色
                    lw=0.5,  # 边框宽度
                )
                # 将上三角形添加到主坐标轴
                ax_main.add_patch(triangle_tl)
                # 获取对应位置的数据值
                val_br = data_drought[row, col]
                # 根据数值从颜色映射中获取对应的颜色
                color_br = cmap_drought(val_br)
                # 下三角形
                triangle_br = patches.Polygon(
                    [(col + 1, row), (col + 1, row + 1), (col, row + 1)],  # 顶点坐标
                    closed=True,  # 闭合图形
                    color=color_br,  # 填充颜色
                    ec="white",  # 框颜色
                    lw=0.5,  # 边框宽度
                )
                # 将下三角形添加到主坐标轴
                ax_main.add_patch(triangle_br)
                # 判断当前格子是否为指定的标记位置
                if row == idx_highlight_row and col == idx_highlight_col:
                    # 创建黑色方框
                    rect = patches.Rectangle(
                        (col, row),  # 左下角坐标
                        1,
                        1,  # 宽度和高度
                        linewidth=4,  # 线宽
                        edgecolor="black",  # 边框颜色
                        facecolor="none",  # 填充颜色
                        zorder=10,  # 图层顺序
                    )
                    # 添加到主坐标轴
                    ax_main.add_patch(rect)
    # X轴刻度位置
    ax_main.set_xticks(np.arange(n) + 0.5)
    # Y轴刻度位置
    ax_main.set_yticks(np.arange(n) + 0.5)
    # X轴刻度标签
    ax_main.set_xticklabels(regions, rotation=45, ha="right", fontsize=12)
    # Y轴刻度标签
    ax_main.set_yticklabels(regions, fontsize=12)
    # 图框
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.spines["left"].set_visible(True)
    ax_main.spines["bottom"].set_visible(True)
    for spine in ax_main.spines.values():
        spine.set_linewidth(2)
    # 刻度线
    ax_main.tick_params(
        axis="both",
        which="both",
        length=4,
        width=2,
    )

    # Y轴标题
    ax_main.set_ylabel("Pluvial", fontsize=16, labelpad=10)
    ax_main.set_xlabel("Drought", fontsize=16, labelpad=10)
    # 主标题
    ax_main.set_title("Pluvial-drought synchronization", fontsize=18, loc="left", pad=20)
    # 标记注释
    # 坐标
    target_x = idx_highlight_col + 1
    target_y = idx_highlight_row
    # 添加注释
    ax_main.annotate(
        "Detailed in\n      " + r"$\bf{c}$",  # 注释文本
        xy=(target_x, target_y),  # 箭头坐标
        xytext=(target_x + 2.5, target_y - 2),  # 文本坐标
        arrowprops=dict(
            arrowstyle="->, head_width=0.4, head_length=0.8",  # 箭头样式
            connectionstyle="arc3,rad=-0.3",  # 连接线弯曲程度
            lw=2,
            color="black",
        ),  # 线宽和颜色
        fontsize=14,
        ha="center",
    )

    # 颜色条
    cbar_gap = 0.02  # 主图与颜色条之间间隔
    cbar_width = 0.03  # 单个颜色条的宽度

    cbar_bottom = 0.15  # 颜色条底部位置
    cbar_height = 0.75  # 颜色条高度
    # 颜色条的起始X坐标
    cbar_left_x = 0.12 + 0.65 + cbar_gap
    # 创建垂直渐变数据
    gradient = np.linspace(0, 1, n_steps).reshape(-1, 1)

    # 绘制左侧颜色条
    # 添加子坐标轴用于绘制颜色条
    ax_cb_left = fig.add_axes([cbar_left_x, cbar_bottom, cbar_width, cbar_height])
    # 绘制图像
    ax_cb_left.imshow(gradient, aspect="auto", cmap=cmap_drought, origin="lower")
    # X轴刻度
    ax_cb_left.set_xticks([])
    # Y轴刻度
    ax_cb_left.set_yticks([])
    # 图框
    for spine in ax_cb_left.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)
    # 绘制右侧颜色条
    # 添加子坐标轴
    ax_cb_right = fig.add_axes([cbar_left_x + cbar_width, cbar_bottom, cbar_width, cbar_height])
    # 绘制图像
    ax_cb_right.imshow(
        gradient, aspect="auto", cmap=cmap_pluvial, origin="lower", extent=[0, 1, 0, 1]
    )
    # X轴刻度
    ax_cb_right.set_xticks([])
    # 右侧颜色条的刻度
    ax_cb_right.yaxis.tick_right()
    # 设置刻度数值位置
    ax_cb_right.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # 刻度标签文本
    ax_cb_right.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=12)
    # 为颜色条添加细黑色边框
    for spine in ax_cb_right.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)
    # 图框
    ax_cb_right.set_ylabel("Area fraction (-)", rotation=270, labelpad=-60, fontsize=15)
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"heatmap_visualization_{SELECTED_SCHEME}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(OUTPUT_DIR / f"heatmap_visualization_{SELECTED_SCHEME}.pdf"), bbox_inches="tight"
    )


# =========================================================================================
# ====================================== 4.执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    excel_path = str(DATA_DIR / "simulation_data.xlsx")
    # 读取数据
    df_pluvial = pd.read_excel(excel_path, sheet_name="Pluvial_Data", index_col=0)
    df_drought = pd.read_excel(excel_path, sheet_name="Drought_Data", index_col=0)
    # 获取索引列表作为区域名称
    regions = df_pluvial.index.tolist()
    # 读取数值
    data_pluvial = df_pluvial.values
    data_drought = df_drought.values
    # 标记行索引
    idx_mexico = regions.index("Mexico")
    idx_west_usa = regions.index("West USA")
    # 调用封装好的绘图函数进行绘制和保存
    plot_and_save_heatmap(
        regions,  # 区域名称列表
        data_pluvial,  # 上三角数据
        data_drought,  # 下三角数据
        idx_highlight_row=idx_mexico,  # 标记行索引
        idx_highlight_col=idx_west_usa,  # 标记行索引
        scheme_index=SELECTED_SCHEME,  # 配色方案
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

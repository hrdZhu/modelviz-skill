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
# ======================================1.库的导入=========================================
# =========================================================================================
import os

import matplotlib
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from skbio.stats.distance import mantel

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
# =========================================================================================
# ======================================2.颜色库=========================================
# =========================================================================================
color_library = {
    1: {
        "heatmap_negative": "#b2182b",
        "heatmap_zero": "#ffffbf",
        "heatmap_positive": "#2166ac",
        "nodes": "#b2182b",
        "mantel_p_high_sig": "#6cae3e",
        "mantel_p_low_sig": "#7d61aa",
        "mantel_p_no_sig": "#cccccc",
    },
    2: {
        "heatmap_negative": "#8e0152",
        "heatmap_zero": "#f7f7f7",
        "heatmap_positive": "#276419",
        "nodes": "#4d4d4d",
        "mantel_p_high_sig": "#1f78b4",
        "mantel_p_low_sig": "#ff7f00",
        "mantel_p_no_sig": "#cccccc",
    },
    3: {
        "heatmap_negative": "#d95f02",
        "heatmap_zero": "#ffffff",
        "heatmap_positive": "#7570b3",
        "nodes": "#d95f02",
        "mantel_p_high_sig": "#1b9e77",
        "mantel_p_low_sig": "#e7298a",
        "mantel_p_no_sig": "#cccccc",
    },
    4: {
        "heatmap_negative": "#8c510a",
        "heatmap_zero": "#f5f5f5",
        "heatmap_positive": "#01665e",
        "nodes": "#8c510a",
        "mantel_p_high_sig": "#762a83",
        "mantel_p_low_sig": "#af8dc3",
        "mantel_p_no_sig": "#cccccc",
    },
    5: {
        "heatmap_negative": "#c51b7d",
        "heatmap_zero": "#e7d4e8",
        "heatmap_positive": "#4d9221",
        "nodes": "#4d9221",
        "mantel_p_high_sig": "#d6604d",
        "mantel_p_low_sig": "#f4a582",
        "mantel_p_no_sig": "#cccccc",
    },
    6: {
        "heatmap_negative": "#d7191c",
        "heatmap_zero": "#ffffbf",
        "heatmap_positive": "#1a9641",
        "nodes": "#2c7bb6",
        "mantel_p_high_sig": "#fdae61",
        "mantel_p_low_sig": "#abdda4",
        "mantel_p_no_sig": "#cccccc",
    },
    7: {
        "heatmap_negative": "#4575b4",
        "heatmap_zero": "#e0f3f8",
        "heatmap_positive": "#d73027",
        "nodes": "#313695",
        "mantel_p_high_sig": "#fee090",
        "mantel_p_low_sig": "#fdae61",
        "mantel_p_no_sig": "#cccccc",
    },
    8: {
        "heatmap_negative": "#a6611a",
        "heatmap_zero": "#f5f5f5",
        "heatmap_positive": "#018571",
        "nodes": "#543005",
        "mantel_p_high_sig": "#bf812d",
        "mantel_p_low_sig": "#80cdc1",
        "mantel_p_no_sig": "#cccccc",
    },
    9: {
        "heatmap_negative": "#7b3294",
        "heatmap_zero": "#f7f7f7",
        "heatmap_positive": "#008837",
        "nodes": "#e66101",
        "mantel_p_high_sig": "#fdb863",
        "mantel_p_low_sig": "#b2abd2",
        "mantel_p_no_sig": "#cccccc",
    },
    10: {
        "heatmap_negative": "#543005",
        "heatmap_zero": "#f6e8c3",
        "heatmap_positive": "#003c30",
        "nodes": "#35978f",
        "mantel_p_high_sig": "#a6611a",
        "mantel_p_low_sig": "#c7eae5",
        "mantel_p_no_sig": "#cccccc",
    },
    11: {
        "heatmap_negative": "#b35806",
        "heatmap_zero": "#f7f7f7",
        "heatmap_positive": "#542788",
        "nodes": "#8073ac",
        "mantel_p_high_sig": "#9970ab",
        "mantel_p_low_sig": "#c2a5cf",
        "mantel_p_no_sig": "#cccccc",
    },
    12: {
        "heatmap_negative": "#e41a1c",
        "heatmap_zero": "#ffffcd",
        "heatmap_positive": "#377eb8",
        "nodes": "#000000",
        "mantel_p_high_sig": "#4daf4a",
        "mantel_p_low_sig": "#984ea3",
        "mantel_p_no_sig": "#cccccc",
    },
    13: {
        "heatmap_negative": "#440154",
        "heatmap_zero": "#21908d",
        "heatmap_positive": "#fde725",
        "nodes": "#440154",
        "mantel_p_high_sig": "#35b779",
        "mantel_p_low_sig": "#5dc863",
        "mantel_p_no_sig": "#cccccc",
    },
    14: {
        "heatmap_negative": "#000004",
        "heatmap_zero": "#b9327f",
        "heatmap_positive": "#fcffa4",
        "nodes": "#57106e",
        "mantel_p_high_sig": "#e65d3a",
        "mantel_p_low_sig": "#fca636",
        "mantel_p_no_sig": "#cccccc",
    },
    15: {
        "heatmap_negative": "#4a1486",
        "heatmap_zero": "#f2f2f2",
        "heatmap_positive": "#ffd700",
        "nodes": "#0d47a1",
        "mantel_p_high_sig": "#b71c1c",
        "mantel_p_low_sig": "#f48fb1",
        "mantel_p_no_sig": "#cccccc",
    },
    16: {
        "heatmap_negative": "#8b4513",
        "heatmap_zero": "#ffdead",
        "heatmap_positive": "#4682b4",
        "nodes": "#daa520",
        "mantel_p_high_sig": "#ff4500",
        "mantel_p_low_sig": "#2e8b57",
        "mantel_p_no_sig": "#cccccc",
    },
    17: {
        "heatmap_negative": "#fb6f92",
        "heatmap_zero": "#ffc1d4",
        "heatmap_positive": "#83c5be",
        "nodes": "#ff6b6b",
        "mantel_p_high_sig": "#55a630",
        "mantel_p_low_sig": "#aacc00",
        "mantel_p_no_sig": "#cccccc",
    },
    18: {
        "heatmap_negative": "#003f5c",
        "heatmap_zero": "#f1f1f1",
        "heatmap_positive": "#ff6361",
        "nodes": "#003f5c",
        "mantel_p_high_sig": "#58508d",
        "mantel_p_low_sig": "#ffa600",
        "mantel_p_no_sig": "#cccccc",
    },
    19: {
        "heatmap_negative": "#ff8fab",
        "heatmap_zero": "#ffffff",
        "heatmap_positive": "#72ddf7",
        "nodes": "#f7aef8",
        "mantel_p_high_sig": "#b2f7ef",
        "mantel_p_low_sig": "#f0f4c3",
        "mantel_p_no_sig": "#cccccc",
    },
    20: {
        "heatmap_negative": "#03045e",
        "heatmap_zero": "#90e0ef",
        "heatmap_positive": "#ffd60a",
        "nodes": "#00b4d8",
        "mantel_p_high_sig": "#f8f9fa",
        "mantel_p_low_sig": "#e9ecef",
        "mantel_p_no_sig": "#cccccc",
    },
}

COLOR_CHOICE = 10
selected_colors = color_library[COLOR_CHOICE]


# =========================================================================================
# ======================================3.绘图函数=========================================
# =========================================================================================
def create_and_save_plot(
    left_vars, right_vars, corr_matrix, corr_p_values, mantel_r, mantel_p, output_filename, colors
):
    n_left = len(left_vars)  # 相关性变量
    n_right = len(right_vars)  # mantel变量
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")
    grid_line_width = 1.0  # 定义热图网格线的宽度
    grid_line_color = "black"  # 定义热图网格线的颜色
    # 绘制左侧的垂直边框线
    ax.plot([0, 0], [1, n_left], color=grid_line_color, linewidth=grid_line_width, zorder=1.5)
    # 绘制下面的水平边框线
    ax.plot(
        [0, n_left - 1],
        [n_left, n_left],
        color=grid_line_color,
        linewidth=grid_line_width,
        zorder=1.5,
    )
    # 绘制热图内部的水平网格线
    for i in range(1, n_left):
        ax.plot([0, i], [i, i], color=grid_line_color, linewidth=grid_line_width, zorder=1.5)
    # 绘制热图内部的垂直网格线
    for j in range(1, n_left):
        ax.plot([j, j], [j, n_left], color=grid_line_color, linewidth=grid_line_width, zorder=1.5)
    # 创建一个归一化对象，将相关性值 (-1 到 1) 映射到 0-1 的范围
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    # 计算 0 点在归一化后的位置，用于定义发散型颜色映射的中心
    zero_point_norm = (0 - (-1)) / (1 - (-1))
    # 创建一个颜色映射
    corr_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_corr_cmap",
        [
            (0, colors["heatmap_negative"]),
            (zero_point_norm, colors["heatmap_zero"]),
            (1, colors["heatmap_positive"]),
        ],
    )
    # 遍历下三角矩阵（不包括对角线）
    for i in range(n_left):
        for j in range(i):
            # 获取皮尔逊相关系数
            corr_val = corr_matrix[i, j]
            # 根据相关系数的绝对值计算矩形的大小
            size = abs(corr_val) * 1
            # 根据相关系数的值获取对应的颜色
            color = corr_cmap(norm(corr_val))
            # 计算矩形的左下角坐标，使其居中
            bottom_left_x = j + 0.5 - size / 2
            bottom_left_y = i + 0.5 - size / 2
            # 创建一个矩形用来表现相关性
            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y),
                size,
                size,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
            # 将矩形添加到坐标轴上
            ax.add_patch(rect)
            # 获取对应的p值
            p_val = corr_p_values[i, j]
            # 根据p值确定显著性标记
            sig_marker = ""
            if p_val < 0.001:
                sig_marker = "***"
            elif p_val < 0.01:
                sig_marker = "**"
            elif p_val < 0.05:
                sig_marker = "*"
            # 在矩形中心添加显著性标记
            ax.text(
                j + 0.5,
                i + 0.5,
                sig_marker,
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                zorder=3,
            )  # zorder=3 使其在最顶层
    # 定义左侧节点的坐标（与热图对角线对应）
    left_node_coords = [(i + 0.5, i + 0.5) for i in range(n_left)]
    # 控制右侧点的左右位置,值越大，离左侧热图越远。
    start_x = n_left - 3
    # 控制第一个右侧节点的上下位置。决定了整组节点的起始高度。
    start_y = 0
    # 控制右侧点之间的间距。值越大，越稀疏；值越小，越密集。
    step = 4
    x_step = step
    y_step = step
    # 根据您上面设置的三个参数，计算出所有右侧节点的最终坐标
    right_node_coords = [(start_x + i * x_step, start_y + i * y_step) for i in range(n_right)]
    # 提取左右两侧节点的 x, y 坐标
    left_x, left_y = zip(*left_node_coords)
    right_x, right_y = zip(*right_node_coords)
    # 绘制左侧节点
    ax.scatter(
        left_x, left_y, s=150, facecolor=colors["nodes"], edgecolor="black", linewidth=1, zorder=10
    )
    # 绘制右侧节点
    ax.scatter(
        right_x,
        right_y,
        s=150,
        facecolor=colors["nodes"],
        edgecolor="black",
        linewidth=1,
        zorder=10,
    )
    # 为右侧节点添加文本标签
    for i, name in enumerate(right_vars):
        ax.text(
            right_node_coords[i][0] + 0.5,
            right_node_coords[i][1],
            name,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
    # 遍历所有左侧和右侧变量的组合，绘制它们之间的连线
    for i in range(n_left):
        for j in range(n_right):
            # 获取 Mantel检验的p值和r值
            p_val = mantel_p[i, j]
            r_val = mantel_r[i, j]
            # 根据 Mantel p 值设置连线颜色
            if p_val < 0.01:
                edge_color = colors["mantel_p_high_sig"]
            elif p_val <= 0.05:
                edge_color = colors["mantel_p_low_sig"]
            else:
                edge_color = colors["mantel_p_no_sig"]
            # 根据Mantel r值的绝对值设置连线宽度
            if abs(r_val) < 0.1:
                edge_width = 1.0
            elif abs(r_val) <= 0.2:
                edge_width = 2.5
            else:
                edge_width = 4.0
            # 绘制节点间的连线
            con = patches.ConnectionPatch(
                xyA=left_node_coords[i],
                xyB=right_node_coords[j],  # A点和B点的坐标
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,  # 坐标系统
                arrowstyle="-",
                connectionstyle="arc3,rad=0.1",  # 连线样式
                color=edge_color,
                linewidth=edge_width,
                zorder=5,  # 连线属性
            )
            # 将连线添加到坐标轴上
            ax.add_patch(con)
    # 设置 X 轴刻度位置和标签
    ax.set_xticks(np.arange(n_left) + 0.5)
    ax.set_xticklabels(left_vars, fontsize=14, rotation=90)
    # 设置 Y 轴刻度位置和标签
    ax.set_yticks(np.arange(n_left) + 0.5)
    ax.set_yticklabels(left_vars, fontsize=14)
    # 调整 X 轴刻度标签的位置，使其更靠近热图
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, pad=-15)
    # 调整 Y 轴刻度标签的位置，使其更靠近热图
    ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False, pad=-25)
    # 翻转 Y 轴，使热图的原点 (0,0) 位于左上角
    ax.invert_yaxis()
    # 设置坐标轴的宽高比为 'equal'，确保热图单元格是正方形
    ax.set_aspect("equal", adjustable="box")
    # 设置 X 轴和 Y 轴的显示范围，以适应所有绘图元素
    ax.set_xlim(-1, right_node_coords[-1][0] + 3)
    ax.set_ylim(n_left + 0.5, -0.5)
    # 隐藏图表的上下左右四条边框线
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    # 隐藏所有刻度线
    ax.tick_params(axis="both", which="both", length=0)
    # 在图的指定位置添加一个新的坐标轴用于颜色条
    cbar_ax = fig.add_axes([0.02, 0.2, 0.015, 0.3])
    # 在新坐标轴上创建颜色条
    cb = colorbar.ColorbarBase(cbar_ax, cmap=corr_cmap, norm=norm, orientation="vertical")
    # 设置颜色条的标签
    cb.set_label("Pearson's r", size=14, labelpad=15)
    # 设置颜色条刻度标签的字体大小
    cb.ax.tick_params(labelsize=12)
    # 设置颜色条的刻度
    cb.set_ticks([-1, -0.5, 0.0, 0.5, 1])
    # 创建 Mantel p 的图例元素
    p_legend_elements = [
        Line2D([0], [0], color=colors["mantel_p_high_sig"], lw=4, label="< 0.01"),
        Line2D([0], [0], color=colors["mantel_p_low_sig"], lw=4, label="0.01 - 0.05"),
        Line2D([0], [0], color=colors["mantel_p_no_sig"], lw=4, label=">= 0.05"),
    ]
    # 创建第一个图例（p值图例）并设置其位置和样式
    legend1 = ax.legend(
        handles=p_legend_elements,
        title="Mantel's p",
        loc="upper left",
        bbox_to_anchor=(-0.15, 0.85),
        fontsize=12,
        title_fontsize=14,
        frameon=False,
    )
    # 将第一个图例添加到坐标轴上，防止被第二个图例覆盖
    ax.add_artist(legend1)
    # 创建 Mantel r值的图例元素
    r_legend_elements = [
        Line2D([0], [0], color="grey", lw=1.0, label="< 0.1"),
        Line2D([0], [0], color="grey", lw=2.5, label="0.1 - 0.2"),
        Line2D([0], [0], color="grey", lw=4.0, label=">= 0.2"),
    ]
    # 创建第二个图例（r值图例）并设置其位置和样式
    ax.legend(
        handles=r_legend_elements,
        title="Mantel's r (abs)",
        loc="upper left",
        bbox_to_anchor=(-0.15, 0.7),
        fontsize=12,
        title_fontsize=14,
        frameon=False,
    )
    # 保存图像
    output_dir = str(OUTPUT_DIR)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    png_path = os.path.join(output_dir, f"{output_filename}_{COLOR_CHOICE}.png")
    pdf_path = os.path.join(output_dir, f"{output_filename}_{COLOR_CHOICE}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"绘图已成功保存至 '{png_path}' 和 '{pdf_path}'")


# =========================================================================================
# ======================================4.数据提取、分析、绘图=========================================
# =========================================================================================
# 定义输入 Excel 文件的路径
excel_path = str(DATA_DIR / "simulated_data.xlsx")
# 使用 pandas 读取 Excel 文件
df_combined = pd.read_excel(excel_path)
# 提取数据
left_data = df_combined.iloc[:, 0:18].values
right_data = df_combined.iloc[:, 18:22].values
# 同时获取对应位置的列名，用于后续的绘图标签
left_vars = df_combined.columns[0:18].tolist()
right_vars = df_combined.columns[18:22].tolist()
# 获取左右两侧变量的数量
n_left = len(left_vars)
n_right = len(right_vars)
# 初始化一个 零矩阵，用于存储相关系数
corr_matrix = np.zeros((n_left, n_left))
# 存储 p 值
corr_p_values = np.zeros((n_left, n_left))
# 嵌套循环，计算每对变量之间的皮尔逊相关性
for i in range(n_left):
    for j in range(n_left):
        # 计算相关系数和p值
        r, p = pearsonr(left_data[:, i], left_data[:, j])
        # 将计算结果存入矩阵
        corr_matrix[i, j] = r
        corr_p_values[i, j] = p
print("相关性计算完成。")
# 进行左侧变量和右侧变量之间逐对的曼特尔检验
print("正在执行逐对曼特尔检验")
# 用于存储 Mantel r 值
mantel_r = np.zeros((n_left, n_right))
# 用于存储 Mantel p 值
mantel_p = np.zeros((n_left, n_right))
# 嵌套循环，对每一对左侧变量和右侧变量进行 Mantel 检验
for i in range(n_left):
    for j in range(n_right):
        # 提取当前循环中的左侧变量数据列
        var_left_col = left_data[:, i : i + 1]
        # 提取当前循环中的右侧变量数据列
        var_right_col = right_data[:, j : j + 1]
        # 计算左侧变量数据的欧几里得距离矩阵
        dist_matrix_left = squareform(pdist(var_left_col, metric="euclidean"))
        # 计算右侧变量数据的欧几里得距离矩阵
        dist_matrix_right = squareform(pdist(var_right_col, metric="euclidean"))
        # 执行 Mantel 检验，方法为皮尔逊，置换检验次数为 999
        r, p, _ = mantel(dist_matrix_left, dist_matrix_right, method="pearson", permutations=999)
        # 将检验结果存入矩阵
        mantel_r[i, j] = r
        mantel_p[i, j] = p
print("曼特尔检验完成。")
# 用封装好的绘图函数
create_and_save_plot(
    left_vars=left_vars,  # 左侧变量名列表
    right_vars=right_vars,  # 右侧变量名列表
    corr_matrix=corr_matrix,  # 相关系数矩阵
    corr_p_values=corr_p_values,  # 相关性p值矩阵
    mantel_r=mantel_r,  # Mantelr值矩阵
    mantel_p=mantel_p,  # Mantelp值矩阵
    output_filename="mantel_plot_from_local_data",  # 输出文件名
    colors=selected_colors,  # 配色
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

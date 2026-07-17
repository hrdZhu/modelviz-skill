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
# ====================================== 1. 环境设置 =======================================
# =========================================================================================
import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ====================================== 2. 颜色库设置 ==========================
# =========================================================================================
COLOR_LIBRARY = {
    1: {
        "heatmap_cmap": "RdBu",
        "network_cmap": "PRGn",
        "radar_cmap": "viridis",
        "node_color": "#333333",
        "ring_fill": "#f0f0f0",
        "ring_edge": "#bfbfbf",
        "grid_color": "gray",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",  # 默认红/蓝
    },
    2: {
        "heatmap_cmap": "PiYG",
        "network_cmap": "RdGy",
        "radar_cmap": "plasma",
        "node_color": "#1b4f72",
        "ring_fill": "#ebf5fb",
        "ring_edge": "#a9cce3",
        "grid_color": "#5499c7",
        "net_pos": "#c0392b",
        "net_neg": "#2980b9",  # 方案2：深红/深蓝
    },
    3: {
        "heatmap_cmap": "PuOr",
        "network_cmap": "BrBG",
        "radar_cmap": "magma",
        "node_color": "#4a235a",
        "ring_fill": "#f4ecf7",
        "ring_edge": "#d2b4de",
        "grid_color": "#884ea0",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    4: {
        "heatmap_cmap": "bwr",
        "network_cmap": "PRGn",
        "radar_cmap": "inferno",
        "node_color": "#641e16",
        "ring_fill": "#fdedec",
        "ring_edge": "#f5b7b1",
        "grid_color": "#cd6155",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    5: {
        "heatmap_cmap": "Spectral",
        "network_cmap": "RdBu",
        "radar_cmap": "cividis",
        "node_color": "#145a32",
        "ring_fill": "#e9f7ef",
        "ring_edge": "#a9dfbf",
        "grid_color": "#27ae60",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    6: {
        "heatmap_cmap": "coolwarm",
        "network_cmap": "PiYG",
        "radar_cmap": "twilight",
        "node_color": "#17202a",
        "ring_fill": "#eaeded",
        "ring_edge": "#d5dbdb",
        "grid_color": "#808b96",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    7: {
        "heatmap_cmap": "seismic",
        "network_cmap": "PuOr",
        "radar_cmap": "turbo",
        "node_color": "#1c2833",
        "ring_fill": "#f2f4f4",
        "ring_edge": "#bdc3c7",
        "grid_color": "#566573",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    8: {
        "heatmap_cmap": "BrBG",
        "network_cmap": "coolwarm",
        "radar_cmap": "cool",
        "node_color": "#0b5345",
        "ring_fill": "#e8f8f5",
        "ring_edge": "#a3e4d7",
        "grid_color": "#16a085",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    9: {
        "heatmap_cmap": "RdGy",
        "network_cmap": "Spectral",
        "radar_cmap": "spring",
        "node_color": "#78281f",
        "ring_fill": "#f9ebea",
        "ring_edge": "#f1948a",
        "grid_color": "#c0392b",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    10: {
        "heatmap_cmap": "YlGnBu",
        "network_cmap": "RdPu",
        "radar_cmap": "summer",
        "node_color": "#154360",
        "ring_fill": "#d4e6f1",
        "ring_edge": "#7fb3d5",
        "grid_color": "#2980b9",
        "net_pos": "#154360",
        "net_neg": "#c0392b",  # 方案10：根据YlGnBu风格，正相关设为深蓝，负相关设为深红
    },
    11: {
        "heatmap_cmap": "RdYlBu",
        "network_cmap": "BrBG",
        "radar_cmap": "autumn",
        "node_color": "#6e2c00",
        "ring_fill": "#f6ddcc",
        "ring_edge": "#e59866",
        "grid_color": "#d35400",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    12: {
        "heatmap_cmap": "RdYlGn",
        "network_cmap": "bwr",
        "radar_cmap": "winter",
        "node_color": "#0e6251",
        "ring_fill": "#d1f2eb",
        "ring_edge": "#73c6b6",
        "grid_color": "#1abc9c",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    13: {
        "heatmap_cmap": "PiYG",
        "network_cmap": "seismic",
        "radar_cmap": "Wistia",
        "node_color": "#7d6608",
        "ring_fill": "#fcf3cf",
        "ring_edge": "#f7dc6f",
        "grid_color": "#f1c40f",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    14: {
        "heatmap_cmap": "twilight_shifted",
        "network_cmap": "RdBu",
        "radar_cmap": "hot",
        "node_color": "#4a235a",
        "ring_fill": "#f5eef8",
        "ring_edge": "#d7bde2",
        "grid_color": "#8e44ad",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    15: {
        "heatmap_cmap": "ocean",
        "network_cmap": "afmhot",
        "radar_cmap": "copper",
        "node_color": "#1b2631",
        "ring_fill": "#d6eaf8",
        "ring_edge": "#85c1e9",
        "grid_color": "#3498db",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    16: {
        "heatmap_cmap": "terrain",
        "network_cmap": "gnuplot",
        "radar_cmap": "bone",
        "node_color": "#186a3b",
        "ring_fill": "#d5f5e3",
        "ring_edge": "#82e0aa",
        "grid_color": "#2ecc71",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    17: {
        "heatmap_cmap": "cubehelix",
        "network_cmap": "rainbow",
        "radar_cmap": "pink",
        "node_color": "#212f3c",
        "ring_fill": "#ebedef",
        "ring_edge": "#aec6cf",
        "grid_color": "#5d6d7e",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    18: {
        "heatmap_cmap": "rainbow",
        "network_cmap": "gist_ncar",
        "radar_cmap": "jet",
        "node_color": "#283747",
        "ring_fill": "#eaecee",
        "ring_edge": "#abb2b9",
        "grid_color": "#566573",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    19: {
        "heatmap_cmap": "hsv",
        "network_cmap": "nipy_spectral",
        "radar_cmap": "nipy_spectral",
        "node_color": "#17202a",
        "ring_fill": "#f0f3f4",
        "ring_edge": "#cacfd2",
        "grid_color": "#909497",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
    20: {
        "heatmap_cmap": "flag",
        "network_cmap": "prism",
        "radar_cmap": "prism",
        "node_color": "#641e16",
        "ring_fill": "#f2d7d5",
        "ring_edge": "#e6b0aa",
        "grid_color": "#c0392b",
        "net_pos": "#d62728",
        "net_neg": "#1f77b4",
    },
}
SCHEME_INDEX = 20
# 获取配色方案
scheme = COLOR_LIBRARY[SCHEME_INDEX]


# =========================================================================================
# ====================================== 3. 绘图函数==========================
# =========================================================================================
def draw_network_heatmap_combo(
    features,
    feature_corr,
    target_name,
    target_corr,
    target_p,
    vif_vals,
    width_ratios=[2, 1],
    w_space=0.1,
):
    # 创建画布
    fig = plt.figure(figsize=(22, 9), facecolor="white")
    # 创建网格布局，1行2列
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=w_space)
    # 网络热图，左侧
    # 在网格的第1个位置创建子图 ax1
    ax1 = fig.add_subplot(gs[0])
    # 设置纵横比相等，保证热图格子是正方形
    # ax1.set_aspect('equal')
    # 特征数量
    n = len(features)

    # 从配色方案中获取热图的colormap
    cmap_heat = plt.get_cmap(scheme["heatmap_cmap"])
    # 设置颜色映射的归一化范围，从-1到1
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    # 定义网络连线颜色
    color_pos = scheme["net_pos"]  # 正相关
    color_neg = scheme["net_neg"]  # 负相关
    # 绘制热图
    for i in range(n):
        for j in range(n):
            # 仅绘制对角线及下三角区域
            if i >= j:
                # 获取特征 i 和特征 j 的相关系数
                val = feature_corr[i, j]
                # 根据相关系数获取对应颜色
                color = cmap_heat(norm(val))
                # 计算矩形格子的 Y 坐标
                rect_y = n - 1 - i
                # 计算矩形格子的 X 坐标
                rect_x = j
                # 创建矩形对象位置, 宽, 高, 填充色, 边框色, 线宽
                rect = patches.Rectangle(
                    (rect_x, rect_y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.8
                )
                # 添加到子图中
                ax1.add_patch(rect)
                # 格式化相关系数值文本
                text_val = f"{val:.2f}"
                # 获取当前颜色的RGB值
                rgb = cmap_heat(norm(val))[:3]
                # 计算亮度，用于决定字体颜色是黑还是白
                brightness = sum(rgb) / 3
                # 如果是对角线
                if i == j:
                    font_weight = "bold"
                    text_color = "white"  # 字体白色
                else:
                    font_weight = "bold"
                    text_color = "black" if 0.4 < brightness else "white"
                    if abs(val) < 0.4:
                        text_color = "black"
                # 添加文本
                ax1.text(
                    rect_x + 0.5,
                    rect_y + 0.5,
                    text_val,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight=font_weight,
                )
    # 绘制网络图部分
    # 目标变量节点的X坐标
    target_x = n * 0.85
    # Y坐标
    target_y = n - 2.0
    # 绘制目标变量的节点
    ax1.scatter(target_x, target_y, s=250, c=scheme["node_color"], zorder=10, edgecolors="white")
    # 添加目标变量的名称文本
    ax1.text(
        target_x + 0.6,
        target_y,
        target_name,
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="black",
    )
    # 遍历所有特征，绘制连接线
    for i in range(n):
        # 获取特征与目标变量的相关系数
        corr_val = target_corr[i]
        # 获取特征与目标变量的P值
        p_val = target_p[i]
        # 根据相关系数设置颜色
        if corr_val > 0:
            line_color = color_pos
        else:
            line_color = color_neg
        # 根据相关系数绝对值设置线宽
        line_width = 0.5 + abs(corr_val) ** 1.5 * 5.0
        # 根据P值决定线型
        line_style = "-" if p_val < 0.05 else "--"
        # 网络线起点的X坐标
        start_x = i + 0.5
        # 网络线起点的Y坐标
        start_y = n - 1 - i + 1.0
        # 网络线终点的X坐标
        end_x = target_x
        # 网络线终点的Y坐标
        end_y = target_y
        # 计算弧度
        if i < n / 2:
            rad = -0.15 - (n / 2 - i) * 0.02
        else:
            rad = 0.1 + (i - n / 2) * 0.02
        # 绘制网络线
        con = patches.ConnectionPatch(
            xyA=(start_x, start_y),
            xyB=(end_x, end_y),  # 坐标
            coordsA="data",
            coordsB="data",  # 坐标系类型
            axesA=ax1,
            axesB=ax1,  # 关联的坐标轴
            arrowstyle="-",  # 箭头样式
            connectionstyle=f"arc3,rad={rad}",  # 连接样式
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
            alpha=1,
            zorder=5,
        )
        # 添加到 ax1
        ax1.add_patch(con)
    # X轴范围
    ax1.set_xlim(0, n)
    # Y轴范围
    ax1.set_ylim(0, n + 3)
    # X轴刻度
    ax1.set_xticks(np.arange(n) + 0.5)
    # X轴刻度标签
    ax1.set_xticklabels(features, rotation=45, ha="right", fontsize=12)
    # Y轴刻度
    ax1.set_yticks(np.arange(n) + 0.5)
    # Y轴刻度标签
    ax1.set_yticklabels(features[::-1], fontsize=12)
    # 开启坐标轴
    ax1.axis("on")
    # 去掉图框
    for spine in ax1.spines.values():
        spine.set_visible(False)
    # 设置刻度线长度
    ax1.tick_params(length=0)
    # 添加子图的标题
    ax1.text(0, n + 2.5, "(a) Correlation Network", fontsize=16, fontweight="bold")
    # 创建一个内嵌坐标轴用于放置颜色条 [x, y, 宽, 高]
    cbar_ax1 = ax1.inset_axes([0.0, -0.12, 0.35, 0.03])
    # 创建颜色条
    cb1 = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap_heat), cax=cbar_ax1, orientation="horizontal"
    )
    # 设置颜色条标题
    cb1.set_label("Correlation", fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    # 网络线颜色图例
    legend_color_elements = [
        matplotlib.lines.Line2D([0], [0], color=color_pos, lw=2, label="Positive"),
        matplotlib.lines.Line2D([0], [0], color=color_neg, lw=2, label="Negative"),
    ]
    # 创建图例对象，放置在底部中央位置
    legend_color = ax1.legend(
        handles=legend_color_elements,
        title="Correlation Type",
        loc="lower center",
        bbox_to_anchor=(0.55, -0.18),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
        ncol=2,
    )
    # 添加
    ax1.add_artist(legend_color)

    # 显著性图例
    sig_handles = [
        matplotlib.lines.Line2D([0], [0], color="gray", lw=2, label="Sig ($p<0.05$)"),
        matplotlib.lines.Line2D(
            [0], [0], color="gray", lw=2, linestyle="--", label=r"Non-sig ($p \geq 0.05$)"
        ),
    ]
    # 创建图例对象
    legend_sig = ax1.legend(
        handles=sig_handles,
        title="Significance",
        loc="lower right",
        bbox_to_anchor=(1.05, -0.18),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    # 添加该图例
    ax1.add_artist(legend_sig)
    # 相关性强度图例
    width_handles = [
        matplotlib.lines.Line2D([0], [0], color="gray", lw=0.5 + 0.5**1.5 * 5, label="|r|=0.5"),
        matplotlib.lines.Line2D([0], [0], color="gray", lw=0.5 + 0.8**1.5 * 5, label="|r|=0.8"),
    ]
    ax1.legend(
        handles=width_handles,
        title="Strength",
        loc="lower right",
        bbox_to_anchor=(0.82, -0.18),
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )
    # -------------------------------------------------------------------------
    # VIF图右侧
    # -------------------------------------------------------------------------
    # 在网格的第2个位置创建子图
    ax2 = fig.add_subplot(gs[1], projection="polar")
    # 获取当前ax2的原始位置 [left, bottom, width, height]
    pos = ax2.get_position()
    # 设置缩放比例
    scale_factor = 0.8
    # 新的宽度和高度
    new_width = pos.width * scale_factor
    new_height = pos.height * scale_factor
    # 新的中心位置，保持居中
    new_x = pos.x0 + (pos.width - new_width) / 2
    new_y = pos.y0 + (pos.height - new_height) / 2
    # 应用新的位置
    ax2.set_position([new_x - 0.07, new_y - 0.07, new_width, new_height])
    # 获取变量数量
    num_vars = len(features)
    # 计算每个变量的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False) + np.pi / 2
    # 设置柱子的宽度
    width = 2 * np.pi / num_vars * 0.6
    # 获取 VIF 的最大值
    vif_max_val = max(vif_vals)
    # 设置绘图的径向限制 (比最大值稍大)
    plot_limit = vif_max_val * 1.15
    # 获取颜色映射
    cmap_vif = plt.get_cmap(scheme["radar_cmap"])
    # 归一化VIF值以映射颜色
    norm_vif = mcolors.Normalize(vmin=min(vif_vals), vmax=vif_max_val)
    # 生成每个柱子的颜色
    colors_vif = [cmap_vif(norm_vif(v)) for v in vif_vals]

    # 绘制极坐标柱状图
    ax2.bar(
        angles,
        vif_vals,
        width=width,
        bottom=0.0,
        color=colors_vif,
        alpha=0.9,
        edgecolor="white",
        zorder=10,
    )

    # 移除 Y 轴标签
    ax2.set_yticklabels([])
    # 移除 X 轴标签
    ax2.set_xticklabels([])
    # 关闭默认网格
    ax2.grid(False)
    # 隐藏极坐标的脊柱
    ax2.spines["polar"].set_visible(False)
    # 设置径向显示范围
    ax2.set_ylim(0, plot_limit)

    # 定义背景网格圈的比例
    percentages = [0.25, 0.50, 0.75, 1.00]
    # 计算网格圈的半径
    grid_radii = [p * vif_max_val for p in percentages]

    # 绘制虚线同心圆网格
    for r in grid_radii:
        circle = plt.Circle(
            (0, 0),
            r,
            transform=ax2.transData._b,
            fill=False,
            edgecolor=scheme["grid_color"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
            zorder=5,
        )
        ax2.add_artist(circle)
    # 绘制及径向线
    for ang in angles:
        ax2.plot(
            [ang, ang],
            [0, plot_limit * 0.95],
            color=scheme["grid_color"],
            linestyle="--",
            linewidth=0.8,
            alpha=0.3,
            zorder=0,
        )

    # 定义外环的内半径
    ring_inner = plot_limit * 0.98
    # 定义外环的外半径
    ring_outer = plot_limit * 1.08
    # 生成外环的角度序列
    theta_ring = np.linspace(0, 2 * np.pi, 360)
    # 填充外环颜色
    ax2.fill_between(theta_ring, ring_inner, ring_outer, color=scheme["ring_fill"], zorder=0)

    # 绘制外环的内边框线
    circle_inner = plt.Circle(
        (0, 0),
        ring_inner,
        transform=ax2.transData._b,
        fill=False,
        edgecolor=scheme["ring_edge"],
        linewidth=2,
        zorder=1,
    )
    # 绘制外环的外边框线
    circle_outer = plt.Circle(
        (0, 0),
        ring_outer,
        transform=ax2.transData._b,
        fill=False,
        edgecolor=scheme["ring_edge"],
        linewidth=2,
        zorder=1,
    )
    # 添加外圈图框
    ax2.add_artist(circle_inner)
    ax2.add_artist(circle_outer)
    # 遍历角度、VIF值和标签，添加文本
    for angle, val, label in zip(angles, vif_vals, features):
        # 在柱子上方添加具体的VI 数值
        ax2.text(
            angle,
            val + plot_limit * 0.05,
            f"{val:.1f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=15,
        )
        # 计算标签位置 (在外环中间)
        label_pos = (ring_inner + ring_outer) / 1.95
        # 计算标签旋转角度 (转换为度)
        rot = np.degrees(angle)
        # 如果在左半圆，文字翻转180度以方便阅读
        if angle > np.pi:
            rot -= 360
        # 在外环添加特征名称标签
        ax2.text(
            angle,
            label_pos,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            rotation=rot,
            rotation_mode="anchor",
            zorder=15,
        )

    # 添加标题
    plt.figtext(0.45, 0.85, "(b) VIF of Features", fontsize=16, fontweight="bold")
    # 颜色条位置
    cbar_ax2 = ax2.inset_axes([0.1, -0.1, 0.8, 0.03])
    # 创建 VIF 颜色条
    cb2 = plt.colorbar(
        cm.ScalarMappable(norm=norm_vif, cmap=cmap_vif), cax=cbar_ax2, orientation="horizontal"
    )
    # 设置颜色条标题
    cb2.set_label("VIF Value", fontsize=10)
    # 保存
    save_path = os.path.join(OUTPUT_DIR, f"{SCHEME_INDEX}.png")
    save_path_pdf = os.path.join(OUTPUT_DIR, f"{SCHEME_INDEX}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")


# =========================================================================================
# ====================================== 4. 数据分析与执行 =================================
# =========================================================================================
if __name__ == "__main__":
    # 定义输出路劲
    OUTPUT_DIR = str(OUTPUT_DIR)
    # 定义需要分析的特征列表
    features_list = [
        "NDVI",
        "NDSI",
        "Aspect",
        "SWE",
        "NDWI",
        "NIR",
        "SWC",
        "PRE",
        "LULC",
        "TEM",
        "PH",
    ]
    # 定义目标变量名称
    target_var = "LST"
    MY_WIDTH_RATIOS = [0.8, 1]  # 左图宽度，右图宽度
    MY_W_SPACE = 0.12  # 左右图间距
    # 读取文件
    df = pd.read_excel(str(DATA_DIR / "data.xlsx"))
    # 计算特征之间的相关性矩阵
    f_corr = df[features_list].corr().values
    # 计算特征与目标变量的相关性及P值
    # 存储相关系数
    t_corr = []
    # 存储P值
    t_p = []
    # 遍历每个特征
    for feat in features_list:
        # 计算当前特征与目标变量的皮尔逊相关系数和P值
        r, p = pearsonr(df[feat], df[target_var])
        # 将r值添加到列表
        t_corr.append(r)
        # 将p值添加到列表
        t_p.append(p)
    # 将列表转换为 numpy 数组
    t_corr = np.array(t_corr)
    t_p = np.array(t_p)
    # 计算 VIF
    # 为特征数据添加常数项
    X = add_constant(df[features_list])
    # 遍历每一列计算VIF
    vif_series = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns
    )
    # 剔除常数项的VIF，只保留特征的 VIF 值
    vifs = [vif_series[f] for f in features_list]
    # 调用绘图函数
    draw_network_heatmap_combo(
        features_list,
        f_corr,
        target_var,
        t_corr,
        t_p,
        vifs,
        width_ratios=MY_WIDTH_RATIOS,
        w_space=MY_W_SPACE,
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

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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, PathPatch
from matplotlib.path import Path
from scipy.stats import pearsonr

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
# =========================================================================================
# ====================================== 2. 颜色库设置===============================
# =========================================================================================
# -------------------------------------------------------------------------------------------------------------------------------------------------
# 一定要记得修改变量名啊！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# -------------------------------------------------------------------------------------------------------------------------------------------------
COLOR_SCHEMES = {
    1: {
        "sites": ["#008B8B", "#87CEFA", "#FFA07A", "#FF0000"],
        "corr_positive": "blue",  # 定义站点颜色和正相关颜色
        "corr_negative": "red",
        "node": "darkgray",
        "regression_line": "black",  # 定义负相关、节点、回归线颜色
        "legend_text": "black",  # 定义图例文本颜色
    },
    2: {
        "sites": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
        "corr_positive": "#377eb8",
        "corr_negative": "#e41a1c",
        "node": "#595959",
        "regression_line": "black",
        "legend_text": "black",
    },
    3: {
        "sites": ["#d87c7c", "#919e8b", "#d7ab82", "#6e7074"],
        "corr_positive": "#546570",
        "corr_negative": "#c4ccd3",
        "node": "gray",
        "regression_line": "black",
        "legend_text": "black",
    },
    4: {
        "sites": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"],
        "corr_positive": "#80b1d3",
        "corr_negative": "#fdb462",
        "node": "#737373",
        "regression_line": "black",
        "legend_text": "black",
    },
    5: {
        "sites": ["#5470c6", "#91cc75", "#fac858", "#ee6666"],
        "corr_positive": "#5470c6",
        "corr_negative": "#ee6666",
        "node": "#696969",
        "regression_line": "black",
        "legend_text": "black",
    },
    6: {
        "sites": ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"],
        "corr_positive": "#2ca02c",
        "corr_negative": "#d62728",
        "node": "dimgray",
        "regression_line": "black",
        "legend_text": "black",
    },
    7: {
        "sites": ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"],
        "corr_positive": "#1f77b4",
        "corr_negative": "#ff7f0e",
        "node": "#606060",
        "regression_line": "black",
        "legend_text": "black",
    },
    8: {
        "sites": ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"],
        "corr_positive": "#1f78b4",
        "corr_negative": "#fb9a99",
        "node": "#828282",
        "regression_line": "black",
        "legend_text": "black",
    },
    9: {
        "sites": ["#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"],
        "corr_positive": "#33a02c",
        "corr_negative": "#e31a1c",
        "node": "darkslategray",
        "regression_line": "black",
        "legend_text": "black",
    },
    10: {
        "sites": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"],
        "corr_positive": "#66c2a5",
        "corr_negative": "#fc8d62",
        "node": "#6b6b6b",
        "regression_line": "black",
        "legend_text": "black",
    },
    11: {
        "sites": ["#5f9e70", "#b3d4a0", "#f7e39c", "#f29c9a"],
        "corr_positive": "#5f9e70",
        "corr_negative": "#f29c9a",
        "node": "darkolivegreen",
        "regression_line": "black",
        "legend_text": "black",
    },
    12: {
        "sites": ["#1e88e5", "#004d40", "#80deea", "#00bcd4"],
        "corr_positive": "#1e88e5",
        "corr_negative": "#ffc107",
        "node": "#424242",
        "regression_line": "black",
        "legend_text": "black",
    },
    13: {
        "sites": ["#ffca28", "#f57c00", "#d84315", "#bf360c"],
        "corr_positive": "#4caf50",
        "corr_negative": "#d84315",
        "node": "saddlebrown",
        "regression_line": "black",
        "legend_text": "black",
    },
    14: {
        "sites": ["#4dd0e1", "#4db6ac", "#81c784", "#aed581"],
        "corr_positive": "#4dd0e1",
        "corr_negative": "#f44336",
        "node": "#546e7a",
        "regression_line": "black",
        "legend_text": "black",
    },
    15: {
        "sites": ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4"],
        "corr_positive": "#b3cde3",
        "corr_negative": "#fbb4ae",
        "node": "#7f7f7f",
        "regression_line": "black",
        "legend_text": "black",
    },
    16: {
        "sites": ["#cccccc", "#969696", "#636363", "#252525"],
        "corr_positive": "steelblue",
        "corr_negative": "firebrick",
        "node": "black",
        "regression_line": "#e41a1c",
        "legend_text": "black",
    },
    17: {
        "sites": ["#9e9ac8", "#807dba", "#6a51a3", "#54278f"],
        "corr_positive": "#807dba",
        "corr_negative": "#ef6548",
        "node": "#404040",
        "regression_line": "black",
        "legend_text": "black",
    },
    18: {
        "sites": ["#e5c494", "#fde2e2", "#b2e2e2", "#ccebc5"],
        "corr_positive": "#8da0cb",
        "corr_negative": "#fc8d62",
        "node": "gray",
        "regression_line": "black",
        "legend_text": "black",
    },
    19: {
        "sites": ["#b15928", "#ffff99", "#6a3d9a", "#cab2d6"],
        "corr_positive": "#33a02c",
        "corr_negative": "#b15928",
        "node": "#4a4a4a",
        "regression_line": "black",
        "legend_text": "black",
    },
    20: {
        "sites": ["#2c7fb8", "#7fcdbb", "#edf8b1", "#fc9272"],
        "corr_positive": "#2c7fb8",
        "corr_negative": "#fc9272",
        "node": "darkgray",
        "regression_line": "black",
        "legend_text": "black",
    },
}
SELECTED_COLOR_SCHEME = 15  # 选择绘图颜色
# =========================================================================================
# ====================================== 3.定义绘图使用的变量===============================
# =========================================================================================
# -------------------------------------------------------------------------------------------------------------------------------------------------
# 一定要记得修改变量名啊！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# -------------------------------------------------------------------------------------------------------------------------------------------------
scatter_vars = [
    "Leaf N concentration",
    "Chl concentration",
    "SLA",
    "Stomatal density",
    "Pore length",
]  # 散点图矩阵中要使用的变量
network_vars = ["G_wmax", "V_cmax", "WUE_i"]  # 网络图部分要使用的变量
all_vars = network_vars + scatter_vars  # 将网络图变量和散点图变量合并成一个总的变量列表
sites = [
    "CB: Cold-temperate forest",
    "DL: Warm-temperate forest",
    "DH: Subtropical forest",
    "XSBN: Tropical forest",
]  # 站点的名称
leaf_types = ["Deciduous", "Evergreen"]  # 叶片的类型
leaf_markers = {"Deciduous": "o", "Evergreen": "^"}  # 为不同叶片类型指定不同的散点标记样式


# =========================================================================================
# ====================================== 4. 绘图函数====================================
# =========================================================================================
def create_network_scatterplot_matrix(df, color_scheme, save_path="."):
    # 从选择的配色方案中提取颜色，为每个站点的数据分别分配一种颜色
    site_colors = {
        site: color_scheme["sites"][i % len(color_scheme["sites"])] for i, site in enumerate(sites)
    }

    # 创建画布和网格布局
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        len(scatter_vars),  # 在画布上创建一个网格布局，用于精细控制子图的位置。
        9,  # 网格的列数
        wspace=0.1,  # 设置子图之间的水平间距
        hspace=0.1,  # 设置子图之间的垂直间距
        left=0.05,  # 设置整个网格左边缘的距离
        bottom=0.1,  # 设置整个网格下边缘的距离
        right=0.95,  # 设置整个网格右边缘的距离
        top=0.9,
    )  # 设置整个网格上边缘的距离
    ax_main = fig.add_subplot(gs[:, 1:], zorder=5)  # 在网格中添加一个主坐标轴，用于绘制网络图
    ax_main.patch.set_visible(False)  # 将主坐标轴的背景补丁设置为透明背景
    ax_main.axis("off")  # 关闭主坐标轴的坐标轴线、刻度和标签
    ax_main.set_xlim(0, 1)  # 设置主坐标轴的X轴范围
    ax_main.set_ylim(0, 1)  # 设置主坐标轴的Y轴范围

    # 绘制右侧的散点图矩阵
    scatter_axes = [
        [None for _ in range(len(scatter_vars))] for _ in range(len(scatter_vars))
    ]  # 创建一个矩阵，用于存储散点图矩阵中每个子图的坐标轴对象
    for i in range(len(scatter_vars)):  # 开始外层循环，遍历每个散点图变量，Y轴，对应行
        for j in range(len(scatter_vars)):  # 开始内层循环，遍历每个散点图变量，X轴，对应列
            ax = fig.add_subplot(gs[i, j + 4], zorder=1)  # 在网格布局的指定位置创建一个子图坐标轴
            scatter_axes[i][j] = ax  # 将创建的坐标轴对象存入之前定义的矩阵中
            if i < j:  # 判断当前子图是否位于散点图矩阵的右上三角部分
                # 使用seaborn的regplot函数绘制回归拟合线
                sns.regplot(
                    data=df,  # 绘图所使用的数据集
                    x=scatter_vars[j],  # x轴对应的数据列
                    y=scatter_vars[i],  # y轴对应的数据列
                    ax=ax,  # 指定将图形绘制在哪个子图坐标轴上
                    scatter=False,  # 设置为False，表示只绘制回归线，不绘制散点
                    ci=None,  # 不显示置信区间
                    line_kws={"color": color_scheme["regression_line"], "lw": 2.5},
                )  # 设置回归线的颜色和线宽
                # 使用seaborn的scatterplot函数在同一子图上绘制散点图
                sns.scatterplot(
                    data=df,  # 指定绘图所使用的数据集
                    x=scatter_vars[j],  # 指定x轴对应的数据列
                    y=scatter_vars[i],  # 指定y轴对应的数据列
                    ax=ax,  # 指定将图形绘制在同一个子图坐标轴(ax)上
                    hue="Site",  # 指定使用'Site'列的数据来为散点分配颜色
                    style="Leaf_type",  # 指定使用'Leaf_type'列的数据来为散点分配形状
                    palette=site_colors,  # 为颜色分配具体的调色板
                    markers=leaf_markers,  # 为形状分配具体的标记样式
                    hue_order=sites,  # 指定颜色类的顺序
                    style_order=leaf_types,  # 指定形状分类的顺序
                    s=60,  # 设置散点的大小
                    alpha=0.9,  # 设置散点的透明度
                    edgecolor="white",  # 设置散点的边缘颜色为白色
                    legend=False,
                )  # 在这个子图上不显示图例
            ax.set_xlabel("")  # 清空子图的X轴标签
            ax.set_ylabel("")  # 清空子图的Y轴标签
            if i == 0 and j > 0:  # 如果是矩阵的第一行（非对角线）
                ax.xaxis.set_ticks_position("top")  # 将X轴的刻度线放在顶部
                ax.xaxis.set_label_position("top")  # 将X轴的标签放在顶部
                ax.tick_params(
                    axis="x",  # 指定要修改的坐标轴
                    which="major",  # 指定修改的是主刻度线
                    labelsize=14,  # 设置刻度标签的字体大小
                    length=8,  # 设置主刻度线长度
                    width=1.5,
                )  # 设置主刻度线粗细
            else:  # 对于其他行
                ax.set_xticks([])  # 隐藏X轴的刻度
            if (
                j == len(scatter_vars) - 1 and i < len(scatter_vars) - 1
            ):  # 如果是矩阵的最后一列（非对角线）
                ax.yaxis.set_ticks_position("right")  # 将Y轴的刻度线放在右侧
                ax.yaxis.set_label_position("right")  # 将Y轴的标签放在右侧
                ax.tick_params(
                    axis="y",
                    labelsize=14,  # 设置刻度标签的字体大小
                    length=8,  # 设置主刻度线长度
                    width=1.5,
                )  # 设置主刻度线粗细
            else:  # 对于其他列
                ax.set_yticks([])  # 隐藏Y轴的刻度
            if i > j:  # 如果当前子图位于散点图矩阵的左下三角部分
                ax.set_visible(False)  # 将该子图设置为不可见

    # 在对角线上添加变量标签
    for i in range(len(scatter_vars)):  # 循环遍历散点图矩阵的对角线位置
        ax = scatter_axes[i][i]  # 获取对角线上的子图坐标轴对象
        ax.text(
            0.5,  # 指定文本位置的x坐标
            0.5,  # 指定文本位置的y坐标
            scatter_vars[i].replace(" ", "\n"),  # 指定要显示的文本内容
            ha="center",  # 水平对齐方式
            va="center",  # 垂直对齐方式
            fontsize=14,  # 指定字体大小
            fontweight="bold",  # 字体粗细
            transform=ax.transAxes,
        )  # 指定前面坐标的变换方式为相对坐标系

    # 计算网络图节点的位置
    # fig.canvas.draw()  # 强制重绘画布，以确保所有坐标变换都已更新
    node_positions = {}  # 创建一个空字典，用于存储每个变量节点的位置坐标
    for i, var_name in enumerate(scatter_vars):  # 循环遍历散点图变量
        ax_scatter_diag = scatter_axes[i][i]  # 获取对角线上的子图坐标轴
        center_pixel = ax_scatter_diag.transAxes.transform(
            (-0.1, -0.1)
        )  # 获取对角线子图左侧某个点的像素坐标
        pos_in_main_ax = ax_main.transData.inverted().transform(
            center_pixel
        )  # 将该像素坐标转换回主坐标轴的数据坐标
        node_positions[var_name] = pos_in_main_ax  # 将计算出的位置存入字典

    # 计算左侧节点的位置
    pos_diag_start = node_positions[scatter_vars[0]]  # 获取散点图变量中第一个节点的坐标
    pos_diag_end = node_positions[scatter_vars[-1]]  # 获取散点图变量中最后一个节点的坐标
    horizontal_offset = -0.5  # 定义一个水平偏移量，用于放置左侧的网络图节点
    pos_left_start = np.array(
        [pos_diag_start[0] + horizontal_offset, pos_diag_start[1]]
    )  # 计算左侧节点区域的起始点坐标
    pos_left_end = np.array(
        [pos_diag_end[0] + horizontal_offset, pos_diag_end[1]]
    )  # 计算左侧节点区域的结束点坐标
    left_node_ratios = [0.4, 0.7, 1]  # 定义左侧节点在线段上的分布比例
    assert len(network_vars) == len(left_node_ratios)  # 检查网络变量数量和比例数量是否一致
    for i, var_name in enumerate(network_vars):  # 循环遍历网络图变量
        ratio = left_node_ratios[i]  # 获取当前变量的分布比例
        node_positions[var_name] = pos_left_start + ratio * (
            pos_left_end - pos_left_start
        )  # 通过线性插值计算当前网络图节点的坐标

    # 绘制所有节点和标签
    ellipse_height_data = 0.024  # 定义节点的高度
    ellipse_width_data = ellipse_height_data * (10 / 16)  # 节点的宽度
    for var, pos in node_positions.items():  # 循环遍历所有变量及其计算出的位置
        node = Ellipse(
            xy=pos,  # 创建一个节点，xy参数指定椭圆的中心坐标
            width=ellipse_width_data,  # 宽度
            height=ellipse_height_data,  # 高度
            color=color_scheme["node"],  # 颜色
            zorder=10,
            ec="black",  # 边框颜色
            clip_on=False,
        )  # 设置为False，允许图形绘制到坐标轴区域之外而不会被裁剪掉
        ax_main.add_patch(node)  # 将节点添加到主坐标轴上
        if var in network_vars:  # 如果当前变量是网络图变量
            label = (
                var.replace("_", "_{") + "}" if "_" in var else var
            )  # 格式化标签文本，用于显示LaTeX下标
            line_vec = pos_left_end - pos_left_start  # 计算左侧节点连线的方向向量
            if np.linalg.norm(line_vec) > 0:  # 如果向量长度不为零
                perp_vec = np.array([-line_vec[1], line_vec[0]])  # 计算该向量的垂直向量
                perp_vec = perp_vec / np.linalg.norm(perp_vec)  # 将垂直向量单位化
                offset_amount = 0.02  # 定义标签的偏移量
                label_pos = pos - perp_vec * offset_amount  # 计算标签的位置
            else:  # 如果向量长度为零
                label_pos = pos  # 标签位置就是节点位置
            ax_main.text(
                label_pos[0],
                label_pos[1],
                f"${label}$",
                ha="right",
                va="center",  # 在计算出的位置添加文本标签，使用LaTeX格式
                fontsize=18,
                fontweight="bold",
                zorder=11,
                color=color_scheme["legend_text"],
            )  # 设置标签的字体、粗细、层级和颜色

    # 绘制节点间的连接线
    connections_to_draw = []  # 创建一个空列表，用于存储需要绘制连接线的变量对
    for left_var in network_vars:  # 遍历所有网络图变量
        for right_var in scatter_vars:  # 遍历所有散点图变量
            connections_to_draw.append((left_var, right_var))  # 将这对变量作为一个连接添加到列表中
    arc_radius = -0.1  # 定义连接线弯曲的半径
    for var1, var2 in connections_to_draw:  # 遍历所有需要连接的变量对
        r_val, p_val = pearsonr(df[var1], df[var2])  # 计算这两个变量之间的皮尔逊相关系数和p值
        color = (
            color_scheme["corr_positive"] if r_val >= 0 else color_scheme["corr_negative"]
        )  # 根据r值的正负选择连接线的颜色
        linestyle = "solid" if p_val < 0.05 else "dashed"  # 根据p值是否显著来选择线的样式
        linewidth = 1.5 + 8 * abs(r_val)  # 根据r值绝对值的大小来设置线的宽度，相关性越强线越粗
        pos1 = np.array(node_positions[var1])  # 获取第一个节点的坐标
        pos2 = np.array(node_positions[var2])  # 获取第二个节点的坐标
        midpoint = (pos1 + pos2) / 2  # 计算两个节点的中点坐标

        dist_vec = pos2 - pos1  # 计算两个节点之间的方向向量
        perp_vec = np.array([-dist_vec[1], dist_vec[0]])  # 计算方向向量的垂直向量

        perp_vec_norm = np.linalg.norm(perp_vec)  # 计算垂直向量的长度
        if perp_vec_norm > 0:
            perp_vec /= perp_vec_norm  # 如果长度不为零，则单位化垂直向量

        control_offset = np.linalg.norm(dist_vec) * arc_radius  # 计算贝塞尔曲线控制点的偏移量
        control_point = (
            midpoint + perp_vec * control_offset
        )  # 计算贝塞尔曲线的控制点坐标，以产生弧线效果
        verts = [pos1, control_point, pos2]  # 定义路径的顶点：起点、控制点、终点
        codes = [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
        ]  # 定义顶点的类型：移动到、二次贝塞尔曲线、二次贝塞尔曲线
        path = Path(verts, codes)  # 创建一个路径对象
        patch = PathPatch(
            path,  # 基于之前创建的路径对象，创建一个可以添加到图表中的对象
            facecolor="none",  # 填充颜色
            edgecolor=color,  # 边框颜色，也就是曲线本身的颜色
            lw=linewidth,  # 曲线的宽度
            linestyle=linestyle,  # 曲线的线条样式
            zorder=8,
            alpha=0.9,  # 透明度
            clip_on=False,
        )  # 设置为False，允许图形在坐标轴区域之外完整绘制，而不会被裁剪
        ax_main.add_patch(patch)  # 将创建的曲线补丁添加到主坐标轴上
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # 一定要记得修改变量名啊！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # -------------------------------------------------------------------------------------------------------------------------------------------------
    # 图例绘制
    ax_legend = fig.add_subplot(gs[:, 0])  # 在网格布局的最左侧添加一个专门用于图例的子图坐标轴
    ax_legend.axis("off")  # 关闭该坐标轴的轴线、刻度和标签
    ax_legend.set_xlim(0, 1)  # 设置图例坐标轴的X轴范围
    ax_legend.set_ylim(0, 1)  # 设置图例坐标轴的Y轴范围

    # 重新调整参数，确保所有图例都在[0,1]的Y轴范围内
    y_start = 0.98  # 设置图例开始绘制的Y轴位置
    x_icon = 0.1  # 设置图例图标的X轴位置
    x_text = 0.35  # 设置图例文本的X轴位置
    y_step = 0.055  # 减小步长，即每个图例项之间的垂直距离
    group_spacing = 0.07  # 不同图例组之间的垂直距离
    font_props = {
        "family": "serif",
        "size": 14,
        "color": color_scheme["legend_text"],
    }  # 定义普通图例文本的字体属性
    title_font_props = {
        "family": "serif",
        "size": 16,
        "weight": "bold",
        "color": color_scheme["legend_text"],
    }  # 定义图例标题的字体属性
    current_y = y_start  # 初始化当前绘制的Y轴位置

    # Site图例
    ax_legend.text(0.05, current_y, "Site", fontdict=title_font_props, va="top")  # 图例的标题
    current_y -= y_step * 1.2  # 更新Y轴位置，为图例项留出空间
    for site_name, color in site_colors.items():  # 遍历每个站点及其对应的颜色
        # 在图例坐标轴上绘制一个点，作为图例的图标
        ax_legend.plot(
            x_icon,  # x坐标
            current_y,  # y坐标
            marker="o",  # 形状
            color=color,  # 颜色
            markersize=12,  # 大小
            linestyle="None",  # 设置线条样式
            clip_on=False,
        )  # 允许图标绘制到坐标轴区域之外而不会被裁剪
        # 在图例坐标轴上添加与图标对应的文本标签
        ax_legend.text(
            x_text,  # x坐标
            current_y,  # y坐标
            site_name,  # 指定要显示的文本内容
            fontdict=font_props,  # 通过一个字典来设置文本的字体属性（如字体、大小、颜色等）
            va="center",
        )  # 设置垂直对齐方
        current_y -= y_step  # 更新Y轴位置，准备绘制下一个图例项

    # Leaf_type图例
    current_y -= group_spacing  # 在两组图例之间增加一些额外的垂直间距
    ax_legend.text(
        0.05, current_y, "Leaf_type", fontdict=title_font_props, va="top"
    )  # 绘制'Leaf_type'图例的标题
    current_y -= y_step * 1.2  # 更新Y轴位置
    for type_name, marker in leaf_markers.items():  # 遍历每个叶片类型及其对应的标记样式
        ax_legend.plot(
            x_icon,
            current_y,
            marker=marker,
            color=color_scheme["legend_text"],
            markersize=10,  # 绘制相应形状的图标
            linestyle="None",
            clip_on=False,
        )  # 设置图标样式
        ax_legend.text(
            x_text, current_y, type_name, fontdict=font_props, va="center"
        )  # 绘制叶片类型名称文本
        current_y -= y_step  # 更新Y轴位置

    # Pearson r图例
    current_y -= group_spacing  # 增加组间距
    ax_legend.text(
        0.05, current_y, "Pearson's " + r"$r$", fontdict=title_font_props, va="top"
    )  # 绘制皮尔逊r值的图例标题，使用LaTeX格式显示'r'
    current_y -= y_step * 1.2  # 更新Y轴位置
    ax_legend.plot(
        [x_icon - 0.05, x_icon + 0.05],
        [current_y, current_y],
        color=color_scheme["corr_negative"],
        lw=3,  # 绘制一条代表负相关的短线
        clip_on=False,
    )  # 允许绘制到坐标轴外
    ax_legend.text(x_text, current_y, "< 0", fontdict=font_props, va="center")  # 绘制文本'< 0'
    current_y -= y_step  # 更新Y轴位置
    ax_legend.plot(
        [x_icon - 0.05, x_icon + 0.05],
        [current_y, current_y],
        color=color_scheme["corr_positive"],
        lw=3,  # 绘制一条代表正相关的短线
        clip_on=False,
    )  # 允许绘制到坐标轴外
    ax_legend.text(
        x_text, current_y, r"$\geq 0$", fontdict=font_props, va="center"
    )  # 绘制文本'>= 0'，使用LaTeX格式
    current_y -= y_step  # 更新Y轴位置

    # P value图例
    current_y -= group_spacing  # 增加组间距
    ax_legend.text(
        0.05, current_y, r"$P$ value", fontdict=title_font_props, va="top"
    )  # 绘制P值的图例标题，使用LaTeX格式显示'P'
    current_y -= y_step * 1.2  # 更新Y轴位置

    ax_legend.plot(
        [x_icon - 0.05, x_icon + 0.15],
        [current_y, current_y],
        color=color_scheme["legend_text"],  # 绘制一条代表显著的实线
        linestyle="solid",
        lw=4,
        clip_on=False,
    )  # 设置线条样式、宽度，并允许绘制到坐标轴外
    ax_legend.text(
        x_text, current_y, "< 0.05", fontdict=font_props, va="center"
    )  # 绘制文本'< 0.05'

    current_y -= y_step  # 更新Y轴位置

    # 使用 plot 函数绘制一条自定义样式的线段，作为图例中的一个条目
    ax_legend.plot(
        [x_icon - 0.05, x_icon + 0.15],  # 指定线段的起始点和结束点的x坐标，从而定义其水平位置和长度
        [
            current_y,
            current_y,
        ],  # 指定线段的起始点和结束点的y坐标，由于两个值相同，所以绘制的是一条水平线
        color=color_scheme["legend_text"],  # 设置线段的颜色
        dashes=(2, 2),  # 自定义虚线样式，表示绘制2个点的实线，然后跳过2个点的空白，如此循环
        lw=4,  # 线段的粗细
        clip_on=False,
    )

    ax_legend.text(
        x_text, current_y, r"$\geq 0.05$", fontdict=font_props, va="center"
    )  # 绘制文本'>= 0.05'，使用LaTeX格式

    # 保存图片文件
    png_path = os.path.join(save_path, f"result_plot_{SELECTED_COLOR_SCHEME}.png")
    pdf_path = os.path.join(save_path, f"result_plot_{SELECTED_COLOR_SCHEME}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"结果图已保存至: {save_path}")


# =========================================================================================
# ====================================== 4. 主程序入口 ======================================
# =========================================================================================
if __name__ == "__main__":
    data_file_path = str(DATA_DIR / "simulated_data.xlsx")  # 加载绘图数据
    output_directory = str(OUTPUT_DIR)  # 设置结果图的保存路径
    main_df = pd.read_excel(data_file_path)  # 读取数据
    print(f"数据已从 '{data_file_path}' 成功加载。")
    # 获取配色方案
    selected_scheme = COLOR_SCHEMES[SELECTED_COLOR_SCHEME]
    # 调用绘图函数
    create_network_scatterplot_matrix(
        main_df,  # 传入数据
        color_scheme=selected_scheme,  # 配色方案
        save_path=output_directory,
    )  # 保存路径

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

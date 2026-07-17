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
# ======================================1.库的导入和全局字体设置=========================================
# =========================================================================================
import matplotlib
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# =========================================================================================
# =====================================2.颜色库设置=========================
# =========================================================================================
color_library = {
    1: {
        "heatmap_negative": "#4575b4",
        "heatmap_zero": "#ffffbf",
        "heatmap_positive": "#d73027",
        "center_circle_face": "#4d4d4d",
        "center_text": "#ffffff",
    },
    2: {
        "heatmap_negative": "#2166ac",
        "heatmap_zero": "#f7f7f7",
        "heatmap_positive": "#b2182b",
        "center_circle_face": "#333333",
        "center_text": "#ffffff",
    },
    3: {
        "heatmap_negative": "#762a83",
        "heatmap_zero": "#f7f7f7",
        "heatmap_positive": "#1b7837",
        "center_circle_face": "#40004b",
        "center_text": "#ffffff",
    },
    4: {
        "heatmap_negative": "#8c510a",
        "heatmap_zero": "#f5f5f5",
        "heatmap_positive": "#01665e",
        "center_circle_face": "#35978f",
        "center_text": "#ffffff",
    },
    5: {
        "heatmap_negative": "#c51b7d",
        "heatmap_zero": "#fde0ef",
        "heatmap_positive": "#4d9221",
        "center_circle_face": "#276419",
        "center_text": "#ffffff",
    },
    6: {
        "heatmap_negative": "#542788",
        "heatmap_zero": "#f7f7f7",
        "heatmap_positive": "#b35806",
        "center_circle_face": "#e08214",
        "center_text": "#ffffff",
    },
    7: {
        "heatmap_negative": "#4d4d4d",
        "heatmap_zero": "#e0e0e0",
        "heatmap_positive": "#ca0020",
        "center_circle_face": "#000000",
        "center_text": "#ffffff",
    },
    8: {
        "heatmap_negative": "#d73027",
        "heatmap_zero": "#ffffbf",
        "heatmap_positive": "#1a9850",
        "center_circle_face": "#006837",
        "center_text": "#ffffff",
    },
    9: {
        "heatmap_negative": "#6b7b8c",
        "heatmap_zero": "#f2eee5",
        "heatmap_positive": "#b05b53",
        "center_circle_face": "#4a5866",
        "center_text": "#ffffff",
    },
    10: {
        "heatmap_negative": "#08519c",
        "heatmap_zero": "#f7fbff",
        "heatmap_positive": "#006d2c",
        "center_circle_face": "#00441b",
        "center_text": "#ffffff",
    },
    11: {
        "heatmap_negative": "#2d004b",
        "heatmap_zero": "#1a1a1a",
        "heatmap_positive": "#ff00cc",
        "center_circle_face": "#ffffff",
        "center_text": "#000000",
    },
    12: {
        "heatmap_negative": "#7f7f7f",
        "heatmap_zero": "#ffffff",
        "heatmap_positive": "#08519c",
        "center_circle_face": "#252525",
        "center_text": "#ffffff",
    },
    13: {
        "heatmap_negative": "#313695",
        "heatmap_zero": "#e0f3f8",
        "heatmap_positive": "#a50026",
        "center_circle_face": "#74add1",
        "center_text": "#000000",
    },
    14: {
        "heatmap_negative": "#2c7fb8",
        "heatmap_zero": "#ffffd9",
        "heatmap_positive": "#41b6c4",
        "center_circle_face": "#253494",
        "center_text": "#ffffff",
    },
    15: {
        "heatmap_negative": "#762a83",
        "heatmap_zero": "#ffffff",
        "heatmap_positive": "#e08214",
        "center_circle_face": "#542788",
        "center_text": "#ffffff",
    },
    16: {
        "heatmap_negative": "#2b83ba",
        "heatmap_zero": "#ffffbf",
        "heatmap_positive": "#d7191c",
        "center_circle_face": "#2c7bb6",
        "center_text": "#ffffff",
    },
    17: {
        "heatmap_negative": "#cccccc",
        "heatmap_zero": "#ffffff",
        "heatmap_positive": "#000000",
        "center_circle_face": "#666666",
        "center_text": "#ffffff",
    },
    18: {
        "heatmap_negative": "#8c510a",
        "heatmap_zero": "#f6e8c3",
        "heatmap_positive": "#01665e",
        "center_circle_face": "#35978f",
        "center_text": "#ffffff",
    },
    19: {
        "heatmap_negative": "#018571",
        "heatmap_zero": "#f5f5f5",
        "heatmap_positive": "#a6611a",
        "center_circle_face": "#80cdc1",
        "center_text": "#000000",
    },
    20: {
        "heatmap_negative": "#8dd3c7",
        "heatmap_zero": "#ffffb3",
        "heatmap_positive": "#fb8072",
        "center_circle_face": "#80b1d3",
        "center_text": "#ffffff",
    },
}

COLOR_CHOICE = 20  # 选择颜色方案
selected_colors = color_library[COLOR_CHOICE]  # 获取选定的颜色配置


def get_cmap_from_selection(colors):
    nodes = [0.0, 0.5, 1.0]  # 定义颜色渐变的节点位置，0为负值，0.5为中间值，1为正值
    colors_list = [
        colors["heatmap_negative"],
        colors["heatmap_zero"],
        colors["heatmap_positive"],
    ]  # 提取颜色配置构建列表
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_corr_cmap", list(zip(nodes, colors_list))
    )  # 生成自定义 Colormap
    return cmap  # 返回生成的颜色映射对象


current_cmap = get_cmap_from_selection(selected_colors)  # 根据当前选定的颜色方案生成颜色映射
norm = mcolors.Normalize(vmin=-1, vmax=1)  # 设置颜色映射的归一化范围


# =========================================================================================
# ======================================3.数据分析函数==============================
# =========================================================================================
def analyze_raw_data(raw_data_path, output_analysis_path, years, vars_list):
    calculated_data = {}  # 用于存储计算结果
    with pd.ExcelWriter(
        output_analysis_path
    ) as writer:  # 使用 pandas创建Excel写入对象，准备保存文件
        for year in years:  # 遍历每一个年份
            df_all = pd.read_excel(
                raw_data_path, sheet_name=f"{year}_RawData", index_col=0
            )  # 读取对应年份的Sheet
            df_vars = df_all[vars_list]  # 特征数据
            center_series = df_all["SHDI"]  # 目标数据
            # 特征数据之间进行相关行分析
            corr_matrix = df_vars.corr(method="pearson")
            # 用于存储 P 值
            p_values = pd.DataFrame(index=vars_list, columns=vars_list, dtype=float)
            for c1 in vars_list:  # 行
                for c2 in vars_list:  # 列
                    if c1 == c2:  # 如果是同一个变量
                        p_values.loc[c1, c2] = 0.0
                    else:
                        _, p = pearsonr(df_vars[c1], df_vars[c2])  # 计算两列数据相关系数和 P 值
                        p_values.loc[c1, c2] = p  # 将计算得到的P值填入矩阵对应位置
            center_corrs = []  # 用于存储特征与目标的相关性
            for var in vars_list:  # 遍历每一个变量
                r, _ = pearsonr(df_vars[var], center_series)  # 计算当前变量与目标的相关系数
                center_corrs.append(r)  # 将相关系数添加到列表
            # 将相关性列表转换为DataFrame格式
            df_center_corr = pd.DataFrame(
                center_corrs, index=vars_list, columns=["Correlation_with_Center"]
            )

            # 保存分析结果
            corr_matrix.to_excel(writer, sheet_name=f"{year}_Corr")  # 相关性
            p_values.to_excel(writer, sheet_name=f"{year}_P_Value")  # P值
            df_center_corr.to_excel(writer, sheet_name=f"{year}_Center_Corr")  # 中心相关性数据
            # 将计算结果直接存入字典
            calculated_data[year] = {
                "corr": corr_matrix.values,
                "p": p_values.values,
                "r": df_center_corr["Correlation_with_Center"].values,
            }
        return calculated_data  # 直接返回计算好的数据，供后续使用


# =========================================================================================
# ======================================4.网络线线线条设置函数=====================================
# =========================================================================================
def get_line_style(r_value):
    abs_r = abs(r_value)  # 计算相关系数的绝对值
    c_pos = selected_colors["heatmap_positive"]  # 正相关使用的颜色
    c_neg = selected_colors["heatmap_negative"]  # 负相关使用的颜色
    # 确定颜色，根据 r 值正负选择
    if r_value >= 0:
        line_color = c_pos
    else:
        line_color = c_neg
    # 根据绝对值大小确定线宽、线型和透明度
    if abs_r < 0.10:
        return {"color": line_color, "linestyle": "--", "linewidth": 1.0, "alpha": 0.5}
    elif 0.10 <= abs_r < 0.25:
        return {"color": line_color, "linestyle": "-", "linewidth": 1.5, "alpha": 0.65}
    elif 0.25 <= abs_r < 0.50:
        return {"color": line_color, "linestyle": "-", "linewidth": 3.0, "alpha": 0.8}
    else:
        return {"color": line_color, "linestyle": "-", "linewidth": 5.0, "alpha": 1.0}


# =========================================================================================
# ======================================5.热图绘制函数=====================================
# =========================================================================================
def draw_triangle_heatmap(
    ax, corr_mat, p_mat, variables, start_x, start_y, type="bottom-left", title_year=""
):
    n = len(variables)  # 变量的数量
    connection_points = []  # 用于存储连接线的锚点坐标
    # 左三角
    if type == "top-left":  # 如果是左上角的倒三角
        for i in range(n):  # 遍历每一行
            cols_to_draw = n - i  # 计算当前行需要绘制的列数，逐行递减
            for j_visual in range(cols_to_draw):  # 遍历当前行的每一列
                col_data_idx = n - 1 - j_visual  # 映射数据列索引，倒序
                row_data_idx = i  # 设置数据行索引
                val = corr_mat[row_data_idx, col_data_idx]  # 从相关性矩阵获取对应的值
                p = p_mat[row_data_idx, col_data_idx]  # 从 P 值矩阵获取对应的值
                rect_x = start_x + j_visual  # 计算方块的 X 轴坐标
                rect_y = start_y - i  # 计算方块的 Y 轴坐标
                # 绘制方块
                rect = patches.Rectangle(
                    (rect_x, rect_y), 1, 1, facecolor=current_cmap(norm(val)), edgecolor="white"
                )
                ax.add_patch(rect)  # 将矩形添加到轴上

                # 标注文本
                text_color = "white" if abs(val) > 0.6 else "black"  # 根据背景色深浅决定文字颜色
                # 绘制相关系数数值
                ax.text(
                    rect_x + 0.5,
                    rect_y + 0.35,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=text_color,
                    fontweight="normal",
                )
                # 设置显著性标记
                if p < 0.05:
                    mark = (
                        "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    )  # 根据 P 值大小确定星号数量
                    # 绘制显著性星号
                    ax.text(
                        rect_x + 0.5,
                        rect_y + 0.52,
                        mark,
                        ha="center",
                        va="center",
                        fontsize=15,
                        color=text_color,
                        fontweight="bold",
                    )

        # 左侧Y轴标签
        for i in range(n):  # 遍历行，绘制 Y 轴标签
            label_y = start_y - i + 0.5  # 标签的 Y 坐标
            # 绘制左侧的变量名
            ax.text(
                start_x - 0.2,
                label_y,
                variables[i],
                ha="right",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            # 锚点
            conn_x = start_x + (n - 1 - i) + 1  # 计算该行对应的连接锚点X坐标
            conn_y = start_y - i + 0.5  # 计算该行对应的连接锚点Y标
            connection_points.append((conn_x, conn_y))  # 将锚点坐标加入列表

        # 顶部标签，X轴
        top_labels = variables[::-1]  # 将变量列表反转，用于顶部 X 轴标签
        for i in range(n):  # 遍历列
            label_x = start_x + i + 0.5  # 标签的 X 坐标
            # 绘制顶部的变量标签
            ax.text(
                label_x,
                start_y + 1.2,
                top_labels[i],
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        # 年份标注
        ax.text(
            start_x,
            start_y + 1.2,
            title_year,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # 右上
    elif type == "top-right":
        # 遍历行
        for i in range(n):
            cols_count = n - i  # 计算当前行方块数
            offset = i  # 计算每行的水平偏移量
            for j_visual in range(cols_count):  # 遍历列
                rect_x = start_x + offset + j_visual  # 方块的 X 轴坐标
                rect_y = start_y - i  # 方块的 Y 轴坐标
                row_data_idx = i  # 数据行索引
                col_data_idx = i + j_visual  # 数据列索引
                val = corr_mat[row_data_idx, col_data_idx]  # 获取相关系数数值
                p = p_mat[row_data_idx, col_data_idx]  # 获取 P 值
                # 创建方块
                rect = patches.Rectangle(
                    (rect_x, rect_y), 1, 1, facecolor=current_cmap(norm(val)), edgecolor="white"
                )
                ax.add_patch(rect)  # 添加
                text_color = "white" if abs(val) > 0.6 else "black"  # 设置文本颜色
                # 绘制数值
                ax.text(
                    rect_x + 0.5,
                    rect_y + 0.35,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=text_color,
                    fontweight="normal",
                )
                if p < 0.05:  # 判断显著性
                    mark = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    # 绘制
                    ax.text(
                        rect_x + 0.5,
                        rect_y + 0.52,
                        mark,
                        ha="center",
                        va="center",
                        fontsize=15,
                        color=text_color,
                        fontweight="bold",
                    )

        # 右侧标签，Y轴
        for i in range(n):  # 遍历行
            label_y = start_y - i + 0.5  # 标签 Y 坐标
            # 绘制
            ax.text(
                start_x + n + 0.2,
                label_y,
                variables[i],
                ha="left",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

            # 锚点
            conn_x = start_x + i + 0  # X 坐标
            conn_y = start_y - i + 0.5  # Y 坐标
            connection_points.append((conn_x, conn_y))  # 添加

        # 顶部标签 ，X轴
        for i in range(n):  # 遍历列
            label_x = start_x + i + 0.5  # X 坐标
            # 绘制顶部变量名
            ax.text(
                label_x,
                start_y + 1.2,
                variables[i],
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        # 年份
        ax.text(
            start_x + n,
            start_y + 1.2,
            title_year,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # 左下角
    elif type == "bottom-left":
        for i in range(n):  # 行
            for j in range(i + 1):  # 列
                rect_x = start_x + j  # 方块 X 坐标
                rect_y = start_y - i  # 方块 Y 坐标
                val = corr_mat[i, j]  # 相关性数值
                p = p_mat[i, j]  # 获取 P 值
                # 创建方块
                rect = patches.Rectangle(
                    (rect_x, rect_y), 1, 1, facecolor=current_cmap(norm(val)), edgecolor="white"
                )
                ax.add_patch(rect)  # 添加

                text_color = "white" if abs(val) > 0.6 else "black"  # 文字颜色
                # 绘制相关数值
                ax.text(
                    rect_x + 0.5,
                    rect_y + 0.35,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color=text_color,
                    fontweight="normal",
                )
                if p < 0.05:  # 显著性
                    mark = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    ax.text(
                        rect_x + 0.5,
                        rect_y + 0.52,
                        mark,
                        ha="center",
                        va="center",
                        fontsize=15,
                        color=text_color,
                        fontweight="bold",
                    )

        # 左侧标签，Y轴
        for i in range(n):  # 行
            label_y = start_y - i + 0.5  # Y 坐标
            # 绘制
            ax.text(
                start_x - 0.2,
                label_y,
                variables[i],
                ha="right",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

            # 锚点
            conn_x = start_x + i + 1  # X 坐标
            conn_y = start_y - i + 0.5  # Y 坐标
            connection_points.append((conn_x, conn_y))  # 添加

        # 底部标签，X轴
        for i in range(n):  # 遍历列
            label_x = start_x + i + 0.5  # 标签 X 坐标
            # 绘制变量
            ax.text(
                label_x,
                start_y - (n - 1) - 0.5,
                variables[i],
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
            )
        # 绘制年份
        ax.text(
            start_x,
            start_y - (n - 1) - 0.4,
            title_year,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
    return connection_points


# =========================================================================================
# ======================================6.主绘图函数=====================================
# =========================================================================================
def create_complex_layout_plot(data, vars_list):
    fig, ax = plt.subplots(figsize=(20, 16))  # 创建画布和坐标轴
    ax.set_aspect("equal")  # 强制纵横比相等，保证方块是正方形
    n = len(vars_list)  # 变量数
    gap_width = 2.0  # 设置中间间距宽度

    start_x_2000 = -gap_width / 2 - n  # 左上热图起始X坐标
    start_y_2000 = n + 2  # 左上热图起始Y坐标

    start_x_2010 = -gap_width / 2 - n  # 右下热图起始X坐标
    start_y_2010 = 0  # 右下热图起始Y坐标

    start_x_2020 = gap_width / 2  # 右上热图起始X坐标
    start_y_2020 = start_y_2000  # 右上热图起始Y坐标

    # 绘制三个热图
    pts_2000 = draw_triangle_heatmap(
        ax,
        data[2000]["corr"],
        data[2000]["p"],
        vars_list,
        start_x=start_x_2000,
        start_y=start_y_2000,
        type="top-left",
        title_year="2000",
    )

    pts_2020 = draw_triangle_heatmap(
        ax,
        data[2020]["corr"],
        data[2020]["p"],
        vars_list,
        start_x=start_x_2020,
        start_y=start_y_2020,
        type="top-right",
        title_year="2020",
    )

    pts_2010 = draw_triangle_heatmap(
        ax,
        data[2010]["corr"],
        data[2010]["p"],
        vars_list,
        start_x=start_x_2010,
        start_y=start_y_2010,
        type="bottom-left",
        title_year="2010",
    )

    # 绘制中心节点
    center_y = ((start_y_2000 - n) + (start_y_2010 + 1)) / 2  # 中心圆的 Y 坐标
    center_x = 0  # 中心圆的 X 坐标

    c_face = selected_colors["center_circle_face"]  # 中心圆的填充色
    c_text = selected_colors["center_text"]  # 中心圆的文本色

    # 绘制中心大圆
    ax.scatter(
        center_x, center_y, s=6000, marker="o", facecolor=c_face, edgecolor="none", zorder=100
    )

    # 绘制中心文字
    # 绘制圆内第一行文字
    ax.text(
        center_x,
        center_y + 0.25,
        "SHDI",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color=c_text,
        zorder=101,
    )
    # 绘制圆内第二行文字
    ax.text(
        center_x,
        center_y,
        "Pastoral area",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=c_text,
        zorder=101,
    )

    # 绘制连接线函数
    def draw_connections(points, year_data):
        for i, pt in enumerate(points):  # 遍历每个锚点点
            r_val = year_data["r"][i]  # 获取该变量对应的Y相关性值
            style = get_line_style(r_val)  # 根据相关性获取线条样式

            con = patches.ConnectionPatch(
                xyA=pt,
                xyB=(center_x, center_y),
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,  # 设置坐标系
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                alpha=style.get("alpha", 1.0),
                zorder=1,
            )
            ax.add_patch(con)  # 将连接线添加到图中
            # 在锚点处画一个小圆点装饰
            ax.scatter(pt[0], pt[1], s=40, facecolor="white", edgecolor="gray", zorder=20)

    draw_connections(pts_2000, data[2000])  # 绘制2000连接线
    draw_connections(pts_2020, data[2020])  # 绘制2020连接线
    draw_connections(pts_2010, data[2010])  # 绘制2010连接线

    ax.set_xlim(start_x_2000 - 3, start_x_2020 + n + 3)  # X 轴显示范围
    ax.set_ylim(start_y_2010 - n - 2, start_y_2000 + 3)  # Y 轴显示范围
    ax.axis("off")  # 隐藏边框和刻度

    # 颜色条
    cbar_ax = fig.add_axes([0.7, 0.25, 0.013, 0.25])  # 在图中添加一个新的坐标轴用于绘制颜色条
    # 创建颜色条对象
    cb = colorbar.ColorbarBase(cbar_ax, cmap=current_cmap, norm=norm, orientation="vertical")
    cb.set_label("Pearson's r", size=18)  # 设置颜色条标题
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])  # 颜色条刻度
    cb.outline.set_visible(False)  # 去掉边框
    cb.ax.tick_params(size=0, labelsize=18)  # 设置刻度参数

    # 图例
    c_pos = selected_colors["heatmap_positive"]  # 正相关颜色
    c_neg = selected_colors["heatmap_negative"]  # 负相关颜色

    # 定义图例项，分为两组
    legend_elements = [
        # 正相关组
        Line2D([0], [0], color=c_pos, lw=1.0, linestyle="--", alpha=0.5, label="Positive < 0.10"),
        Line2D(
            [0], [0], color=c_pos, lw=1.5, linestyle="-", alpha=0.65, label="Positive 0.10 - 0.25"
        ),
        Line2D(
            [0], [0], color=c_pos, lw=3.0, linestyle="-", alpha=0.8, label="Positive 0.25 - 0.50"
        ),
        Line2D([0], [0], color=c_pos, lw=5.0, linestyle="-", alpha=1.0, label="Positive > 0.50"),
        # 负相关组
        Line2D([0], [0], color=c_neg, lw=1.0, linestyle="--", alpha=0.5, label="Negative > -0.10"),
        Line2D(
            [0],
            [0],
            color=c_neg,
            lw=1.5,
            linestyle="-",
            alpha=0.65,
            label="Negative -0.10 to -0.25",
        ),
        Line2D(
            [0], [0], color=c_neg, lw=3.0, linestyle="-", alpha=0.8, label="Negative -0.25 to -0.50"
        ),
        Line2D([0], [0], color=c_neg, lw=5.0, linestyle="-", alpha=1.0, label="Negative < -0.50"),
    ]

    # 绘制图例
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        bbox_to_anchor=(0.76, 0.2),
        title="Correlation Network (Lines)",
        frameon=False,
        fontsize=14,
        title_fontsize=16,
        ncol=1,
    )
    # 小标题
    ax.text(start_x_2020 + n - 4, start_y_2010 - n + 1, "(a)", fontsize=24, fontweight="bold")
    # 保存

    plt.savefig(
        str(OUTPUT_DIR / f"combined_heatmap{COLOR_CHOICE}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(str(OUTPUT_DIR / f"combined_heatmap{COLOR_CHOICE}.pdf"), bbox_inches="tight")


# =========================================================================================
# ======================================7. 主程序 =======================================
# =========================================================================================

if __name__ == "__main__":
    vars_list = ["CS", "FP", "HQ", "SR", "WY"]  # 特征
    years = [2000, 2010, 2020]  # 表
    raw_data_file = str(DATA_DIR / "raw_data.xlsx")  # 原始数据文件完整路径
    analysis_result_file = str(OUTPUT_DIR / "simulation_results.xlsx")  # 分析结果文件完整路径
    # 调用分析函数
    plot_data = analyze_raw_data(raw_data_file, analysis_result_file, years, vars_list)
    # 调用绘图函数
    create_complex_layout_plot(plot_data, vars_list)

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

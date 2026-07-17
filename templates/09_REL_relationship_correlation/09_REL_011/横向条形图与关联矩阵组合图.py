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

import os  # 导入 os 模块，用于处理文件和目录路径

import matplotlib  # 导入 matplotlib 主库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于创建图表，通常简写为 plt
import matplotlib.ticker as mticker  # 导入 matplotlib 的 ticker 模块，用于更精细地控制坐标轴刻度
import matplotlib.transforms as transforms  # 从 matplotlib.transforms 模块导入变换功能，用于控制图形元素的位置和形状
import numpy as np  # 导入 NumPy 库，用于高效的数值计算，通常简写为 np
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析，特别是DataFrame的使用
from matplotlib.gridspec import (
    GridSpec,  # 从 matplotlib.gridspec 模块导入 GridSpec，用于创建更灵活的子图网格布局
)
from matplotlib.patches import Ellipse  # 从 matplotlib.patches 模块导入 Ellipse 类，用于绘制椭圆
from scipy.stats import (
    pearsonr,  # 从 scipy.stats 模块导入 pearsonr 函数，用于计算皮尔逊相关系数和p值
)

matplotlib.use(
    "Agg"
)  # 设置 matplotlib 的图形后端为 'TkAgg'，以确保在某些交互式环境中能正确显示窗口


# --- 1. 第一个图表（条形图）的函数 ---
def plot_bar_chart(save_path=None):  # 定义绘制条形图的函数，可选参数 save_path 用于指定保存路径
    """
    绘制第一个条形图的函数, 并根据提供的路径保存图片。
    参数:
    - save_path (str, optional): 图片保存的完整路径。如果为 None, 则不保存。
    """
    # --- 准备数据 ---
    categories = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]  # 定义条形图的类别标签
    p_values = [
        0.34,
        0.07,
        0.01,
        0.51,
        0.68,
        0.38,
        0.33,
        0.21,
        0.47,
    ]  # 定义与每个类别对应的数值（P值）
    colors = [  # 定义每个条形图的颜色
        "#f8c4a0",
        "#fdebd0",
        "#a93226",
        "#827717",
        "#85c1e9",
        "#f9e79f",
        "#2e4053",
        "#e74c3c",
        "#28b463",
    ]
    # --- 全局样式设置 ---
    plt.rcParams["font.family"] = "serif"  # 设置全局默认的字体家族为衬线字体 (serif)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams[
        "font.serif"
    ]  # 在衬线字体列表中，优先使用 'Times New Roman'
    # --- 创建图表 ---
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )  # 创建一个图窗(fig)和一个子图(ax)，并设置图窗的尺寸为10x7英寸
    # --- 绘制水平条形图 ---
    bars = ax.barh(
        categories[::-1], p_values[::-1], color=colors[::-1]
    )  # 绘制水平条形图，[::-1]用于将列表反转，使图表从上到下按原始顺序显示
    # --- 自定义坐标轴和标签 ---
    ax.set_xlabel("P value", fontsize=14)  # 设置X轴的标签为'P value'，并指定字体大小
    ax.set_ylabel(
        "2020", fontsize=18, fontweight="bold", rotation=90, labelpad=10
    )  # 设置Y轴的标签为'2020'，并设置字体大小、粗细、旋转角度及与轴的间距
    ax.set_xlim(0, 0.8)  # 设置X轴的显示范围从0到0.8
    ax.set_xticks([i / 10 for i in range(9)])  # 设置X轴的刻度位置为0, 0.1, 0.2, ..., 0.8
    # --- 美化图表样式 ---
    ax.spines["top"].set_visible(False)  # 将顶部边框设置为不可见
    ax.spines["right"].set_visible(False)  # 将右侧边框设置为不可见
    ax.spines["bottom"].set_linewidth(1.5)  # 设置底部边框的线宽为1.5
    ax.spines["left"].set_linewidth(1.5)  # 设置左侧边框的线宽为1.5
    ax.xaxis.set_minor_locator(
        mticker.MultipleLocator(0.02)
    )  # 设置X轴的次要刻度定位器，每隔0.02单位一个次要刻度
    ax.tick_params(
        axis="x", which="major", direction="out", length=5, width=1.5
    )  # 设置X轴主刻度线：方向朝外，长度为5，宽度为1.5
    ax.tick_params(
        axis="x", which="minor", direction="out", length=3, width=1.5
    )  # 设置X轴次要刻度线：方向朝外，长度为3，宽度为1.5
    ax.yaxis.set_minor_locator(
        mticker.AutoMinorLocator(2)
    )  # 设置Y轴次要刻度自动定位器，每个主刻度之间分2个次刻度
    ax.tick_params(
        axis="y", which="major", direction="out", length=5, width=1.5
    )  # 设置Y轴主刻度线：方向朝外，长度为5，宽度为1.5
    ax.tick_params(
        axis="y", which="minor", direction="out", length=3, width=1.5
    )  # 设置Y轴次要刻度线：方向朝外，长度为3，宽度为1.5
    ax.tick_params(axis="both", labelsize=12)  # 将X轴和Y轴的刻度标签字体大小都设置为12
    ax.yaxis.grid(
        True, which="minor", linestyle="--", color="black", linewidth=0.3
    )  # 在Y轴的次要刻度上显示网格线，样式为虚线，颜色为黑，线宽为0.3
    ax.set_axisbelow(True)  # 设置网格线绘制在图表元素的下方，防止遮挡条形图
    # --- 在每个条形图右侧添加数值标签 ---
    for bar in bars:  # 遍历图中的每一个条形对象
        width = bar.get_width()  # 获取当前条形的宽度（即其对应的P值）
        ax.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",  # 在指定位置添加文本标签
            va="center",
            ha="left",
            fontsize=12,
        )  # 设置文本的垂直对齐、水平对齐和字体大小
    # --- 保存和显示图表 ---
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图窗区域，避免标签重叠
    if save_path:  # 如果提供了保存路径
        try:  # 尝试执行保存操作
            plt.savefig(
                save_path, dpi=300, bbox_inches="tight"
            )  # 保存图表，设置分辨率为300dpi，并裁剪掉空白边缘
            print(f"条形图已保存到: {save_path}")  # 打印成功信息
        except Exception as e:  # 如果保存过程中出现任何异常
            print(f"保存条形图时出错: {e}")  # 打印错误信息
    plt.close("all")  # Interactive display removed; assets were exported above.


# --- 2. 第二个图表（关联矩阵）的函数 ---
def plot_association_matrix(
    matrix_df, p_values_df, figsize=(11, 12), cmap_name="bwr_r", save_path=None
):  # 定义绘制关联矩阵的函数
    """
    根据指定的数值矩阵和P值矩阵，绘制一个可视化的关联矩阵图, 并保存。
    参数:
    - ... (其他参数保持不变)
    - save_path (str, optional): 图片保存的完整路径。如果为 None, 则不保存。
    """
    n_vars = len(matrix_df.columns)  # 获取变量的数量，即矩阵的维度
    vmin = -1.0  # 定义颜色映射的最小值
    vmax = 1.0  # 定义颜色映射的最大值
    norm = plt.Normalize(
        vmin=vmin, vmax=vmax
    )  # 创建一个归一化对象，将数据值从[vmin, vmax]线性映射到[0, 1]
    cmap = plt.get_cmap(cmap_name)  # 根据名称获取一个颜色映射（colormap）对象
    fig = plt.figure(figsize=figsize)  # 创建一个图窗对象，并设置其尺寸
    gs = GridSpec(
        2, 1, height_ratios=[30, 1], hspace=0.1
    )  # 创建一个2行1列的网格，两行的高度比例为30:1，行间距为0.1
    ax = fig.add_subplot(gs[0, 0])  # 在网格的第一个位置（第0行，第0列）添加主图的子图
    cbar_ax = fig.add_subplot(gs[1, 0])  # 在网格的第二个位置（第1行，第0列）添加颜色条的子图
    ax.set_xlim(-0.5, n_vars - 0.5)  # 设置X轴的显示范围
    ax.set_ylim(-0.5, n_vars - 0.5)  # 设置Y轴的显示范围
    ax.invert_yaxis()  # 反转Y轴，使(0,0)坐标位于图表的左上角
    ax.set_xticks(np.arange(n_vars))  # 设置X轴主刻度的位置
    ax.set_yticks(np.arange(n_vars))  # 设置Y轴主刻度的位置
    ax.set_xticklabels(
        matrix_df.columns, fontfamily="serif", fontsize=12, fontweight="bold"
    )  # 设置X轴刻度标签，并指定字体、大小和粗细
    ax.set_yticklabels(
        matrix_df.columns, fontfamily="serif", fontsize=12, fontweight="bold"
    )  # 设置Y轴刻度标签，并指定字体、大小和粗细
    ax.set_xticks(
        np.arange(n_vars + 1) - 0.5, minor=True
    )  # 设置X轴次要刻度的位置，用于绘制垂直网格线
    ax.set_yticks(
        np.arange(n_vars + 1) - 0.5, minor=True
    )  # 设置Y轴次要刻度的位置，用于绘制水平网格线
    ax.grid(
        which="minor", color="black", linestyle="-", linewidth=1
    )  # 在次要刻度上绘制网格线，颜色为黑，样式为实线，线宽为1
    ax.tick_params(
        which="major", top=False, bottom=False, left=False, right=False
    )  # 隐藏主刻度线（与标签对应的刻度线）
    ax.tick_params(
        which="minor", top=False, bottom=False, left=False, right=False
    )  # 隐藏次要刻度线（与网格线对应的刻度线）
    for spine in ax.spines.values():  # 遍历子图的所有边框（上下左右）
        spine.set_linewidth(1.5)  # 设置边框的线宽为1.5
    for i in range(n_vars):  # 遍历矩阵的行
        for j in range(n_vars):  # 遍历矩阵的列
            r = matrix_df.iloc[i, j]  # 获取在(i, j)位置的数值
            p = p_values_df.iloc[i, j]  # 获取在(i, j)位置的p值
            color = cmap(norm(r))  # 根据数值和归一化规则计算出对应的颜色
            if i == j:  # 如果是主对角线上的单元格
                ax.text(
                    j,
                    i,
                    matrix_df.columns[i],
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",  # 在单元格中心添加变量名
                    fontfamily="serif",
                )
            elif i > j:  # 如果是下三角部分的单元格
                if r != 0:  # 对于非0值
                    ax.text(
                        j,
                        i,
                        f"{r:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=12,
                        fontweight="bold",  # 在单元格中心添加格式化后的数值
                        fontfamily="serif",
                    )
            else:  # 如果是上三角部分的单元格
                if r != 0:  # 对于非0值
                    r_geom = np.clip(
                        r, -1.0, 1.0
                    )  # 将数值裁剪到[-1, 1]范围内，仅用于计算椭圆几何形状
                    ellipse = Ellipse(
                        xy=(j, i),  # 创建一个椭圆对象，中心在(j, i)
                        width=np.sqrt(1 + abs(r_geom)) * 0.9,  # 根据数值计算椭圆宽度
                        height=np.sqrt(1 - abs(r_geom)) * 0.9,  # 根据数值计算椭圆高度
                        angle=45 if r > 0 else -45,  # 如果数值为正，椭圆倾斜45度；为负，则倾斜-45度
                        facecolor=color,  # 设置椭圆的填充颜色
                        edgecolor="none",
                    )  # 不设置椭圆的边框颜色
                    ax.add_patch(ellipse)  # 将创建的椭圆添加到子图中
                    if p < 0.05:  # 如果p值小于0.05（通常表示统计上显著）
                        ax.text(
                            j,
                            i,
                            "*",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=24,
                            fontweight="bold",  # 在单元格中心添加一个星号
                            fontfamily="serif",
                        )
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=norm
    )  # 创建一个可映射标量对象，它包含颜色映射和归一化信息
    cbar = plt.colorbar(
        sm, cax=cbar_ax, orientation="horizontal", ticks=np.linspace(-1, 1, 11)
    )  # 在指定的子图位置(cbar_ax)创建颜色条
    cbar.set_label(
        "Correlation Coefficient", fontsize=12, fontfamily="serif", fontweight="bold"
    )  # 设置颜色条的标签及其样式
    cbar.outline.set_linewidth(1.5)  # 设置颜色条外边框的线宽
    cbar.ax.tick_params(axis="x", direction="out", length=5, width=1.5)  # 设置颜色条刻度线的样式
    if save_path:  # 如果提供了保存路径
        try:  # 尝试执行保存操作
            plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 保存图表
            print(f"关联矩阵图已保存到: {save_path}")  # 打印成功信息
        except Exception as e:  # 如果保存过程中出现任何异常
            print(f"保存关联矩阵图时出错: {e}")  # 打印错误信息
    plt.close("all")  # Interactive display removed; assets were exported above.


def plot_and_save_combined_figure(
    matrix_df, p_values_df, save_path=None
):  # 定义绘制并保存组合图的函数
    """
    将条形图和关联矩阵图合并到一张图中，并根据需要保存。
    左侧为条形图，右侧为关联矩阵图。
    参数:
    - matrix_df (pd.DataFrame): 用于矩阵图的相关性矩阵。
    - p_values_df (pd.DataFrame): 用于矩阵图的P值矩阵。
    - save_path (str, optional): 组合图的保存路径。
    """
    print("\n--- 正在创建组合图 ---")  # 打印提示信息
    # --- 1. 全局样式设置 ---
    plt.rcParams["font.family"] = "serif"  # 设置全局默认的字体家族为衬线字体
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams[
        "font.serif"
    ]  # 优先使用 'Times New Roman' 字体
    # --- 2. 创建组合图布局 ---
    # 使用 GridSpec 创建一个灵活的网格布局
    # 2行2列，右侧列的宽度是左侧的1.2倍，底部行的高度很小，用于放置颜色条
    fig = plt.figure(figsize=(22, 11))  # 创建一个大的图窗
    gs = GridSpec(
        2,
        2,  # 定义网格为2行2列
        height_ratios=[30, 1.5],  # 定义两行的高度比例
        width_ratios=[1, 1.2],  # 定义两列的宽度比例
        hspace=0.1,
        wspace=0.05,
    )  # 定义行间距和列间距
    # 定义子图位置
    ax_bar = fig.add_subplot(gs[:, 0])  # 条形图占据左侧整列（所有行，第0列）
    ax_matrix = fig.add_subplot(gs[0, 1])  # 矩阵图占据右上方的单元格（第0行，第1列）
    cbar_ax = fig.add_subplot(gs[1, 1])  # 颜色条占据右下方的单元格（第1行，第1列）
    # --- 3. (A) 绘制左侧的条形图 ---
    # 这部分逻辑直接从 `plot_bar_chart` 函数中提取并适配到 `ax_bar`
    ax_bar.set_title(
        "A", loc="left", fontsize=20, fontweight="bold"
    )  # 在左侧子图左上角添加标题 "A"
    # 准备数据
    categories = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"]  # 定义类别
    p_values = [0.34, 0.07, 0.01, 0.51, 0.68, 0.38, 0.33, 0.21, 0.47]  # 定义P值
    colors = [
        "#f8c4a0",
        "#fdebd0",
        "#a93226",
        "#827717",
        "#85c1e9",
        "#f9e79f",
        "#2e4053",
        "#e74c3c",
        "#28b463",
    ]  # 定义颜色
    # 绘制
    bars = ax_bar.barh(
        categories[::-1], p_values[::-1], color=colors[::-1]
    )  # 在左侧子图(ax_bar)上绘制水平条形图
    # 自定义坐标轴和标签
    ax_bar.set_xlabel("P value", fontsize=14)  # 设置X轴标签
    ax_bar.set_ylabel(
        "2020", fontsize=18, fontweight="bold", rotation=90, labelpad=10
    )  # 设置Y轴标签
    ax_bar.set_xlim(0, 0.8)  # 设置X轴范围
    ax_bar.set_xticks([i / 10 for i in range(9)])  # 设置X轴刻度
    # 美化图表样式
    ax_bar.spines["top"].set_visible(False)  # 隐藏上边框
    ax_bar.spines["right"].set_visible(False)  # 隐藏右边框
    ax_bar.spines["bottom"].set_linewidth(1.5)  # 设置下边框线宽
    ax_bar.spines["left"].set_linewidth(1.5)  # 设置左边框线宽
    ax_bar.xaxis.set_minor_locator(mticker.MultipleLocator(0.02))  # 设置X轴次要刻度
    ax_bar.tick_params(
        axis="x", which="major", direction="out", length=5, width=1.5
    )  # 设置X轴主刻度线样式
    ax_bar.tick_params(
        axis="x", which="minor", direction="out", length=3, width=1.5
    )  # 设置X轴次要刻度线样式
    ax_bar.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))  # 设置Y轴次要刻度
    ax_bar.tick_params(
        axis="y", which="major", direction="out", length=5, width=1.5
    )  # 设置Y轴主刻度线样式
    ax_bar.tick_params(
        axis="y", which="minor", direction="out", length=3, width=1.5
    )  # 设置Y轴次要刻度线样式
    ax_bar.tick_params(axis="both", labelsize=12)  # 设置两个轴的刻度标签大小
    ax_bar.yaxis.grid(
        True, which="minor", linestyle="--", color="black", linewidth=0.3
    )  # 显示Y轴次要刻度网格线
    ax_bar.set_axisbelow(True)  # 将网格线置于图表元素之下
    # 添加数值标签
    for bar in bars:  # 遍历每个条形
        width = bar.get_width()  # 获取条形宽度
        ax_bar.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",  # 在条形右侧添加数值
            va="center",
            ha="left",
            fontsize=12,
        )
    # --- 4. (B) 绘制右侧的关联矩阵图 ---
    # 这部分逻辑直接从 `plot_association_matrix` 函数中提取并适配到 `ax_matrix` 和 `cbar_ax`
    ax_matrix.set_title(
        "B", loc="left", fontsize=20, fontweight="bold"
    )  # 在右侧子图左上角添加标题 "B"
    n_vars = len(matrix_df.columns)  # 获取变量数量
    vmin = -1.0  # 定义颜色映射最小值
    vmax = 1.0  # 定义颜色映射最大值
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # 创建归一化对象
    cmap = plt.get_cmap("bwr_r")  # 获取 'bwr_r' 颜色映射
    ax_matrix.set_xlim(-0.5, n_vars - 0.5)  # 设置X轴范围
    ax_matrix.set_ylim(-0.5, n_vars - 0.5)  # 设置Y轴范围
    ax_matrix.invert_yaxis()  # 反转Y轴
    ax_matrix.set_xticks(np.arange(n_vars))  # 设置X轴主刻度
    ax_matrix.set_yticks(np.arange(n_vars))  # 设置Y轴主刻度
    ax_matrix.set_xticklabels(
        matrix_df.columns, fontfamily="serif", fontsize=12, fontweight="bold"
    )  # 设置X轴刻度标签
    ax_matrix.set_yticklabels(
        matrix_df.columns, fontfamily="serif", fontsize=12, fontweight="bold"
    )  # 设置Y轴刻度标签
    ax_matrix.set_xticks(np.arange(n_vars + 1) - 0.5, minor=True)  # 设置X轴次要刻度（用于网格线）
    ax_matrix.set_yticks(np.arange(n_vars + 1) - 0.5, minor=True)  # 设置Y轴次要刻度（用于网格线）
    ax_matrix.grid(which="minor", color="black", linestyle="-", linewidth=1)  # 显示次要刻度网格线
    ax_matrix.tick_params(
        which="major", top=False, bottom=False, left=False, right=False
    )  # 隐藏主刻度线
    ax_matrix.tick_params(
        which="minor", top=False, bottom=False, left=False, right=False
    )  # 隐藏次要刻度线
    for spine in ax_matrix.spines.values():  # 遍历所有边框
        spine.set_linewidth(1.5)  # 设置边框线宽
    for i in range(n_vars):  # 遍历行
        for j in range(n_vars):  # 遍历列
            r = matrix_df.iloc[i, j]  # 获取相关性值
            p = p_values_df.iloc[i, j]  # 获取P值
            color = cmap(norm(r))  # 获取对应颜色
            if i == j:  # 如果在对角线上
                ax_matrix.text(
                    j,
                    i,
                    matrix_df.columns[i],
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",  # 写变量名
                    fontfamily="serif",
                )
            elif i > j:  # 如果在下三角
                if r != 0:  # 如果值不为0
                    ax_matrix.text(
                        j,
                        i,
                        f"{r:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=12,  # 写数值
                        fontweight="bold",
                        fontfamily="serif",
                    )
            else:  # 如果在上三角
                if r != 0:  # 如果值不为0
                    r_geom = np.clip(r, -1.0, 1.0)  # 裁剪数值
                    ellipse = Ellipse(
                        xy=(j, i),  # 创建椭圆
                        width=np.sqrt(1 + abs(r_geom)) * 0.9,  # 设置宽度
                        height=np.sqrt(1 - abs(r_geom)) * 0.9,  # 设置高度
                        angle=45 if r > 0 else -45,  # 设置角度
                        facecolor=color,
                        edgecolor="none",
                    )  # 设置颜色
                    ax_matrix.add_patch(ellipse)  # 添加椭圆到子图
                    if p < 0.05:  # 如果P值显著
                        ax_matrix.text(
                            j,
                            i,
                            "*",
                            ha="center",
                            va="center",
                            color="black",
                            fontsize=24,  # 添加星号
                            fontweight="bold",
                            fontfamily="serif",
                        )
    # 绘制颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # 创建可映射标量对象
    cbar = plt.colorbar(
        sm, cax=cbar_ax, orientation="horizontal", ticks=np.linspace(-1, 1, 11)
    )  # 在指定位置创建颜色条
    cbar.set_label(
        "Correlation Coefficient", fontsize=12, fontfamily="serif", fontweight="bold"
    )  # 设置颜色条标签
    cbar.outline.set_linewidth(1.5)  # 设置颜色条边框线宽
    cbar.ax.tick_params(axis="x", direction="out", length=5, width=1.5)  # 设置颜色条刻度线样式
    # --- 5. 保存和显示组合图 ---
    if save_path:  # 如果提供了保存路径
        try:  # 尝试保存
            # 使用 fig.savefig() 而不是 plt.savefig() 来保存整个图窗
            fig.savefig(save_path, dpi=300, bbox_inches="tight")  # 保存图窗
            print(f"组合图已成功保存到: {save_path}")  # 打印成功信息
        except Exception as e:  # 如果出错
            print(f"保存组合图时出错: {e}")  # 打印错误信息
    plt.close("all")  # Interactive display removed; assets were exported above.


# --- 在主程序中调用新增的组合函数 ---
if __name__ == "__main__":  # 判断是否是主程序入口
    # --- 定义文件和文件夹路径 ---
    output_folder = str(OUTPUT_DIR)  # 定义输出文件夹路径
    input_excel_file = os.path.join(output_folder, "simulated_data.xlsx")  # 定义输入文件路径
    # --- 绘制并保存第一个图 ---
    print("正在绘制第一个条形图...")  # 打印提示
    bar_chart_save_path = os.path.join(
        output_folder, str(OUTPUT_DIR / "bar_chart.png")
    )  # 定义第一个图的保存路径
    plot_bar_chart(save_path=bar_chart_save_path)
    # --- 从Excel文件加载数据 ---
    print(f"\n正在从 {input_excel_file} 读取数据...")  # 打印提示
    try:  # 尝试读取和处理
        df = pd.read_excel(input_excel_file)  # 读取Excel
        print("数据读取成功。")  # 打印成功信息
        # --- 计算相关性矩阵和P值矩阵 ---
        print("正在计算相关性矩阵和P值...")  # 打印提示
        matrix_df = df.corr()  # 计算相关性矩阵
        p_values_df = pd.DataFrame(
            np.ones((df.shape[1], df.shape[1])), columns=df.columns, index=df.columns
        )  # 初始化P值矩阵
        for col1 in df.columns:  # 遍历列
            for col2 in df.columns:  # 再次遍历列
                if col1 == col2:  # 跳过自身
                    continue
                _, p = pearsonr(df[col1], df[col2])  # 计算P值
                p_values_df.loc[col1, col2] = p  # 存储P值
        print("计算完成。")  # 打印成功信息
        # --- 绘制并保存第二个图 ---
        print("\n正在绘制第二个关联矩阵图...")  # 打印提示
        association_matrix_save_path = os.path.join(
            output_folder, str(OUTPUT_DIR / "association_matrix.png")
        )  # 定义第二个图的保存路径
        plot_association_matrix(
            matrix_df, p_values_df, figsize=(10, 11), save_path=association_matrix_save_path
        )
        combined_save_path = os.path.join(
            output_folder, str(OUTPUT_DIR / "combined_plot.png")
        )  # 定义组合图的保存路径
        plot_and_save_combined_figure(
            matrix_df, p_values_df, save_path=combined_save_path
        )  # 调用函数绘制、显示并保存组合图
    except FileNotFoundError:  # 捕获文件未找到错误
        print(
            f"错误: 找不到文件 {input_excel_file}。请确认文件是否存在于指定路径。"
        )  # 打印错误信息
    except Exception as e:  # 捕获其他所有错误
        print(f"处理数据或绘图时发生错误: {e}")  # 打印错误信息

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

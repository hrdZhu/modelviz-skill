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
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib_venn import venn3, venn3_circles

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# =========================================================================================
# ====================================== 2.数据分析函数=========================================
# =========================================================================================
def analyze_perception_data(df):
    charts_config = {
        "a": {
            "factors": ["OCV_neg", "OSV_neg", "OPV_neg"],  # 因子列
            "target": "OEV_neg",  # 目标变量列名
            "title": "a) Overall perception",  # 图表标题
            "legend": [
                "Whole combinations of OCV, OSV and OPV",
                "Uncomfortable in OCV",
                "Unsafe in OSV",
                "Unpleasant in OPV",
            ],  # 图例的文本
        },
        "b": {
            "factors": ["TCV_neg", "ACV_neg", "VCV_neg"],
            "target": "OCV_neg_result",
            "title": "b) Overall comfort",
            "legend": [
                "Whole combinations of TCV, ACV and VCV",
                "Uncomfortable in TCV",
                "Uncomfortable in ACV",
                "Uncomfortable in VCV",
            ],
        },
        "c": {
            "factors": ["ASV_neg", "DSV_neg", "PSV_neg"],
            "target": "OSV_neg_result",
            "title": "c) Overall safety",
            "legend": [
                "Whole combinations of ASV, DSV and PSV",
                "Unsafe in ASV",
                "Unsafe in DSV",
                "Unsafe in PSV",
            ],
        },
        "d": {
            "factors": ["SPV_neg", "CPV_neg", "NPV_neg"],
            "target": "OPV_neg_result",
            "title": "d) Overall pleasure",
            "legend": [
                "Whole combinations of SPV, CPV and NPV",
                "Unpleasant in SPV",
                "Unpleasant in CPV",
                "Unpleasant in NPV",
            ],
        },
    }
    analyzed_data = {}  # 用于存储分析后的数据结果
    for key, config in charts_config.items():  # 遍历每一个图表配置
        grouped = df.groupby(config["factors"])[
            config["target"]
        ].mean()  # 根据配置的因子对数据进行分组，并计算目标变量的平均值
        print(grouped)
        values = {}  # 初始化一个字典，用于存储韦恩图各部分的百分比数值
        # 格式化百分比
        values["top"] = f"{grouped[(1, 0, 0)] * 100:.2f}%"  # 顶部圆独有部分的均值
        values["left"] = f"{grouped[(0, 1, 0)] * 100:.2f}%"  # 左下圆独有部分的均值
        values["right"] = f"{grouped[(0, 0, 1)] * 100:.2f}%"  # 右下圆独有部分的均值
        values["left_top"] = f"{grouped[(1, 1, 0)] * 100:.2f}%"  # 左上交叉部分的均值
        values["right_top"] = f"{grouped[(1, 0, 1)] * 100:.2f}%"  # 右上交叉部分的均值
        values["left_right"] = f"{grouped[(0, 1, 1)] * 100:.2f}%"  # 下方左右交叉部分的均值
        values["center"] = f"{grouped[(1, 1, 1)] * 100:.2f}%"  # 中心三圆交叉部分的均值
        outside_val = f"{grouped[(0, 0, 0)] * 100:.2f}%"  # 三个因子均为0时的均值，外部
        analyzed_data[key] = {  # 将分析结果存入字典中
            "title": config["title"],  # 标题
            "values": values,  # 各部分数值
            "outside": outside_val,  # 外部数值
            "legend": config["legend"],  # 图例文本
        }
        print(values)
    return analyzed_data  # 返回分析数据


# =========================================================================================
# ====================================== 3.颜色库 =========================================
# =========================================================================================
COLOR_LIBRARY = {
    1: {
        "m1": "#FFFFFF",
        "m2": "#FFF5EB",
        "m3": "#EBF4FA",
        "m4": "#F5FAFE",
        "m5": "#F9CB9C",
        "m6": "#9BC2E6",
        "m7": "#E0E0E0",
        "m8": "#E6F2FF",
        "border_top": "#F48024",
        "border_left": "#2E75B6",
        "border_right": "#5B9BD5",
        "border_gray": "#A0A0A0",
    },
    2: {
        "m1": "#FFFFFF",
        "m2": "#FFEBEE",
        "m3": "#FFF3E0",
        "m4": "#FFFDE7",
        "m5": "#E57373",
        "m6": "#FFB74D",
        "m7": "#FFF176",
        "m8": "#FF8A65",
        "border_top": "#D32F2F",
        "border_left": "#F57C00",
        "border_right": "#FBC02D",
        "border_gray": "#BDBDBD",
    },
    3: {
        "m1": "#FFFFFF",
        "m2": "#E3F2FD",
        "m3": "#E0F2F1",
        "m4": "#F3E5F5",
        "m5": "#64B5F6",
        "m6": "#4DB6AC",
        "m7": "#BA68C8",
        "m8": "#7986CB",
        "border_top": "#1976D2",
        "border_left": "#009688",
        "border_right": "#7B1FA2",
        "border_gray": "#9E9E9E",
    },
    4: {
        "m1": "#FFFFFF",
        "m2": "#E8F5E9",
        "m3": "#F1F8E9",
        "m4": "#DCEDC8",
        "m5": "#81C784",
        "m6": "#AED581",
        "m7": "#DCE775",
        "m8": "#4CAF50",
        "border_top": "#388E3C",
        "border_left": "#689F38",
        "border_right": "#AFB42B",
        "border_gray": "#8D6E63",
    },
    5: {
        "m1": "#FFFFFF",
        "m2": "#F3E5F5",
        "m3": "#EDE7F6",
        "m4": "#F8BBD0",
        "m5": "#BA68C8",
        "m6": "#9575CD",
        "m7": "#F06292",
        "m8": "#AB47BC",
        "border_top": "#7B1FA2",
        "border_left": "#512DA8",
        "border_right": "#C2185B",
        "border_gray": "#9E9E9E",
    },
    6: {
        "m1": "#FFFFFF",
        "m2": "#ECEFF1",
        "m3": "#E3F2FD",
        "m4": "#CFD8DC",
        "m5": "#90A4AE",
        "m6": "#64B5F6",
        "m7": "#B0BEC5",
        "m8": "#607D8B",
        "border_top": "#455A64",
        "border_left": "#1976D2",
        "border_right": "#546E7A",
        "border_gray": "#78909C",
    },
    7: {
        "m1": "#FFFFFF",
        "m2": "#E0F7FA",
        "m3": "#B2EBF2",
        "m4": "#B3E5FC",
        "m5": "#4DD0E1",
        "m6": "#4FC3F7",
        "m7": "#81D4FA",
        "m8": "#00BCD4",
        "border_top": "#0097A7",
        "border_left": "#0288D1",
        "border_right": "#039BE5",
        "border_gray": "#B0BEC5",
    },
    8: {
        "m1": "#FFFFFF",
        "m2": "#FFCDD2",
        "m3": "#BBDEFB",
        "m4": "#FFF9C4",
        "m5": "#EF9A9A",
        "m6": "#90CAF9",
        "m7": "#FFF59D",
        "m8": "#CE93D8",
        "border_top": "#E53935",
        "border_left": "#1E88E5",
        "border_right": "#FDD835",
        "border_gray": "#BDBDBD",
    },
    9: {
        "m1": "#FFFFFF",
        "m2": "#D7CCC8",
        "m3": "#F0F4C3",
        "m4": "#FFE0B2",
        "m5": "#A1887F",
        "m6": "#DCE775",
        "m7": "#FFB74D",
        "m8": "#8D6E63",
        "border_top": "#5D4037",
        "border_left": "#AFB42B",
        "border_right": "#F57C00",
        "border_gray": "#795548",
    },
    10: {
        "m1": "#FFFFFF",
        "m2": "#F9FBE7",
        "m3": "#F0F4C3",
        "m4": "#FFFDE7",
        "m5": "#DCE775",
        "m6": "#C5E1A5",
        "m7": "#FFF176",
        "m8": "#AED581",
        "border_top": "#AFB42B",
        "border_left": "#689F38",
        "border_right": "#FBC02D",
        "border_gray": "#BDBDBD",
    },
    11: {
        "m1": "#FFFFFF",
        "m2": "#C5CAE9",
        "m3": "#D1C4E9",
        "m4": "#B39DDB",
        "m5": "#7986CB",
        "m6": "#9575CD",
        "m7": "#673AB7",
        "m8": "#5C6BC0",
        "border_top": "#303F9F",
        "border_left": "#512DA8",
        "border_right": "#4527A0",
        "border_gray": "#757575",
    },
    12: {
        "m1": "#FFFFFF",
        "m2": "#FFAB91",
        "m3": "#80CBC4",
        "m4": "#B2DFDB",
        "m5": "#FF7043",
        "m6": "#4DB6AC",
        "m7": "#26A69A",
        "m8": "#009688",
        "border_top": "#D84315",
        "border_left": "#00796B",
        "border_right": "#00695C",
        "border_gray": "#9E9E9E",
    },
    13: {
        "m1": "#FFFFFF",
        "m2": "#F8BBD0",
        "m3": "#E1BEE7",
        "m4": "#FFCDD2",
        "m5": "#F06292",
        "m6": "#BA68C8",
        "m7": "#E57373",
        "m8": "#D81B60",
        "border_top": "#C2185B",
        "border_left": "#7B1FA2",
        "border_right": "#D32F2F",
        "border_gray": "#BDBDBD",
    },
    14: {
        "m1": "#FFFFFF",
        "m2": "#FFE0B2",
        "m3": "#D7CCC8",
        "m4": "#FFECB3",
        "m5": "#FFB74D",
        "m6": "#A1887F",
        "m7": "#FFD54F",
        "m8": "#FF9800",
        "border_top": "#F57C00",
        "border_left": "#5D4037",
        "border_right": "#FFA000",
        "border_gray": "#8D6E63",
    },
    15: {
        "m1": "#FFFFFF",
        "m2": "#CFD8DC",
        "m3": "#B0BEC5",
        "m4": "#90A4AE",
        "m5": "#78909C",
        "m6": "#607D8B",
        "m7": "#546E7A",
        "m8": "#455A64",
        "border_top": "#37474F",
        "border_left": "#263238",
        "border_right": "#102027",
        "border_gray": "#78909C",
    },
    16: {
        "m1": "#FFFFFF",
        "m2": "#FFFF8D",
        "m3": "#84FFFF",
        "m4": "#FF80AB",
        "m5": "#FBC02D",
        "m6": "#00E5FF",
        "m7": "#FF4081",
        "m8": "#D500F9",
        "border_top": "#F57F17",
        "border_left": "#00B8D4",
        "border_right": "#C51162",
        "border_gray": "#9E9E9E",
    },
    17: {
        "m1": "#FFFFFF",
        "m2": "#F5F5DC",
        "m3": "#FAF0E6",
        "m4": "#FFF8DC",
        "m5": "#BDB76B",
        "m6": "#D2B48C",
        "m7": "#DEB887",
        "m8": "#DAA520",
        "border_top": "#556B2F",
        "border_left": "#8B4513",
        "border_right": "#CD853F",
        "border_gray": "#A9A9A9",
    },
    18: {
        "m1": "#FFFFFF",
        "m2": "#D3C4BE",
        "m3": "#C0C5C1",
        "m4": "#C4CCD4",
        "m5": "#A69086",
        "m6": "#89948B",
        "m7": "#8F9DA8",
        "m8": "#7D8893",
        "border_top": "#745C52",
        "border_left": "#5D6860",
        "border_right": "#5A6875",
        "border_gray": "#9E9E9E",
    },
    19: {
        "m1": "#FFFFFF",
        "m2": "#FFCCBC",
        "m3": "#C8E6C9",
        "m4": "#FFF9C4",
        "m5": "#FF7043",
        "m6": "#66BB6A",
        "m7": "#FFEE58",
        "m8": "#8D6E63",
        "border_top": "#BF360C",
        "border_left": "#2E7D32",
        "border_right": "#F9A825",
        "border_gray": "#795548",
    },
    20: {
        "m1": "#FFFFFF",
        "m2": "#E0E0E0",
        "m3": "#EA80FC",
        "m4": "#8C9EFF",
        "m5": "#AA00FF",
        "m6": "#304FFE",
        "m7": "#00B0FF",
        "m8": "#6200EA",
        "border_top": "#D50000",
        "border_left": "#C51162",
        "border_right": "#2962FF",
        "border_gray": "#212121",
    },
}

SELECTED_SCHEME = 20  # 选择配色方案
palette = COLOR_LIBRARY.get(SELECTED_SCHEME, COLOR_LIBRARY[1])  # 获取配色方案


# =========================================================================================
# ====================================== 4.韦恩图绘制函数=========================================
# =========================================================================================
def draw_venn_unit(ax, data_dict):
    vals = data_dict["values"]  # 提取分析的数值
    colors = palette  # 获取当前选定的颜色方案

    # 绘制基础韦恩图
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), ax=ax, set_labels=None)
    # 绘制韦恩图的圆周轮廓线
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), ax=ax, linewidth=1.5)

    # 绕中心旋转 180 度，将倒三角变为正三角
    tr = Affine2D().rotate_deg(180) + ax.transData
    # 定义韦恩图各部分ID及颜色的映射关系
    mapping = {
        "001": {"key": "top", "color": colors["m2"]},
        "010": {"key": "left", "color": colors["m3"]},
        "100": {"key": "right", "color": colors["m4"]},
        "011": {"key": "left_top", "color": colors["m5"]},
        "101": {"key": "right_top", "color": colors["m6"]},
        "110": {"key": "left_right", "color": colors["m7"]},
        "111": {"key": "center", "color": colors["m8"]},
    }
    # 逐个处理韦恩图的各个区域
    for pid, info in mapping.items():
        patch = v.get_patch_by_id(pid)  # 获取对应的图形块
        label = v.get_label_by_id(pid)  # 获取对应的标签
        if patch:
            patch.set_transform(tr)  # 对图形块应用旋转变换
            if info.get("color"):
                patch.set_facecolor(info["color"])  # 设置图形块的填充颜色
                patch.set_alpha(1.0)  # 设置透明度
        if label:
            x, y = label.get_position()  # 获取标签当前的坐标位置
            new_x, new_y = -x, -y  # 坐标反转
            # 微调文字位置
            if pid == "001":
                new_y -= 0.05
            if pid == "110":
                new_y += 0.05
            if pid == "111":
                new_y -= 0.02
            label.set_position((new_x, new_y))
            label.set_text(vals[info["key"]])  # 设置标签文本为对应的数值
            label.set_fontsize(9)  # 设置字体大小
            label.set_fontweight("bold")  # 设置字体加粗
    # 遍历所有的圆轮廓
    for circle in c:
        circle.set_transform(tr)  # 对圆轮廓应用旋转变换
    # 设置的边框颜色
    c[0].set_edgecolor(colors["border_right"])
    c[1].set_edgecolor(colors["border_left"])
    c[2].set_edgecolor(colors["border_top"])
    # 外围圆圈
    outer_circle = Circle(
        (0, 0), radius=0.75, transform=ax.transData, fill=False, edgecolor="gray", linewidth=0.8
    )
    ax.add_patch(outer_circle)  # 添加到坐标轴上
    # 外围圆圈数值
    ax.text(0, -0.65, data_dict["outside"], ha="center", va="center", fontsize=9, fontweight="bold")

    # 绘制右侧图例
    legend_left = 1.0  # 图例左边界的x坐标
    start_y = 0.35  # 图例起始的y坐标
    step_y = 0.25  # 图例之间的垂直间距
    r = 0.06  # 图例小圆圈的半径

    # 图例属性设置，文本, 填充色, 边框色
    items = [
        (data_dict["legend"][0], "none", colors["border_gray"]),
        (data_dict["legend"][1], "none", colors["border_top"]),
        (data_dict["legend"][2], "none", colors["border_left"]),
        (data_dict["legend"][3], "none", colors["border_right"]),
    ]
    # 循环绘制
    for i, (text, face, edge) in enumerate(items):
        y_pos = start_y - i * step_y  # 当前图例的y坐标
        # 绘制圆圈
        ax.add_patch(Circle((legend_left, y_pos), r, facecolor=face, edgecolor=edge, linewidth=1.5))
        # 绘制文字
        ax.text(legend_left + 0.15, y_pos, text, va="center", fontsize=11)
    # 绘制子图标题
    ax.text(
        0.25,
        -0.05,
        data_dict["title"],
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=14,
    )

    ax.set_xlim(-0.8, 2.5)  # x轴显示范围
    ax.set_ylim(-0.8, 0.8)  # y轴显示范围
    ax.axis("off")  # 关闭坐标轴显示
    ax.set_aspect("equal")  # 设置坐标轴比例为1:1


# =========================================================================================
# ====================================== 5.子图绘制及图例设置函数=========================================
# =========================================================================================
def plot_and_save_individual(data_charts):
    palette_name = f"palette_{SELECTED_SCHEME}"  # 文件名
    for key, data_dict in data_charts.items():  # 遍历所有分析好的数据
        fig, ax = plt.subplots(figsize=(8, 6))  # 创建图形
        draw_venn_unit(ax, data_dict)  # 调用绘图函数绘制韦恩图
        # 添加一个新的坐标轴用于绘制图例
        legend_ax = fig.add_axes(
            [
                0.1,  # 左
                0.2,  # 下
                0.8,  # 宽
                0.05,
            ]
        )  # 高
        legend_ax.axis("off")
        colors = palette  # 获取当前选定的颜色方案
        m_legend_items = [  # 定义底部图例项列表
            ("m=1", colors["m1"], "lightgray"),
            ("m=2", colors["m2"], "lightgray"),
            ("m=3", colors["m3"], "lightgray"),
            ("m=4", colors["m4"], "lightgray"),
            ("m=5", colors["m5"], "lightgray"),
            ("m=6", colors["m6"], "lightgray"),
            ("m=7", colors["m7"], "lightgray"),
            ("m=8", colors["m8"], "lightgray"),
        ]
        num_items = len(m_legend_items)  # 图例的总数
        item_width = 1.0 / num_items  # 每个图例的宽度
        # 绘制图例
        for i, (label, fill, edge) in enumerate(m_legend_items):
            x_pos = i * item_width + 0.02  # 计算当前项的x起始位置
            # 创建图例的图形
            rect = Rectangle(
                (x_pos, 0.2),
                0.06,
                0.75,
                facecolor=fill,
                edgecolor=edge,
                transform=legend_ax.transAxes,
                zorder=20,
            )
            legend_ax.add_patch(rect)  # 添加到坐标轴
            # 图例的文本
            legend_ax.text(
                x_pos + 0.065,
                0.65,
                label,
                transform=legend_ax.transAxes,
                va="center",
                fontsize=12,
                fontweight="bold",
            )

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.35)  # 调整子图布局边距
        # 保存
        save_path_png = str(OUTPUT_DIR / f"chart_{key}_{palette_name}.png")
        save_path_pdf = str(OUTPUT_DIR / f"chart_{key}_{palette_name}.pdf")
        plt.savefig(save_path_png, dpi=300)
        plt.savefig(save_path_pdf, dpi=300)
        plt.close(fig)


# =========================================================================================
# ====================================== 6.数据读取分析、子图及组合图绘制执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    excel_filename = str(DATA_DIR / "simulated_research_data.xlsx")  # 数据文件
    raw_df = pd.read_excel(excel_filename)  # 读取
    all_charts_data = analyze_perception_data(raw_df)  # 调用分析函数处理数据
    print("--- 开始绘制单张子图 ---")
    plot_and_save_individual(all_charts_data)  # 调用函数绘制
    # 绘制组合图
    keys_to_draw = ["a", "b", "c", "d"]  # 定义需要包含在组合图中的图表key列表
    num_plots = len(keys_to_draw)  # 子图数量
    cols = 2  # 列
    rows = math.ceil(num_plots / cols)  # 行数
    fig_width = 8 * cols  # 总宽度
    fig_height = 5 * rows  # 总高度
    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    axes_flat = axes.flatten()  # 将多维axes数组展平为一维

    # 遍历每一个子图区域并调用绘图函数
    for i, ax in enumerate(axes_flat):  # 遍历所有的子图坐标轴
        if i < num_plots:  # 如果索引在需要绘制的图表范围内
            key = keys_to_draw[i]  # 获取对应的图表key
            draw_venn_unit(ax, all_charts_data[key])  # >>> 调用核心绘图函数在当前子图区域绘图 <<<
        else:  # 如果超出范围（即空余的子图位）
            ax.axis("off")  # 关闭该子图显示

    # 3. 在大画布底部绘制通用图例 (内嵌逻辑)
    legend_ax = fig.add_axes([0.1, 0.02, 0.8, 0.05])  # 添加一个新的坐标轴用于绘制图例
    legend_ax.axis("off")  # 关闭该坐标轴的显示
    colors = palette  # 获取当前选定的颜色方案
    m_legend_items = [  # 定义底部图例项
        ("m=1", colors["m1"], "lightgray"),
        ("m=2", colors["m2"], "lightgray"),
        ("m=3", colors["m3"], "lightgray"),
        ("m=4", colors["m4"], "lightgray"),
        ("m=5", colors["m5"], "lightgray"),
        ("m=6", colors["m6"], "lightgray"),
        ("m=7", colors["m7"], "lightgray"),
        ("m=8", colors["m8"], "lightgray"),
    ]
    num_items_combo = len(m_legend_items)  # 获取图例项的总数
    item_width_combo = 1.0 / num_items_combo  # 计算每个图例项的宽度

    for i, (label, fill, edge) in enumerate(m_legend_items):  # 遍历底部图例进行绘制
        x_pos = i * item_width_combo + 0.02  # x起始位置
        rect = Rectangle(
            (x_pos, 0.2), 0.06, 0.6, facecolor=fill, edgecolor=edge, transform=legend_ax.transAxes
        )
        legend_ax.add_patch(rect)  # 添加到坐标轴
        legend_ax.text(
            x_pos + 0.08,
            0.5,
            label,
            transform=legend_ax.transAxes,
            va="center",
            fontsize=13,
            fontweight="bold",
        )
    # 调整整体布局和间距
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.15 / rows, wspace=0.01, hspace=-0.1
    )
    # 保存
    save_path_png = str(OUTPUT_DIR / f"combined_{SELECTED_SCHEME}.png")
    save_path_pdf = str(OUTPUT_DIR / f"combined_{SELECTED_SCHEME}.pdf")
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf, dpi=300)

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

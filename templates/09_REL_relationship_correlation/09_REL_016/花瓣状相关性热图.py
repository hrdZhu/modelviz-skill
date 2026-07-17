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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from scipy import stats

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"
# =========================================================================================
# ======================================2.颜色库设置=========================================
# =========================================================================================
COLOR_THEMES = {
    1: {
        "group_colors": {
            "TFs": "#3B5F8A",
            "VIs": "#E68B3A",
            "DVS": "#707070",
            "PI": "#A65E60",
            "MC": "#8A6B94",
        },
        "heatmap_colors": ["#43A8A8", "white", "#D75F5F"],
        "group_label_color": "white",
    },
    2: {
        "group_colors": {
            "TFs": "#2E7D32",
            "VIs": "#FF8F00",
            "DVS": "#616161",
            "PI": "#8D6E63",
            "MC": "#6A1B9A",
        },
        "heatmap_colors": ["#388E3C", "white", "#D32F2F"],
        "group_label_color": "white",
    },
    3: {
        "group_colors": {
            "TFs": "#4A148C",
            "VIs": "#AD1457",
            "DVS": "#455A64",
            "PI": "#C2185B",
            "MC": "#004D40",
        },
        "heatmap_colors": ["#1976D2", "#FFFFFF", "#E91E63"],
        "group_label_color": "white",
    },
    4: {
        "group_colors": {
            "TFs": "#003B46",
            "VIs": "#07575B",
            "DVS": "#66A5AD",
            "PI": "#007580",
            "MC": "#0A918C",
        },
        "heatmap_colors": ["#0D47A1", "#FFFFFF", "#FFC107"],
        "group_label_color": "white",
    },
    5: {
        "group_colors": {
            "TFs": "#6D4C41",
            "VIs": "#BF360C",
            "DVS": "#795548",
            "PI": "#D84315",
            "MC": "#4E342E",
        },
        "heatmap_colors": ["#8D6E63", "#FFF3E0", "#E64A19"],
        "group_label_color": "white",
    },
    6: {
        "group_colors": {
            "TFs": "#42275A",
            "VIs": "#734B6D",
            "DVS": "#E86A92",
            "PI": "#F47983",
            "MC": "#6A1B9A",
        },
        "heatmap_colors": ["#311B92", "#FFECB3", "#E65100"],
        "group_label_color": "white",
    },
    7: {
        "group_colors": {
            "TFs": "#1B5E20",
            "VIs": "#2E7D32",
            "DVS": "#558B2F",
            "PI": "#7CB342",
            "MC": "#3E2723",
        },
        "heatmap_colors": ["#2E7D32", "#F1F8E9", "#BF360C"],
        "group_label_color": "white",
    },
    8: {
        "group_colors": {
            "TFs": "#1C3144",
            "VIs": "#30475E",
            "DVS": "#7E8A97",
            "PI": "#5C6F7F",
            "MC": "#046582",
        },
        "heatmap_colors": ["#263238", "#ECEFF1", "#0277BD"],
        "group_label_color": "white",
    },
    9: {
        "group_colors": {
            "TFs": "#311B92",
            "VIs": "#4527A0",
            "DVS": "#7B1FA2",
            "PI": "#9C27B0",
            "MC": "#F9A825",
        },
        "heatmap_colors": ["#4A148C", "#FFF8E1", "#FFAB00"],
        "group_label_color": "white",
    },
    10: {
        "group_colors": {
            "TFs": "#212121",
            "VIs": "#424242",
            "DVS": "#757575",
            "PI": "#9E9E9E",
            "MC": "#616161",
        },
        "heatmap_colors": ["#000000", "#FFFFFF", "#BDBDBD"],
        "group_label_color": "white",
    },
    11: {
        "group_colors": {
            "TFs": "#A2D0C1",
            "VIs": "#E2A0B7",
            "DVS": "#F6E5A8",
            "PI": "#B1A8D0",
            "MC": "#FFB085",
        },
        "heatmap_colors": ["#81C784", "#FFFFFF", "#CE93D8"],
        "group_label_color": "black",
    },
    12: {
        "group_colors": {
            "TFs": "#440154",
            "VIs": "#3B528B",
            "DVS": "#21908C",
            "PI": "#5DC863",
            "MC": "#FDE725",
        },
        "heatmap_colors": ["#0D0887", "#FFFFFF", "#F0F921"],
        "group_label_color": "white",
    },
    13: {
        "group_colors": {
            "TFs": "#93634B",
            "VIs": "#C88C5A",
            "DVS": "#E3B27D",
            "PI": "#A5492A",
            "MC": "#5D3A2D",
        },
        "heatmap_colors": ["#A1887F", "#FFFFFF", "#42A5F5"],
        "group_label_color": "white",
    },
    14: {
        "group_colors": {
            "TFs": "#009FB7",
            "VIs": "#FED766",
            "DVS": "#247BA0",
            "PI": "#F25F5C",
            "MC": "#70C1B3",
        },
        "heatmap_colors": ["#005969", "#FFFFFF", "#FF427E"],
        "group_label_color": "black",
    },
    15: {
        "group_colors": {
            "TFs": "#588157",
            "VIs": "#84A98C",
            "DVS": "#A3B18A",
            "PI": "#B38A58",
            "MC": "#3A5A40",
        },
        "heatmap_colors": ["#344E41", "#FFFFFF", "#BC6C25"],
        "group_label_color": "white",
    },
    16: {
        "group_colors": {
            "TFs": "#C36B84",
            "VIs": "#E69597",
            "DVS": "#F5B9B2",
            "PI": "#813745",
            "MC": "#59292F",
        },
        "heatmap_colors": ["#880E4F", "#FCE4EC", "#4E342E"],
        "group_label_color": "white",
    },
    17: {
        "group_colors": {
            "TFs": "#0B0C10",
            "VIs": "#1F2833",
            "DVS": "#45A29E",
            "PI": "#66FCF1",
            "MC": "#C5C6C7",
        },
        "heatmap_colors": ["#00003f", "#FFFFFF", "#7A0099"],
        "group_label_color": "white",
    },
    18: {
        "group_colors": {
            "TFs": "#4878CF",
            "VIs": "#6ACC65",
            "DVS": "#D65F5F",
            "PI": "#B47CC7",
            "MC": "#DC843D",
        },
        "heatmap_colors": ["#4C72B0", "#FFFFFF", "#DD8452"],
        "group_label_color": "white",
    },
    19: {
        "group_colors": {
            "TFs": "#F4B71B",
            "VIs": "#FA7921",
            "DVS": "#E55934",
            "PI": "#9BC53D",
            "MC": "#5BC0EB",
        },
        "heatmap_colors": ["#588157", "#FFFFFF", "#F9A620"],
        "group_label_color": "black",
    },
    20: {
        "group_colors": {
            "TFs": "#2D2D2A",
            "VIs": "#595954",
            "DVS": "#83837E",
            "PI": "#B46549",
            "MC": "#988880",
        },
        "heatmap_colors": ["#36454F", "#FFFFFF", "#B87333"],
        "group_label_color": "white",
    },
    21: {
        "group_colors": {
            "TFs": "#1A425D",
            "VIs": "#546A7B",
            "DVS": "#8C9BAB",
            "PI": "#C2CCDA",
            "MC": "#2B2D42",
        },
        "heatmap_colors": "coolwarm",
        "group_label_color": "white",
    },
    22: {
        "group_colors": {
            "TFs": "#440154",
            "VIs": "#31688E",
            "DVS": "#21918C",
            "PI": "#35B779",
            "MC": "#1F4060",
        },
        "heatmap_colors": "viridis",
        "group_label_color": "white",
    },
    23: {
        "group_colors": {
            "TFs": "#0D0887",
            "VIs": "#5B02A3",
            "DVS": "#9B179E",
            "PI": "#CC4678",
            "MC": "#4B0395",
        },
        "heatmap_colors": "plasma",
        "group_label_color": "white",
    },
    24: {
        "group_colors": {
            "TFs": "#000004",
            "VIs": "#57106E",
            "DVS": "#9C179E",
            "PI": "#ED6925",
            "MC": "#420A68",
        },
        "heatmap_colors": "inferno",
        "group_label_color": "white",
    },
    25: {
        "group_colors": {
            "TFs": "#000004",
            "VIs": "#3B0F70",
            "DVS": "#8C2981",
            "PI": "#DE4968",
            "MC": "#6A217C",
        },
        "heatmap_colors": "magma",
        "group_label_color": "white",
    },
    26: {
        "group_colors": {
            "TFs": "#00204E",
            "VIs": "#00528A",
            "DVS": "#4080A5",
            "PI": "#7AB0C2",
            "MC": "#2F4858",
        },
        "heatmap_colors": "cividis",
        "group_label_color": "white",
    },
    27: {
        "group_colors": {
            "TFs": "#053061",
            "VIs": "#2166AC",
            "DVS": "#92CDEE",
            "PI": "#67001F",
            "MC": "#4393C3",
        },
        "heatmap_colors": "bwr",
        "group_label_color": "white",
    },
    28: {
        "group_colors": {
            "TFs": "#003C30",
            "VIs": "#01665E",
            "DVS": "#5AB4AC",
            "PI": "#8E0152",
            "MC": "#35978F",
        },
        "heatmap_colors": "seismic",
        "group_label_color": "white",
    },
    29: {
        "group_colors": {
            "TFs": "#B2182B",
            "VIs": "#D6604D",
            "DVS": "#F4A582",
            "PI": "#4393C3",
            "MC": "#2166AC",
        },
        "heatmap_colors": "RdBu_r",
        "group_label_color": "white",
    },
    30: {
        "group_colors": {
            "TFs": "#00008F",
            "VIs": "#0090FF",
            "DVS": "#00FFFF",
            "PI": "#FFFF00",
            "MC": "#FF0000",
        },
        "heatmap_colors": "jet",
        "group_label_color": "white",
    },
    31: {
        "group_colors": {
            "TFs": "#23171B",
            "VIs": "#4D3C5D",
            "DVS": "#6A65A3",
            "PI": "#6895D1",
            "MC": "#53A196",
        },
        "heatmap_colors": "turbo",
        "group_label_color": "white",
    },
    32: {
        "group_colors": {
            "TFs": "#9400D3",
            "VIs": "#0000FF",
            "DVS": "#00FF00",
            "PI": "#FFFF00",
            "MC": "#FF0000",
        },
        "heatmap_colors": "gist_rainbow",
        "group_label_color": "white",
    },
    33: {
        "group_colors": {
            "TFs": "#00425A",
            "VIs": "#1F8A70",
            "DVS": "#BFDB38",
            "PI": "#FC7300",
            "MC": "#005B7F",
        },
        "heatmap_colors": "ocean",
        "group_label_color": "white",
    },
    34: {
        "group_colors": {
            "TFs": "#346B31",
            "VIs": "#779F63",
            "DVS": "#C4D69B",
            "PI": "#A49580",
            "MC": "#567D46",
        },
        "heatmap_colors": "terrain",
        "group_label_color": "white",
    },
    35: {
        "group_colors": {
            "TFs": "#2D274E",
            "VIs": "#4A4C7D",
            "DVS": "#6A74A8",
            "PI": "#8D9DD1",
            "MC": "#3A3A69",
        },
        "heatmap_colors": "cubehelix",
        "group_label_color": "white",
    },
    36: {
        "group_colors": {
            "TFs": "#000000",
            "VIs": "#550000",
            "DVS": "#AA0000",
            "PI": "#FF5500",
            "MC": "#FFAA00",
        },
        "heatmap_colors": "gnuplot",
        "group_label_color": "white",
    },
    37: {
        "group_colors": {
            "TFs": "#FADADD",
            "VIs": "#F9C4D2",
            "DVS": "#F7AEC1",
            "PI": "#F38FB8",
            "MC": "#E26D9F",
        },
        "heatmap_colors": "spring",
        "group_label_color": "black",
    },
    38: {
        "group_colors": {
            "TFs": "#C4DFBC",
            "VIs": "#A9D3B2",
            "DVS": "#8DC6A9",
            "PI": "#65B49F",
            "MC": "#4BA296",
        },
        "heatmap_colors": "summer",
        "group_label_color": "black",
    },
    39: {
        "group_colors": {
            "TFs": "#4B120B",
            "VIs": "#7B1B0B",
            "DVS": "#AF2B0B",
            "PI": "#E3440B",
            "MC": "#63150B",
        },
        "heatmap_colors": "autumn",
        "group_label_color": "white",
    },
    40: {
        "group_colors": {
            "TFs": "#0F3C58",
            "VIs": "#1E526C",
            "DVS": "#316C85",
            "PI": "#4989A1",
            "MC": "#255E7A",
        },
        "heatmap_colors": "winter",
        "group_label_color": "white",
    },
}
# =========================================================================================
# ======================================3.绘图前需要做的准备=========================================
# =========================================================================================
# 选择配色方案
selected_scheme = 39
# 选择分析方法spearman, pearson,kendall
selected_method = "spearman"
# 输入文件的地址，输出结果的路径
data_directory = str(DATA_DIR)
# 每个数据文件的特征和目标
################################这里要根据你得数据来修改####################################
data_slices = {
    "Flexural": {"features": slice(0, 16), "targets": slice(16, 24)},
    "Flexural-shear": {"features": slice(0, 16), "targets": slice(16, 24)},
    "Shear": {"features": slice(0, 16), "targets": slice(16, 24)},
    "Bond": {"features": slice(0, 16), "targets": slice(16, 24)},
}
# 从颜色库里提取配色方案，如果没有就是用默认的颜色
select_color = COLOR_THEMES.get(selected_scheme, 1)


# =========================================================================================
# ======================================4.绘图函数=========================================
# =========================================================================================
def create_full_ring_plot(
    all_data,  # 相关性分析结果
    all_feature_names,  # 包含所有特征名称的字典
    all_target_names,  # 目标名称
    color_palette,  # 配色方案
    sector_params,
):  # 每个扇区的几何参数（如角度）

    fig, ax = plt.subplots(figsize=(24, 24), subplot_kw={"aspect": "equal"})
    # 去掉坐标轴的显示，包括刻度和图框
    ax.axis("off")
    heatmap_colors_value = color_palette["heatmap_colors"]  # 从传入的调色板字典中获取热图所用的颜色

    # 判断热图颜色配置是Matplotlib自带的还是自定义的
    if isinstance(heatmap_colors_value, str):
        # 判断热图颜色配置是Matplotlib自带的还是自定义的
        cmap = plt.get_cmap(heatmap_colors_value)
    else:
        # 根据节点和颜色创建分段线性颜色映射
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", list(zip([0.0, 0.5, 1.0], heatmap_colors_value))
        )
    # 创建一个归一化对象，将相关性系数值从映射到[0, 1]区间，便于颜色映射
    norm = plt.Normalize(vmin=-1, vmax=1)

    # 获取所有分组的名称列表
    group_names = list(all_data.keys())
    # 根据第一个分组的目标数量来确定需要绘制多少层图
    num_targets = len(all_target_names[group_names[0]])
    # 生成一个从8开始的半径数组，用于绘制每一层图
    radii = np.arange(8, 8 + num_targets)

    # 设置不同组的目标变量标记的颜色
    group_legend_colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
    # 设置目标白能量标记的形状
    target_markers = ["o", "s", "^", "v", "D", "p", "h", "*"]

    # 开始进行绘图，循环绘制出4个扇形图
    for idx, group_name in enumerate(group_names):
        # 根据当前分组名，获取其对应的特征名称列表
        features = all_feature_names[group_name]

        # 获取相关系数和显著性
        df, df_sig = all_data[group_name]["correlation_df"], all_data[group_name]["p_value_df"]
        # 获取当前区域的目标名称
        current_targets = all_target_names[group_name]

        # 获取当前扇区的起始角度和结束角度
        start_angle_deg = sector_params[group_name]["start"]
        end_angle_deg = sector_params[group_name]["end"]

        # 在起始和结束角度之间，生成每一个特征的角度
        theta_deg = np.linspace(start_angle_deg, end_angle_deg, len(features))
        # 将角度从度转换为弧度
        theta_rad = np.deg2rad(theta_deg)

        # 每个特征所占的角度宽度
        angle_span_deg = abs(end_angle_deg - start_angle_deg) / len(features) * 0.95
        # 目标标记的颜色
        current_group_color = group_legend_colors[idx]

        # 遍历当前分组的每一个目标，用于绘制每一层图
        for i, target_name in enumerate(current_targets):
            # 获取当前层的内半径和外半径
            r_inner = radii[i]
            r_outer = radii[i] + 0.9
            # 当前目标与所有特征的相关性值
            values = df[target_name]
            # 对应的显著性判断结果（True/False）
            sig_values = df_sig[target_name]
            # 根据相关性值和颜色映射，获取每个单元格的颜色
            cell_colors = cmap(norm(values))

            # 将当前目标的标记的角度从度转换为弧度
            marker_angle_rad = np.deg2rad(sector_params[group_name]["marker_angle"])
            # 计算目标标记的半径位置，+0.45是为了接近中心
            marker_radius = r_inner + 0.45
            # 目标标记x坐标
            mx = marker_radius * np.cos(marker_angle_rad)
            # 目标标记的y坐标
            my = marker_radius * np.sin(marker_angle_rad)
            # 绘制目标标记
            ax.scatter(
                mx,  # x坐标
                my,  # y坐标
                marker=target_markers[i],  # 标记样式
                s=120,  # 大小
                c=current_group_color,  # 颜色
                edgecolor="black",  # 边缘颜色
                zorder=10,
            )

            # 遍历特征开始绘制每一层的每一列
            for j in range(len(features)):
                # 创建一个楔形，用于每一层的一个单元格
                wedge = Wedge(
                    center=(0, 0),  # 圆心坐标，因为都是围绕这圆形绘图的
                    r=r_outer,  # 外边缘半径
                    width=r_outer - r_inner,  # 厚度
                    theta1=theta_deg[j] - angle_span_deg / 2,  # 起始角度
                    theta2=theta_deg[j] + angle_span_deg / 2,  # 结束角度
                    facecolor=cell_colors[j],  # 填充颜色
                    edgecolor="white",  # 遍缘颜色
                    linewidth=0.7,
                )  # 线条宽度
                # 将创建的楔形添加到坐标轴上
                ax.add_patch(wedge)

                # 获取当前单元格中心点的角度
                text_angle_rad = theta_rad[j]
                # 相关性标注的半径位置
                text_radius = r_inner + 0.45
                # 相关性标注的x坐标
                x = text_radius * np.cos(text_angle_rad)
                # 相关性标注的y坐标
                y = text_radius * np.sin(text_angle_rad)

                # 当前单元格的相关性数值
                val = values.iloc[j]
                # 根据显著性结果确定标记
                sig_marker = "*" if sig_values.iloc[j] else ""
                # 格式化要显示的文本
                text_val = f"{val:.2f}{sig_marker}"

                # 计算文本的旋转角度，使其始终朝向圆外，方便阅读
                rot = theta_deg[j] - 90 if np.cos(text_angle_rad) > -0.01 else theta_deg[j] - 90
                # 添加标注
                ax.text(
                    x,
                    y,
                    text_val,
                    ha="center",
                    va="center",
                    fontsize=10,
                    rotation=rot,
                    rotation_mode="anchor",
                )

        # 计算最外圈特征的半径
        label_radius = radii.max() + 1.7
        # 遍历所有特征，在最外圈加上特征名称标注
        for i in range(len(features)):
            # 获取当前特征标注的角度
            text_angle_rad = theta_rad[i]
            # 特征标签的x坐标
            x = label_radius * np.cos(text_angle_rad)
            # 特征标签的y坐标
            y = label_radius * np.sin(text_angle_rad)
            # 计算特征标注的旋转角度
            rot = theta_deg[i] if np.cos(text_angle_rad) > -0.01 else theta_deg[i]  # - 180
            # 添加特征标注文本
            ax.text(
                x,
                y,
                features[i],
                ha="center",
                va="center",
                fontsize=12,
                rotation=rot,
                rotation_mode="anchor",
            )

        # 分组标题的角度，对应扇区的中心
        group_label_angle_deg = (start_angle_deg + end_angle_deg) / 2
        # 将角度从度转换为弧度
        group_label_angle_rad = np.deg2rad(group_label_angle_deg)
        # 分组标题的半径
        group_label_radius = radii.max() + 4.5
        # 分组标题的x坐标
        x = group_label_radius * np.cos(group_label_angle_rad)
        # 分组标题的y坐标
        y = group_label_radius * np.sin(group_label_angle_rad)
        # 添加分组标题文
        ax.text(
            x,
            y,
            group_name,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
        )

    # 添加每一组目标标记的形状图例
    # 定义四个图例在中心区域的位置
    legend_positions = [
        {"bbox_to_anchor": (0.5, 0.5), "loc": "lower right"},
        {"bbox_to_anchor": (0.5, 0.5), "loc": "lower left"},
        {"bbox_to_anchor": (0.5, 0.5), "loc": "upper right"},
        {"bbox_to_anchor": (0.5, 0.5), "loc": "upper left"},
    ]

    # 遍历每个分组，为其单独创建图例
    for i, group_name in enumerate(group_names):
        # 创建一个空列表，用于存放当前分组的图例句柄
        handles_for_group = []
        # 获取当前分组的目标名称列表
        current_targets = all_target_names[group_name]
        # 获取当前分组的标记的颜色
        current_group_color = group_legend_colors[i]

        # 遍历当前分组的每个目标，为其创建图例项
        for j, target_name in enumerate(current_targets):
            # 创建一个Line2D对象作为图例项的句柄，只显示标记，不显示线条
            handle = Line2D(
                [0],
                [0],  # 线条的(x, y)数据点
                marker=target_markers[j],  # 标记样子
                color="w",  # 线条的颜色
                label=target_name,  # 图例对应的标签文本
                markerfacecolor=current_group_color,  # 标记的填充颜色
                markeredgecolor="black",  # 标记的边缘颜色
                markersize=12,  # 标记的大小
                linestyle="None",
            )  # 线条的样式，None表示不显示线条，只显示标记
            # 将创建的句柄添加到列表中
            handles_for_group.append(handle)

        # 为当前分组创建一个图例
        leg = fig.legend(
            handles=handles_for_group,  # 使用上面创建的图例句柄
            ncol=2,  # 列
            title=group_name,  # 图例标题
            loc=legend_positions[i]["loc"],  # 图例在锚点的位置
            bbox_to_anchor=legend_positions[i]["bbox_to_anchor"],  # 图例的锚点（中心点）
            frameon=False,  # 不显示边框
            edgecolor="black",  # 边框颜色
            fontsize=12,  # 字体大小
            title_fontsize=14,  # 标题字体大小
            labelspacing=1.0,  # 标签之间的垂直间距
            columnspacing=1.0,  # 列之间的水平间距
            borderpad=0.8,
        )  # 内边距
        # 设置图例字体的粗细
        leg.get_title().set_fontweight("bold")

    # 设置颜色条
    # 在图形的指定位置创建一个新的坐标轴用于放置颜色条
    cax = fig.add_axes([0.2, 0.15, 0.6, 0.01])  # [左, 下, 宽, 高]
    # 色条使用的映射对象
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # 创建一个水平方向的颜色条
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    # 设置颜色条的标题和大小
    cbar.set_label("spearman", size=16)
    # 颜色条数值标注和刻度线
    cbar.ax.tick_params(size=14, labelsize=14)

    # x轴的显示范围
    ax.set_xlim(-20, 20)
    # y轴的显示范围
    ax.set_ylim(-20, 20)
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"{selected_scheme}_{selected_method}.png"),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(OUTPUT_DIR / f"{selected_scheme}_{selected_method}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# =========================================================================================
# ======================================5.执行函数=========================================
# =========================================================================================
def main():
    # 定义四个数据分组的名称，这需要与Excel文件名一致
    group_names = [
        "Flexural",
        "Flexural-shear",
        "Shear",
        "Bond",
    ]  ################################这里要根据你得数据来修改####################################
    # 定义每个扇区的几何参数起始角度、结束角度、目标标记角度
    sector_params = {
        group_names[0]: {"start": 100, "end": 170, "marker_angle": 175},
        group_names[1]: {"start": 10, "end": 80, "marker_angle": 85},
        group_names[2]: {"start": 280, "end": 350, "marker_angle": 355},
        group_names[3]: {"start": 190, "end": 260, "marker_angle": 265},
    }

    # 存储所有分组的相关性分析结果
    all_correlation_data = {}
    # 存储每个分组的特征名称
    all_features_dict = {}
    # 存储每个分组目标标签
    generic_targets_dict = {}

    # 设置一个标志，用于识别是否是正在处理的第一个分组
    is_first_group = True
    # 初始化特征数量为0
    num_features = 0
    # 初始化目标数量为0
    num_targets = 0

    # 遍历每个分组名称，读取并处理对应的文件
    for group in group_names:
        # 特征列
        feature_slice = data_slices[group]["features"]
        # 目标列
        target_slice = data_slices[group]["targets"]
        # 获取当前处理的文件的路径
        excel_path = os.path.join(data_directory, f"{group}.xlsx")

        # 读取数据
        data = pd.read_excel(excel_path)
        # 提取特征数据
        feature_data = data.iloc[:, feature_slice]
        # 提取目标数据
        target_data = data.iloc[:, target_slice]

        # 获取当前处理的文件的的特征和目标名称
        current_features = data.columns[feature_slice].tolist()
        current_targets = data.columns[target_slice].tolist()

        # 将当前分组的特征名称存入字典
        all_features_dict[group] = current_features
        # 将当前分组的目标名称存入字典
        generic_targets_dict[group] = current_targets

        # 如果这是第一个被成功处理的文件，记录下特征和目标的数量作为标准
        if is_first_group:
            num_features = len(current_features)
            num_targets = len(current_targets)
            is_first_group = False
        # 检查后续文件的数据维度是否与第一个文件匹配，以保证图形可以对齐
        elif len(current_features) != num_features or len(current_targets) != num_targets:
            print(f"文件 {excel_path} 的数据维度与第一个文件不匹配")
            # 从字典中移除刚刚添加的数据，以免出错
            del all_features_dict[group]
            del generic_targets_dict[group]
            continue

        # 初始化一个全为0的矩阵，用于存放相关性系数值
        correlation_matrix = np.zeros((num_features, num_targets))
        # 初始化一个全为0的矩阵，用于存放p值
        p_value_matrix = np.zeros((num_features, num_targets))

        # 遍历特征列
        for i in range(num_features):
            # 遍历目标列
            for j in range(num_targets):
                # 按索引提取当前要分析的特征列
                feature_col = feature_data.iloc[:, i]
                # 按索引提取当前要分析的目标列
                target_col = target_data.iloc[:, j]

                # 强制将特征列转换为数字格式，任何无法转换的值都会变成NaN
                feature_col_numeric = pd.to_numeric(feature_col, errors="coerce")
                # 强制将目标列转换为数字格式
                target_col_numeric = pd.to_numeric(target_col, errors="coerce")

                # 将转换后的两列数据合并成一个DataFrame，并移除任何包含NaN的行，以确保数据对齐且有效
                combined = pd.concat([feature_col_numeric, target_col_numeric], axis=1).dropna()

                # 如果有效数据点少于2个，则无法计算相关性
                if len(combined) < 2:
                    # 将相关系数和p值设为NaN
                    corr, p_value = np.nan, np.nan
                else:
                    # 如果有足够的数据点，则使用清理后的数据进行计算
                    if selected_method == "spearman":
                        corr, p_value = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
                    elif selected_method == "pearson":
                        corr, p_value = stats.pearsonr(combined.iloc[:, 0], combined.iloc[:, 1])
                    else:
                        corr, p_value = stats.kendalltau(combined.iloc[:, 0], combined.iloc[:, 1])
                # 将计算出的相关系数值存入矩阵
                correlation_matrix[i, j] = corr
                # 将计算出的p值存入矩阵
                p_value_matrix[i, j] = p_value
        # 使用当前分组的特征和目标名称创建DataFrame
        df_corr = pd.DataFrame(correlation_matrix, index=current_features, columns=current_targets)
        df_sig = pd.DataFrame(
            p_value_matrix < 0.05, index=current_features, columns=current_targets
        )
        # 打印相关性系数矩阵
        print("\n相关性系数:")
        print(df_corr.to_string())

        # 打印显著性矩阵
        print("\n显著性:")
        print(df_sig.to_string())
        # 保存相关性和显著性结果到总的字典里
        all_correlation_data[group] = {"correlation_df": df_corr, "p_value_df": df_sig}

    # 调用绘图函数
    create_full_ring_plot(
        all_data=all_correlation_data,  # 相关性
        all_feature_names=all_features_dict,  # 包含所有特征名称的字典
        all_target_names=generic_targets_dict,  # 目标
        color_palette=select_color,  # 配色
        sector_params=sector_params,  # 扇区几何参数
    )


# =========================================================================================
# ======================================执行分析绘图=========================================
# =========================================================================================
if __name__ == "__main__":
    main()

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

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
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
color_palettes = {
    1: {
        "name": "Original (Beige)",
        "bg_color": "#F5F5EC",
        "header_color": "#FADBAA",
        "header_text_color": "#5C3A1A",
        "bar_label_color": "#333333",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "custom_green", ["#C1D7A8", "#89A84B"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "custom_orange", ["#FADBAA", "#F5A85A"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("custom_pink", ["#F5D6E6", "#EC9FCB"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "custom_green_orange", ["#66B04C", "#E8843C"]
        ),
        "legend_equity": "#FADBAA",
        "legend_quality": "#C1D7A8",
        "legend_regulatory": "#F5D6E6",
    },
    2: {
        "name": "Dark Mode",
        "bg_color": "#2E2E2E",
        "header_color": "#00A0A0",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#F0F0F0",
        "group_green_cmap": LinearSegmentedColormap.from_list("dark_green", ["#508D69", "#9ADE7B"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "dark_orange", ["#B55B00", "#FF8F00"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("dark_purple", ["#713ABE", "#BEADFA"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("dark_beeswarm", ["#007BFF", "#FFD700"]),
        "legend_equity": "#FF8F00",
        "legend_quality": "#9ADE7B",
        "legend_regulatory": "#BEADFA",
    },
    3: {
        "name": "Corporate Blue",
        "bg_color": "#FFFFFF",
        "header_color": "#4A90E2",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#1A1A1A",
        "group_green_cmap": LinearSegmentedColormap.from_list("corp_green", ["#AEE1E1", "#008080"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "corp_orange", ["#F3DABC", "#E88E3C"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("corp_gray", ["#D0D0D0", "#606060"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("corp_beeswarm", ["#4A90E2", "#D0021B"]),
        "legend_equity": "#E88E3C",
        "legend_quality": "#008080",
        "legend_regulatory": "#606060",
    },
    4: {
        "name": "Slate & Steel",
        "bg_color": "#F4F6F6",
        "header_color": "#5D6D7E",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#17202A",
        "group_green_cmap": LinearSegmentedColormap.from_list("steel_blue", ["#A9CCE3", "#2471A3"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list("amber", ["#FAD7A0", "#E67E22"]),
        "group_pink_cmap": LinearSegmentedColormap.from_list("silver", ["#D7DBDD", "#839192"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("blue_red", ["#3498DB", "#E74C3C"]),
        "legend_equity": "#E67E22",
        "legend_quality": "#2471A3",
        "legend_regulatory": "#839192",
    },
    5: {
        "name": "Forest Floor",
        "bg_color": "#FDFEFE",
        "header_color": "#4A7C59",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#212F3D",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "forest_green", ["#ABEBC6", "#1E8449"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "burnt_orange", ["#FDEBD0", "#D35400"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("thistle", ["#EBDEF0", "#884EA0"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("blue_yellow", ["#5DADE2", "#F1C40F"]),
        "legend_equity": "#D35400",
        "legend_quality": "#1E8449",
        "legend_regulatory": "#884EA0",
    },
    6: {
        "name": "Midnight",
        "bg_color": "#1B2631",
        "header_color": "#8E44AD",
        "header_text_color": "#FDFEFE",
        "bar_label_color": "#EAECEE",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "bright_green", ["#7DCEA0", "#2ECC71"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "bright_orange", ["#F8C471", "#F39C12"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("bright_blue", ["#AED6F1", "#5DADE2"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("purple_gold", ["#AF7AC5", "#FAD7A0"]),
        "legend_equity": "#F39C12",
        "legend_quality": "#2ECC71",
        "legend_regulatory": "#5DADE2",
    },
    7: {
        "name": "Sunset",
        "bg_color": "#FEF9E7",
        "header_color": "#E74C3C",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#34495E",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "sunset_orange", ["#FAD7A0", "#F39C12"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "sunset_red", ["#F1948A", "#C0392B"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list(
            "sunset_purple", ["#D7BDE2", "#8E44AD"]
        ),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("purple_orange", ["#8E44AD", "#F39C12"]),
        "legend_equity": "#C0392B",
        "legend_quality": "#F39C12",
        "legend_regulatory": "#8E44AD",
    },
    8: {
        "name": "Ocean Deep",
        "bg_color": "#FFFFFF",
        "header_color": "#1A5276",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#1B2631",
        "group_green_cmap": LinearSegmentedColormap.from_list("teal", ["#A3E4D7", "#16A085"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list("sand", ["#FAD7A0", "#F39C12"]),
        "group_pink_cmap": LinearSegmentedColormap.from_list("sky_blue", ["#D4E6F1", "#3498DB"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("teal_coral", ["#16A085", "#E67E22"]),
        "legend_equity": "#F39C12",
        "legend_quality": "#16A085",
        "legend_regulatory": "#3498DB",
    },
    9: {
        "name": "Graphite & Gold",
        "bg_color": "#34495E",
        "header_color": "#F1C40F",
        "header_text_color": "#34495E",
        "bar_label_color": "#FDFEFE",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "graphite_green", ["#A9DFBF", "#27AE60"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list("gold", ["#FAD7A0", "#F39C12"]),
        "group_pink_cmap": LinearSegmentedColormap.from_list("silver", ["#D5D8DC", "#B2BABB"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("white_gold", ["#ECF0F1", "#F1C40F"]),
        "legend_equity": "#F39C12",
        "legend_quality": "#27AE60",
        "legend_regulatory": "#B2BABB",
    },
    10: {
        "name": "Mint & Chocolate",
        "bg_color": "#F0FBF4",
        "header_color": "#7D6658",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#4A403A",
        "group_green_cmap": LinearSegmentedColormap.from_list("mint", ["#A2D9CE", "#1ABC9C"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list("caramel", ["#FAD7A0", "#E67E22"]),
        "group_pink_cmap": LinearSegmentedColormap.from_list("strawberry", ["#F5B7B1", "#E74C3C"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("brown_mint", ["#7D6658", "#1ABC9C"]),
        "legend_equity": "#E67E22",
        "legend_quality": "#1ABC9C",
        "legend_regulatory": "#E74C3C",
    },
    11: {
        "name": "Academic",
        "bg_color": "#FFFFFF",
        "header_color": "#A93226",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#000000",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "strong_blue", ["#A9CCE3", "#2980B9"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "strong_orange", ["#FAD7A0", "#F39C12"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("strong_gray", ["#CCD1D1", "#7F8C8D"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("blue_red_acad", ["#2980B9", "#A93226"]),
        "legend_equity": "#F39C12",
        "legend_quality": "#2980B9",
        "legend_regulatory": "#7F8C8D",
    },
    12: {
        "name": "Lavender",
        "bg_color": "#FBFAFC",
        "header_color": "#8E44AD",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#3B1E53",
        "group_green_cmap": LinearSegmentedColormap.from_list("lav_green", ["#A2D9CE", "#1ABC9C"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "lav_yellow", ["#F9E79F", "#F1C40F"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("lav_purple", ["#D7BDE2", "#8E44AD"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("lav_beeswarm", ["#3498DB", "#E74C3C"]),
        "legend_equity": "#F1C40F",
        "legend_quality": "#1ABC9C",
        "legend_regulatory": "#8E44AD",
    },
    13: {
        "name": "Sepia Tone",
        "bg_color": "#FDF8F3",
        "header_color": "#7B5E4A",
        "header_text_color": "#FDF8F3",
        "bar_label_color": "#4E3629",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "sepia_green", ["#DDCBBE", "#A38C7A"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "sepia_orange", ["#FAD7A0", "#E67E22"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("sepia_pink", ["#D5C6B1", "#8C7853"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "sepia_beeswarm", ["#7B5E4A", "#E67E22"]
        ),
        "legend_equity": "#E67E22",
        "legend_quality": "#A38C7A",
        "legend_regulatory": "#8C7853",
    },
    14: {
        "name": "Cyberpunk",
        "bg_color": "#0A0A0A",
        "header_color": "#F000FF",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#F0F0F0",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "cyber_green", ["#00F0FF", "#007FFF"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "cyber_orange", ["#F0FF00", "#F0A000"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("cyber_pink", ["#FF00BF", "#B00080"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "cyber_beeswarm", ["#00F0FF", "#F000FF"]
        ),
        "legend_equity": "#F0A000",
        "legend_quality": "#007FFF",
        "legend_regulatory": "#B00080",
    },
    15: {
        "name": "Minimalist Gray",
        "bg_color": "#FFFFFF",
        "header_color": "#34495E",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#000000",
        "group_green_cmap": LinearSegmentedColormap.from_list("gray_green", ["#D5D8DC", "#7F8C8D"]),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "gray_orange", ["#B2BABB", "#566573"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("gray_pink", ["#F2F3F4", "#BDC3C7"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list("gray_beeswarm", ["#34495E", "#E74C3C"]),
        "legend_equity": "#566573",
        "legend_quality": "#7F8C8D",
        "legend_regulatory": "#BDC3C7",
    },
    16: {
        "name": "Spring",
        "bg_color": "#FDFEFE",
        "header_color": "#ABEBC6",
        "header_text_color": "#145A32",
        "bar_label_color": "#34495E",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "spring_green", ["#A9DFBF", "#27AE60"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "spring_orange", ["#F9E79F", "#F1C40F"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("spring_pink", ["#F5B7B1", "#E74C3C"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "spring_beeswarm", ["#5DADE2", "#F1C40F"]
        ),
        "legend_equity": "#F1C40F",
        "legend_quality": "#27AE60",
        "legend_regulatory": "#E74C3C",
    },
    17: {
        "name": "Watermelon",
        "bg_color": "#FFFFFF",
        "header_color": "#E74C3C",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#000000",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "melon_green", ["#A9DFBF", "#27AE60"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "melon_orange", ["#FADBD8", "#F1948A"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("melon_pink", ["#D5D8DC", "#7F8C8D"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "melon_beeswarm", ["#27AE60", "#E74C3C"]
        ),
        "legend_equity": "#F1948A",
        "legend_quality": "#27AE60",
        "legend_regulatory": "#7F8C8D",
    },
    18: {
        "name": "Autumn",
        "bg_color": "#FEF5E7",
        "header_color": "#A04000",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#4A2304",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "autumn_green", ["#FAD7A0", "#F39C12"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "autumn_orange", ["#EDBB99", "#D35400"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("autumn_pink", ["#F5B7B1", "#C0392B"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "autumn_beeswarm", ["#2980B9", "#C0392B"]
        ),
        "legend_equity": "#D35400",
        "legend_quality": "#F39C12",
        "legend_regulatory": "#C0392B",
    },
    19: {
        "name": "High-Contrast (Accessible)",
        "bg_color": "#FFFFFF",
        "header_color": "#000000",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#000000",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "access_green", ["#A9CCE3", "#1F618D"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "access_orange", ["#FAD7A0", "#E67E22"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list("access_pink", ["#D7DBDD", "#808B96"]),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "access_beeswarm", ["#000000", "#E67E22"]
        ),
        "legend_equity": "#E67E22",
        "legend_quality": "#1F618D",
        "legend_regulatory": "#808B96",
    },
    20: {
        "name": "Primary Colors",
        "bg_color": "#FFFFFF",
        "header_color": "#4285F4",
        "header_text_color": "#FFFFFF",
        "bar_label_color": "#1A1A1A",
        "group_green_cmap": LinearSegmentedColormap.from_list(
            "primary_green", ["#B7E1CD", "#34A853"]
        ),
        "group_orange_cmap": LinearSegmentedColormap.from_list(
            "primary_orange", ["#FCE8B2", "#FBBC05"]
        ),
        "group_pink_cmap": LinearSegmentedColormap.from_list(
            "primary_pink", ["#F4B4B4", "#EA4335"]
        ),
        "beeswarm_cmap": LinearSegmentedColormap.from_list(
            "primary_beeswarm", ["#4285F4", "#EA4335"]
        ),
        "legend_equity": "#FBBC05",
        "legend_quality": "#34A853",
        "legend_regulatory": "#EA4335",
    },
}
selected_palette = 1  # 配色方案选择

color = color_palettes[selected_palette]  # 提取方案


# =========================================================================================
# =====================================3.绘图函数=========================================
# =========================================================================================
def plot_shap_summary(
    shap_values: shap.Explanation,  # SHAP解释对象
    feature_names: list,  # 特征的名称
    group_green: set,  # 分组特征名称
    group_orange: set,  # 分组特征名称
    theme: dict,  # 方案
    output_filename_base: str = "shap_summary_plot",  # 文件名
):
    shap_vals_arr = shap_values.values  # 获取原始SHAP值
    mean_abs_shap = np.abs(shap_vals_arr).mean(axis=0)  # 每个特征的平均绝对SHAP值
    num_features = len(feature_names)  # 特征的总数
    bg_color = theme["bg_color"]  # 背景颜色
    # 创建一个图形和两个子图
    fig, (ax_bar, ax_swarm) = plt.subplots(
        1,
        2,
        figsize=(13, 9),
        gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0},  # 宽度比例和它们之间的水平间距
        sharey=True,  # 共享Y轴
    )
    fig.patch.set_facecolor("white")  # 最外层的背景色为白色
    ax_bar.set_facecolor(bg_color)  # 左侧条形图的背景色
    ax_swarm.set_facecolor(bg_color)  # 右侧蜂群图的背景色

    header_height_ratio = 0.08  # 头部矩形占图形总高度的比例
    header_color = theme["header_color"]  # 头部矩形的颜色
    header_text_color = theme["header_text_color"]  # 头部标题的文字颜色
    header_text = "Activity Quality"  # 头部标题的文本内容

    plot_left_margin = 0.07  # 整个绘图区域距离图形左边缘的距离
    plot_right_margin = 0.98  # 整个绘图区域距离图形右边缘的距离
    plot_width = plot_right_margin - plot_left_margin  # 绘图区域的宽

    # 头部矩形
    rect = patches.Rectangle(
        (plot_left_margin, 1 - header_height_ratio),  # 矩形的左下角坐标
        plot_width,  # 宽度
        header_height_ratio,  # 高度
        facecolor=header_color,  # 填充颜色
        edgecolor="none",  # 边缘颜色
        transform=fig.transFigure,  # 指定坐标系为图形坐标系
        zorder=3,  # 设置绘制顺序
    )
    fig.add_artist(rect)  # 添加到图形上

    # 添加标题文本
    fig.text(
        (plot_left_margin + plot_right_margin) / 2,  # X坐标
        1 - header_height_ratio / 2,  # Y坐标
        header_text,  # 要显示的文本
        fontsize=24,  # 体大小
        fontweight="bold",  # 字体粗细
        color=header_text_color,  # 字体颜色
        ha="center",  # 水平对齐方式
        va="center",  # 垂直对齐方式
        transform=fig.transFigure,  # 图形坐标系
        zorder=4,  # 绘制顺序
        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],  # 添加描边效果
    )

    # 调整子图在图形中的位置
    fig.subplots_adjust(
        top=1 - header_height_ratio,  # 子图的顶部边缘
        left=plot_left_margin,  # 子图的左侧边缘
        right=plot_right_margin,  # 图的右侧边缘
        bottom=0.2,  # 子图的底部边缘
    )

    # 不同分组特征的颜色渐变
    green_cmap = theme["group_green_cmap"]
    pink_cmap = theme["group_pink_cmap"]
    orange_cmap = theme["group_orange_cmap"]

    # 创建一个从 0.3 到 1 的线性间隔数组，用于给条形图上色 (避免颜色太浅)
    color_values_scaled = np.linspace(0.3, 1, num_features)

    # 获取按 mean_abs_shap 降序排列的原始特征索引
    sorted_idx_for_color = np.argsort(mean_abs_shap)[::-1]
    sorted_colors_temp = []  # 初始化一个临时列表，用于存放排序后的颜色

    # 遍历排序后的特征索引
    for i, original_feature_idx in enumerate(sorted_idx_for_color):
        feature = feature_names[original_feature_idx]  # 根据原始序号获取特征名称
        color_val = color_values_scaled[i]  # 获取当前排序位置对应的颜色深浅值
        # 检查特征对应的分组，进行颜色赋予
        if feature in group_green:
            sorted_colors_temp.append(green_cmap(color_val))
        elif feature in group_orange:
            sorted_colors_temp.append(orange_cmap(color_val))
        else:
            sorted_colors_temp.append(pink_cmap(color_val))

    # 获取按重要性排序的特征名称列表
    sorted_features_for_color = [feature_names[i] for i in sorted_idx_for_color]
    # 创建一个特征名称，对应颜色的字典
    feature_color_map = {
        feature_name: color
        for feature_name, color in zip(sorted_features_for_color, sorted_colors_temp)
    }
    # 创建一个特征名称，平均绝对SHAP值的字典
    feature_shap_map = dict(zip(feature_names, mean_abs_shap))

    custom_cmap = theme["beeswarm_cmap"]  # 获取蜂群图的颜色映射

    # 绘制蜂巢图
    shap.plots.beeswarm(
        shap_values,  # shap值
        max_display=num_features,  # 显示所有特征
        ax=ax_swarm,  # 指定在右侧子图上绘制
        show=False,  # 不立即显示图形
        color_bar=False,  # 不自动显示颜色条
        plot_size=None,  # 自动调整大小
        color=custom_cmap,  # 指定点的颜色映射
    )
    ax_swarm.set_ylabel("")  # 移除蜂群图 Y轴标题
    ax_swarm.set_xlabel(
        "SHAP value (impact on model output)", fontsize=24, fontname="Times New Roman"
    )  # x轴标题
    ax_swarm.spines[["right", "top", "left"]].set_visible(False)  # 隐藏右、上、左的坐标轴框线

    swarm_feature_order = sorted_features_for_color
    swarm_feature_order = swarm_feature_order[::-1]  # 反转列表
    y_pos = np.arange(len(swarm_feature_order))  # 创建Y轴的位置
    bar_height = 0.7  # 设置每个条形的高度
    rgb_bg = to_rgb(bg_color)  # 将背景颜色转换为RGB元组
    bar_label_color = theme["bar_label_color"]  # 从主题中获取条形图标签的颜色

    # 遍历排好序的特征
    for i, feature_name in enumerate(swarm_feature_order):
        y = y_pos[i]  # 获取当前特征的Y坐标
        width = feature_shap_map.get(feature_name, 0)  # 该特征的平均SHAP值
        solid_color = feature_color_map.get(feature_name)  # 查找该特征的颜色
        rgb_solid_color = to_rgb(solid_color)  # 将条形颜色转换为RGB元组

        if width > 0:  # 大于0时绘制
            # 为每个条形创建一个从背景色到其特征色的线性渐变
            bar_cmap = LinearSegmentedColormap.from_list(
                f"bar_cmap_{i}",  # 名称
                [rgb_bg, rgb_solid_color],  # 颜色列表[起始色, 结束色]
            )
            # 创建一个的数组，表示这个渐变
            gradient_img = bar_cmap(np.linspace(0, 1, 256).reshape(1, -1))

            # 绘制这个渐变图像
            ax_bar.imshow(
                gradient_img,  # 渐变数据
                aspect="auto",  # 自动调整宽高比
                extent=[
                    0,
                    width,
                    y - bar_height / 2,
                    y + bar_height / 2,
                ],  # [x_min, x_max, y_min, y_max]
                zorder=2,
            )
            text_x = width + 0.02  # 设置条形图右侧数值坐标
            # 添加数值
            ax_bar.text(
                text_x,  # X坐标
                y,  # Y坐标
                f"+{width:.2f}",  # 文本
                va="center",  # 垂直居中
                ha="left",  # 水平左对齐
                fontsize=20,  # 字体大小
                color=bar_label_color,  # 字体颜色
            )

    ax_bar.tick_params(axis="y", length=0)  # 隐藏Y轴的刻度线
    ax_bar.set_xlim(0, mean_abs_shap.max() * 1.15)  # 设置X轴的范围
    ax_bar.spines[["right", "top", "left"]].set_visible(False)  # 隐藏条形图的右、上、左框线
    ax_bar.set_xlabel("Mean(|SHAP value|)", fontsize=24)  # 条形图x轴标题
    # 设置Y轴刻度标签（特征名）的大小
    ax_bar.tick_params(axis="y", labelsize=20)
    for label in ax_swarm.get_yticklabels():
        label.set_fontweight("bold")
    # 设置两个子图的 X 轴刻度标签
    ax_bar.tick_params(axis="x", labelsize=20)
    ax_swarm.tick_params(axis="x", labelsize=20)
    for label in ax_bar.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax_swarm.get_xticklabels():
        label.set_fontweight("bold")
    # 在图形的指定位置手动添加一个新的轴用于放置颜色条
    cbar_ax = fig.add_axes(
        [
            plot_left_margin,  # 左下角的X坐标
            0.08,  # 左下角Y坐标
            plot_width,  # 宽度
            0.02,  # 高度
        ]
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)  # 创建一个归一化对象
    # 绘制一个颜色条
    cb = mcolorbar.ColorbarBase(cbar_ax, cmap=custom_cmap, norm=norm, orientation="horizontal")
    cb.outline.set_visible(False)  # 隐藏颜色条的轮廓
    cb.set_ticks([0, 1])  # 设置刻度
    cb.set_ticklabels(["Low", "High"])  # 刻度标签
    cb.ax.tick_params(labelsize=24, length=0)  # 隐藏刻度线

    # 颜色条标题
    fig.text(
        0.5,  # X坐标
        0.05,  # Y坐标
        "Feature Value",  # 内容
        ha="center",  # 左对齐
        va="top",  # 顶部对齐
        fontsize=24,  # 字体大小
    )

    # 获取不同分组的图例颜色
    color_equity = theme["legend_equity"]
    color_quality = theme["legend_quality"]
    color_regulatory = theme["legend_regulatory"]

    # 创建用于图例的 色块
    legend_patches = [
        patches.Patch(color=color_equity, label="Equity Indicators"),
        patches.Patch(color=color_quality, label="Quality Indicators"),
        patches.Patch(color=color_regulatory, label="Regulatory Indicators"),
    ]

    # 绘制图例
    fig.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(plot_left_margin, -0.05),
        ncol=3,
        frameon=False,
        fontsize=24,
    )
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"{output_filename_base}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.savefig(
        str(OUTPUT_DIR / f"{output_filename_base}.pdf"),
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)  # 关闭


if __name__ == "__main__":
    # =========================================================================================
    # =====================================4.数据提取与预处理=========================================
    # =========================================================================================
    # 读取数据
    excel_filename = str(DATA_DIR / "data.xlsx")
    df = pd.read_excel(excel_filename)
    y = df["target"]
    X = df.drop("target", axis=1)
    # 特征名称列表
    feature_names = list(X.columns)
    # 训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("数据加载和划分成功。")
    print(f"训练集: {X_train.shape[0]} 条样本, {X_train.shape[1]} 个特征")
    print(f"验证集: {X_test.shape[0]} 条样本, {X_test.shape[1]} 个特征")
    # 定义分组
    group_green = set(["TE", "RAP", "COHESION", "SWNDVIM", "GSAR", "PCGSA"])
    group_orange = set(["RA", "LSI", "PAP", "SLSTM"])
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转回DataFrame，加上特征名称
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    # =========================================================================================
    # =====================================5.模型构建=========================================
    # =========================================================================================
    # 定义要搜索的超参数网格
    param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10], "min_samples_leaf": [1, 5]}
    # 实例化一个随机森林回归器
    rf = RandomForestRegressor(random_state=42)
    # 例化网格搜索对象
    grid_search = GridSearchCV(
        estimator=rf,  # 要优化的模型
        param_grid=param_grid,  # 要搜索的参数网格
        cv=3,  # 3 折交叉验证
        n_jobs=-1,  # 使用所有可用的 CPU 核心
        scoring="neg_mean_squared_error",  # 评估指标 (负均方误差)
        verbose=1,  # 打印搜索过程
    )
    # 执行网格搜索
    grid_search.fit(X_train_scaled_df, y_train)

    # 获取交叉验证中表现最好的模型
    model = grid_search.best_estimator_
    print(f"网格搜索完成。最佳参数: {grid_search.best_params_}")
    print(f"最佳模型在验证集上的 R^2 分数: {model.score(X_test_scaled_df, y_test):.4f}")
    # =========================================================================================
    # =====================================6.分析=========================================
    # =========================================================================================
    # 创建一个TreeExplainer
    explainer = shap.TreeExplainer(model)
    # 在缩放后的测试集上计算SHA 值
    shap_values = explainer(X_test_scaled_df)
    print("SHAP 值计算完成。")
    # 计算每个特征的平均绝对 SHAP 值
    mean_abs_shap_for_print = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,  # 特征
            "Importance (Mean |SHAP|)": mean_abs_shap_for_print,  # 重要性
        }
    ).sort_values(by="Importance (Mean |SHAP|)", ascending=False)  # 按重要性降序排序

    print("\n--- SHAP 特征重要性排序 ---")
    print(importance_df.to_string())
    # =========================================================================================
    # =====================================7.绘图=========================================
    # =========================================================================================
    # 调用绘图函数
    plot_shap_summary(
        shap_values=shap_values,  # 传入计算好的 SHAP 值对象
        feature_names=feature_names,  # 传入特征名称列表
        group_green=group_green,  # 传入绿色组的定义
        group_orange=group_orange,  # 传入橙色组的定义
        theme=color,  # 颜色方案
        # 文件名
        output_filename_base=f"shap_{selected_palette}",
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

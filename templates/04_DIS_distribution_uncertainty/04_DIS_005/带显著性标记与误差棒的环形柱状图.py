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

# -*- coding: utf-8 -*- # 指定文件编码为 UTF-8，确保可以处理中文字符
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# --- 设置 Matplotlib 支持中文显示 ---
matplotlib.use("Agg")  # (可选) 明确指定 Matplotlib 后端，有时用于解决特定环境下的兼容性问题
plt.rcParams["font.sans-serif"] = [
    "Times New Roman"
]  # 设置默认的无衬线字体为 'SimHei' (黑体)，用于正确显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示负号，Matplotlib 默认可能将负号显示为方块
# --- 图表整体配置参数 ---
n_segments = 10  # 图表中的扇区数量 (通常对应指标或分组的数量)
outer_radius = 1.0  # 环形图的 *最* 外圈半径 (决定背景扇形的最外边界)
inner_radius = 0.3  # 环形图的内圈半径 (中心空白区域的边界)
radial_padding_top = 0.05  # 在最外圈半径和绘图元素（如条形图）最大高度之间保留的径向空白距离
plotting_outer_radius = (
    outer_radius - radial_padding_top
)  # 实际用于绘制条形图、误差线、内部标注、虚线的最大半径
gap_degree = 8  # 每个扇区之间的间隔角度 (单位：度)
side_padding_degree = 0.5  # 每个扇区内部，两侧边缘与条形图组之间的留白角度 (单位：度)
start_angle_offset = (
    90  # 整个图表的起始角度偏移量 (单位：度)，90 度表示第一个扇区从正上方开始，按顺时针排列
)
edge_color = "black"  # 扇区、条形图等元素的边框颜色
segment_line_width = 1  # 背景扇区的边框线宽
bar_line_width = 0.5  # 内部条形图的边框线宽
error_bar_color = "black"  # 误差线的颜色
error_bar_linewidth = 0.8  # 误差线的线宽
error_cap_size_deg = 1.5  # 误差线顶端 "帽子" 的宽度 (以角度表示其跨度)
label_fontsize = 10  # 指标类别标签 (如 "SOC", "MAOC") 的字体大小
legend_fontsize = 12  # 图例的字体大小
background_segment_alpha = 0.5  # 背景扇区的透明度 (0=完全透明, 1=完全不透明)
# --- 外部彩色弧形参数 ---
show_single_outer_arc = True  # 是否显示外部的、与背景扇区同色的弧形段
outer_arc_radial_gap = 0.08  # 外部弧形的内边缘 与 背景扇区外边缘 (outer_radius) 之间的径向间隙
outer_arc_thickness = 0.04  # 外部弧形的厚度 (径向宽度)
outer_arc_edge_color = "black"  # 外部弧形的边框颜色
outer_arc_linewidth = 0.5  # 外部弧形的边框线宽
outer_arc_zorder = 1.5  # 外部弧形的绘制层级 (zorder 值越大越靠上)
# --- 指标类别标签 ("SOC", "MAOC"...) 位置参数 ---
category_label_radial_pos_in_gap = 0.5  # 类别标签在 (outer_radius) 和 (外部弧形内侧) 之间 gap 中的径向位置 (0=贴近内侧, 0.5=中间, 1=贴近外侧)
# --- 计算指标类别标签的实际绘制半径 ---
if show_single_outer_arc:  # 如果显示外部弧形
    category_label_radius = (
        outer_radius + outer_arc_radial_gap * category_label_radial_pos_in_gap
    )  # 半径在 gap 内
else:  # 如果不显示外部弧形
    category_label_radius = outer_radius * 1.05  # 将标签放在背景扇区外侧稍远的位置
# --- 数值标注参数 (对应内部虚线的 1/4, 1/2, 3/4 值) ---
show_annotations = True  # 是否显示这些数值标注文本
annotation_fontsize = 10  # 数值标注文本的字体大小
annotation_color = "black"  # 数值标注文本的颜色
annotation_outside_offset_deg = (
    -2.7
)  # 数值标注文本相对于扇区起始角度的角度偏移量 (负值表示顺时针偏移出扇区)
# --- 内部参考虚线参数 ---
show_annotation_lines = True  # 是否绘制内部的虚线圆弧
annotation_line_color = "#555555"  # 内部虚线的颜色
annotation_line_width = 0.6  # 内部虚线的线宽
annotation_line_style = "--"  # 内部虚线的样式 (例如 '--', ':', '-.')
annotation_line_points = 20  # 用于绘制每段虚线圆弧的点的数量 (越多越平滑)
# --- 显著性标注参数 (字母 'a', 'b', 'c' 等) ---
show_significance = True  # 是否显示显著性字母标记
sig_label_radial_offset = 0.02  # 显著性字母距离其对应误差棒顶端的径向偏移量
sig_fontsize = 10  # 显著性字母的字体大小
sig_color = "black"  # 显著性字母的颜色
# sig_rotation = 0                         # <-- 移除: 旋转角度将动态计算，不再需要固定值
# --- 背景扇区颜色列表 ---
# 为每个扇区指定颜色，如果扇区数多于颜色数，将循环使用
your_manual_colors = [
    "#CCCCFF",
    "#CCFFCC",
    "#FFFFCC",
    "#99FFFF",
    "#99CCFF",
    "#FFCCFF",
    "#66FFCC",
    "#CCFFCC",
    "#FF9966",
    "#FFCC33",
]
# --- 条形图特定参数 ---
num_bars_per_group = 4  # 每个扇区内的条形图数量 (对应不同的处理或分组)
treatment_colors = ["#00CC99", "#FF3300", "#CCFFFF", "#66ff66"]  # 每个处理/分组对应的条形图颜色
treatment_labels = ["CK", "Cd", "GSH", "GSH+Cd"]  # 每个处理/分组的标签 (用于图例)
intra_bar_gap_factor = 0.1  # 条形图之间的间隙大小，是单个条形图角度宽度的倍数
# --- 示例数据 ---
# 数据结构: 每个元组代表一个扇区 (指标)
# (指标标签, 均值列表, 误差列表, 显著性字母列表, 最小值, 最大值)
# !!! 重要: 请将 'a', 'b', 'c' 等替换为你的实际显著性分析结果 !!!
plot_data = [
    ("qP", [10, 15, 12, 18], [1, 1.5, 1.2, 2], ["a", "b", "a", "c"], 0, 25),  # 第1个指标数据
    ("qN", [25, 20, 22, 15], [2, 2.5, 2.2, 1.8], ["a", "a", "a", "b"], 10, 30),  # 第2个指标数据
    ("LDW", [8, 5, 7, 10], [0.8, 0.5, 0.7, 1.2], ["b", "a", "a", "c"], 0, 15),  # 第3个指标数据
    ("LFW", [12, 18, 15, 20], [1.1, 1.9, 1.6, 2.1], ["a", "c", "b", "d"], 5, 25),  # 第4个指标数据
    ("CHL", [30, 25, 28, 22], [3, 2.8, 2.9, 2.4], ["c", "a", "b", "a"], 15, 35),  # 第5个指标数据
    ("CHI", [14, 10, 11, 16], [1.4, 1.0, 1.1, 1.7], ["b", "a", "a", "c"], 5, 20),  # 第6个指标数据
    ("CAT", [22, 28, 25, 30], [2.1, 3.0, 2.6, 3.2], ["a", "c", "b", "c"], 20, 35),  # 第7个指标数据
    ("POD", [16, 12, 14, 19], [1.5, 1.3, 1.4, 2.0], ["b", "a", "a", "c"], 10, 25),  # 第8个指标数据
    ("FV", [19, 24, 21, 26], [1.9, 2.5, 2.1, 2.7], ["a", "c", "b", "c"], 15, 30),  # 第9个指标数据
    ("Y", [28, 21, 26, 19], [2.7, 2.0, 2.5, 1.8], ["c", "a", "b", "a"], 15, 30),  # 第10个指标数据
]
# --- 数据结构和参数一致性检查 ---
if len(plot_data) != n_segments:  # 检查数据列表的长度是否与设定的扇区数量匹配
    exit(
        "错误：数据列表 plot_data 的长度与扇区数量 n_segments 不匹配"
    )  # 如果不匹配，打印错误信息并退出程序
# 检查 plot_data 中每个元素是否都有 6 个部分，并且第4部分（显著性字母列表）的长度是否等于每个组的条形图数
if not all(len(item) == 6 for item in plot_data):  # 检查元组长度
    exit(
        "错误：plot_data 中数据项格式不正确 (应为6个元素: Label, Means, Errors, SigLetters, Min, Max)"
    )
if not all(
    len(item[3]) == num_bars_per_group for item in plot_data if item[3] is not None
):  # 检查显著性列表长度
    # 找出哪些指标的显著性字母列表长度不正确
    mismatched = [
        item[0] for item in plot_data if item[3] is not None and len(item[3]) != num_bars_per_group
    ]
    if mismatched:  # 如果存在长度不匹配的项
        exit(
            f"错误：以下指标的显著性字母列表长度不等于 num_bars_per_group ({num_bars_per_group}): {', '.join(mismatched)}"
        )  # 打印错误信息并退出


# --- 辅助函数：将数值按比例缩放到半径 ---
# 这个函数将一个在 [segment_min, segment_max] 范围内的值，映射到环形的 [inner_r, plot_outer_r] 半径范围内
def scale_value_to_radius(value, segment_min, segment_max, inner_r, plot_outer_r):
    value_range = segment_max - segment_min  # 计算原始数值范围的宽度
    ring_radial_width_effective = plot_outer_r - inner_r  # 计算环形绘图区域的有效径向宽度
    if ring_radial_width_effective <= 0:
        return inner_r  # 如果有效宽度小于等于0，直接返回内半径
    if value_range <= 0:  # 如果原始数值范围宽度小于等于0 (例如 min=max)
        relative_pos = 0.5  # 将相对位置设为中间 (0.5)
    else:  # 正常情况
        relative_pos = (value - segment_min) / value_range  # 计算值在范围内的相对位置 (0到1之间)
        relative_pos = max(0.0, min(1.0, relative_pos))  # 将相对位置限制在 [0, 1] 区间内，防止超出
    # 根据相对位置计算目标半径
    target_radius = inner_r + relative_pos * ring_radial_width_effective
    # 再次确保计算出的半径在允许的 [inner_r, plot_outer_r] 范围内
    clipped_radius = max(inner_r, min(plot_outer_r, target_radius))
    return clipped_radius  # 返回最终计算并裁剪后的半径


# --- 计算角度相关的参数 ---
ring_radial_width = outer_radius - inner_radius  # 计算背景环的径向宽度
if ring_radial_width <= 0:
    exit("错误：外圈半径 outer_radius 必须大于内圈半径 inner_radius")  # 检查半径设置是否合理
total_gap_degrees = n_segments * gap_degree  # 计算所有扇区间隙占用的总角度
if total_gap_degrees >= 360:
    exit("错误：总间隔角度 gap_degree 过大，扇区没有空间了")  # 检查总间隙是否超过360度
total_segment_degrees = 360.0 - total_gap_degrees  # 计算所有扇区本身占用的总角度
segment_degree = total_segment_degrees / n_segments  # 计算每个扇区的角度宽度
# 计算每个扇区内用于绘制条形图的可用角度宽度
available_bar_angle = segment_degree - 2 * side_padding_degree
if available_bar_angle <= 0:
    exit(
        f"错误：扇区角度 ({segment_degree:.1f}度) 不足以容纳两侧留白 side_padding_degree ({2 * side_padding_degree:.1f}度)"
    )
# 计算绘制条形图和它们之间间隙所需的总“单位角度宽度”
# 每个条形图占1个单位，每个间隙占 intra_bar_gap_factor 个单位
total_bar_units = num_bars_per_group + (num_bars_per_group - 1) * intra_bar_gap_factor
if total_bar_units <= 0:
    total_bar_units = 1  # 防止除以零
# 计算每个“单位角度”实际对应的角度值
bar_angular_unit_width = available_bar_angle / total_bar_units
# 单个条形图的角度宽度
bar_angular_width = bar_angular_unit_width
# 条形图之间的间隙角度宽度
intra_bar_gap_angle = bar_angular_unit_width * intra_bar_gap_factor
# --- 计算每个扇区的中心角度、起始角度和结束角度 ---
segment_center_angles = []  # 存储每个扇区中心角度的列表
current_center_angle = start_angle_offset  # 第一个扇区的中心角度从偏移量开始
for _ in range(n_segments):  # 循环 n_segments 次
    segment_center_angles.append(current_center_angle)  # 将当前中心角度加入列表
    # 计算下一个扇区的中心角度 (当前角度 - 单个扇区宽度 - 单个间隙宽度)
    current_center_angle -= segment_degree + gap_degree
# 计算每个扇区的起始角度 (中心角度 - 扇区宽度的一半)
segment_start_angles = [center - segment_degree / 2.0 for center in segment_center_angles]
# 计算每个扇区的结束角度 (中心角度 + 扇区宽度的一半)
segment_end_angles = [center + segment_degree / 2.0 for center in segment_center_angles]
# --- Matplotlib 绘图设置 ---
fig, ax = plt.subplots(
    figsize=(10, 10)
)  # 创建一个图形(fig)和一个轴域(ax)，设置图形大小为 10x10 英寸
ax.set_aspect("equal", adjustable="box")  # 设置轴域的横纵坐标比例相等，确保圆形或环形不变形
# --- 计算画布的显示范围 (xlim, ylim) ---
if show_single_outer_arc:  # 如果显示外部弧形
    # 最外层元素的半径是外部弧形的外半径
    figure_outermost_radius = outer_radius + outer_arc_radial_gap + outer_arc_thickness
else:  # 如果不显示外部弧形
    # 最外层元素是类别标签
    figure_outermost_radius = category_label_radius
# 考虑显著性标签可能会稍微超出误差棒，增加一点边界
# 选择 0.03 和显著性标签偏移量中较大的一个作为额外填充
lim_padding = max(0.03, sig_label_radial_offset if show_significance else 0.03)
# 计算最终的坐标轴范围限制，基于最外层元素半径 + 填充 + 再加一点点(2%)的额外边距
lim = (figure_outermost_radius + lim_padding) * 1.02
ax.set_xlim(-lim, lim)  # 设置 x 轴的显示范围
ax.set_ylim(-lim, lim)  # 设置 y 轴的显示范围
ax.axis("off")  # 关闭坐标轴的刻度、标签和边框线，使背景干净
# --- 开始绘制图表的主要部分 ---
print(f"开始绘制图表，总共 {n_segments} 个指标扇区...")  # 打印提示信息
# --- 循环绘制每个扇区及其内部元素 ---
for i in range(n_segments):  # 遍历每个扇区的数据索引 i
    # --- 获取当前扇区的数据 ---
    # 从 plot_data 中解包当前扇区的数据: 指标标签, 均值列表, 误差列表, 显著性字母列表, 最小值, 最大值
    metric_label, means, errors, sig_letters, segment_min, segment_max = plot_data[i]
    # 获取当前扇区的起始角度和结束角度
    segment_start_angle = segment_start_angles[i]
    segment_end_angle = segment_end_angles[i]
    # 根据索引 i 获取当前扇区的背景颜色 (循环使用颜色列表)
    segment_bg_color = your_manual_colors[i % len(your_manual_colors)]
    # --- 1. 绘制背景扇形 ---
    # 使用 patches.Wedge 创建一个楔形 (扇形环)
    background_wedge = patches.Wedge(
        center=(0, 0),  # 圆心坐标
        r=outer_radius,  # 外半径
        theta1=segment_start_angle,  # 起始角度 (顺时针)
        theta2=segment_end_angle,  # 结束角度 (顺时针)
        width=ring_radial_width,  # 环的宽度 (outer_radius - inner_radius)
        facecolor=segment_bg_color,  # 填充颜色
        edgecolor=edge_color,  # 边框颜色
        linewidth=segment_line_width,  # 边框线宽
        alpha=background_segment_alpha,  # 透明度
        zorder=1,  # 绘制层级 (较低，在底层)
    )
    ax.add_patch(background_wedge)  # 将创建的背景扇形添加到轴域中
    # --- 1.5 绘制单个外部完整弧形 (如果启用) ---
    if show_single_outer_arc:  # 检查是否需要绘制外部弧形
        # 计算外部弧形的内半径和外半径
        arc_r_inner = outer_radius + outer_arc_radial_gap
        arc_r_outer = arc_r_inner + outer_arc_thickness
        # 创建外部弧形的 Wedge 对象
        outer_arc_wedge = patches.Wedge(
            center=(0, 0),  # 圆心
            r=arc_r_outer,  # 外半径 (弧形的最外边界)
            theta1=segment_start_angle,  # 起始角度
            theta2=segment_end_angle,  # 结束角度
            width=outer_arc_thickness,  # 弧形的厚度
            facecolor=segment_bg_color,  # 填充颜色 (与背景扇区一致)
            alpha=1.0,  # 透明度 (设为不透明)
            edgecolor=outer_arc_edge_color,  # 边框颜色
            linewidth=outer_arc_linewidth,  # 边框线宽
            zorder=outer_arc_zorder,  # 绘制层级 (比背景高一点)
        )
        ax.add_patch(outer_arc_wedge)  # 添加到轴域
    # --- 2. 绘制扇区内的条形图、误差棒和显著性字母 ---
    # 初始化当前条形图的起始角度 (扇区起始角度 + 左侧留白)
    current_bar_start_angle = segment_start_angle + side_padding_degree
    # 循环绘制扇区内的每个条形图 (根据 num_bars_per_group)
    for j in range(num_bars_per_group):
        # 获取当前条形图的均值、误差值和颜色
        mean_val = means[j]
        error_val = errors[j]
        bar_color = treatment_colors[j % len(treatment_colors)]
        # --- 计算半径 ---
        # 使用 scale_value_to_radius 函数计算条形图顶部的半径
        bar_top_radius = scale_value_to_radius(
            mean_val, segment_min, segment_max, inner_radius, plotting_outer_radius
        )
        # 计算误差棒下端的半径
        lower_radius = scale_value_to_radius(
            mean_val - error_val, segment_min, segment_max, inner_radius, plotting_outer_radius
        )
        # 计算误差棒上端的半径
        upper_radius = scale_value_to_radius(
            mean_val + error_val, segment_min, segment_max, inner_radius, plotting_outer_radius
        )
        # 计算条形图的径向厚度 (条形图顶部半径 - 内圈半径)，确保不为负
        bar_radial_thickness = max(0, bar_top_radius - inner_radius)
        # --- 计算角度 ---
        # 当前条形图的起始角度
        bar_theta1 = current_bar_start_angle
        # 当前条形图的结束角度
        bar_theta2 = current_bar_start_angle + bar_angular_width
        # 当前条形图的中心角度 (用于放置误差线和显著性标记)
        bar_center_angle_deg = bar_theta1 + bar_angular_width / 2.0
        # 将中心角度转换为弧度，用于三角函数计算坐标
        center_angle_rad = np.radians(bar_center_angle_deg)
        # --- 绘制条形图 (Wedge) ---
        if bar_radial_thickness > 1e-6:  # 仅当厚度大于一个很小的值时才绘制，避免绘制宽度为0的图形
            bar_wedge = patches.Wedge(
                center=(0, 0),  # 圆心
                r=bar_top_radius,  # 外半径 (条形图顶部)
                theta1=bar_theta1,  # 起始角度
                theta2=bar_theta2,  # 结束角度
                width=bar_radial_thickness,  # 条形图的径向厚度
                facecolor=bar_color,  # 填充颜色
                edgecolor=edge_color,  # 边框颜色
                linewidth=bar_line_width,  # 边框线宽
                zorder=2,  # 绘制层级 (在背景之上)
            )
            ax.add_patch(bar_wedge)  # 添加到轴域
        # --- 绘制误差棒 ---
        if upper_radius > lower_radius + 1e-6:  # 仅当误差棒有实际长度时绘制
            # 计算误差棒下端和上端的 x, y 坐标
            x_lower = lower_radius * np.cos(center_angle_rad)
            y_lower = lower_radius * np.sin(center_angle_rad)
            x_upper = upper_radius * np.cos(center_angle_rad)
            y_upper = upper_radius * np.sin(center_angle_rad)
            # 绘制误差棒的主体竖线
            ax.plot(
                [x_lower, x_upper],
                [y_lower, y_upper],
                color=error_bar_color,
                linewidth=error_bar_linewidth,
                zorder=3,
            )  # zorder=3 使其在条形图之上
            # --- 绘制误差棒的 "帽子" (Caps) ---
            # 计算帽子半宽对应的角度偏移量 (弧度)
            cap_angle_offset_rad = np.radians(error_cap_size_deg / 2.0)
            # 计算上端帽子的两个端点的 x, y 坐标
            x_cap_upper1 = upper_radius * np.cos(center_angle_rad - cap_angle_offset_rad)
            y_cap_upper1 = upper_radius * np.sin(center_angle_rad - cap_angle_offset_rad)
            x_cap_upper2 = upper_radius * np.cos(center_angle_rad + cap_angle_offset_rad)
            y_cap_upper2 = upper_radius * np.sin(center_angle_rad + cap_angle_offset_rad)
            # 绘制上端帽子线段
            ax.plot(
                [x_cap_upper1, x_cap_upper2],
                [y_cap_upper1, y_cap_upper2],
                color=error_bar_color,
                linewidth=error_bar_linewidth,
                zorder=3,
            )
            # 计算下端帽子的两个端点的 x, y 坐标
            x_cap_lower1 = lower_radius * np.cos(center_angle_rad - cap_angle_offset_rad)
            y_cap_lower1 = lower_radius * np.sin(center_angle_rad - cap_angle_offset_rad)
            x_cap_lower2 = lower_radius * np.cos(center_angle_rad + cap_angle_offset_rad)
            y_cap_lower2 = lower_radius * np.sin(center_angle_rad + cap_angle_offset_rad)
            # 绘制下端帽子线段
            ax.plot(
                [x_cap_lower1, x_cap_lower2],
                [y_cap_lower1, y_cap_lower2],
                color=error_bar_color,
                linewidth=error_bar_linewidth,
                zorder=3,
            )
        # --- 绘制显著性字母 ---
        if show_significance and sig_letters:  # 检查是否需要显示，并且当前指标有显著性字母数据
            # 获取当前条形图 (索引 j) 对应的显著性字母
            sig_letter = sig_letters[j]
            if sig_letter:  # 仅当字母非空字符串时绘制 (允许留空表示无显著性标记)
                # 计算字母放置的半径 (误差棒顶端半径 + 设定的径向偏移量)
                sig_radius = upper_radius + sig_label_radial_offset
                # 计算字母放置位置的 x, y 坐标 (使用条形图中心角度)
                sig_x = sig_radius * np.cos(center_angle_rad)
                sig_y = sig_radius * np.sin(center_angle_rad)
                # --- 计算显著性字母的旋转角度，使其垂直于柱子方向 ---
                sig_angle_deg_norm = bar_center_angle_deg % 360  # 将角度标准化到 0-360 度
                if 0 <= sig_angle_deg_norm < 180:  # 如果在图的上半部分 (0 到 180 度)
                    calculated_sig_rotation = bar_center_angle_deg - 90  # 角度减 90 度使其基线朝外
                else:  # 如果在图的下半部分 (180 到 360 度)
                    calculated_sig_rotation = (
                        bar_center_angle_deg + 90
                    )  # 角度加 90 度使其基线也朝外且可读
                # 使用 ax.text 绘制显著性字母
                ax.text(
                    sig_x,
                    sig_y,
                    sig_letter,  # 坐标和文本内容
                    ha="center",  # 水平对齐方式: 居中
                    va="center",  # 垂直对齐方式: 居中 (对于旋转文本效果较好)
                    fontsize=sig_fontsize,  # 字体大小
                    color=sig_color,  # 文本颜色
                    rotation=calculated_sig_rotation,  # 使用计算得到的旋转角度
                    rotation_mode="anchor",  # 旋转模式: 以文本锚点为中心旋转
                    zorder=6,
                )  # 绘制层级 (最高，确保在所有元素之上)
        # --- 更新下一个条形图的起始角度 ---
        # 下一个条形图的起始角度 = 当前条形图起始角度 + 条形图宽度 + 条形图间隙
        current_bar_start_angle += bar_angular_width + intra_bar_gap_angle
    # --- 3. 添加内部参考虚线 和 数值标注文本 (1/4, 1/2, 3/4 值) ---
    if show_annotations and (segment_max > segment_min):  # 检查是否需要显示，并且数值范围有效
        value_range = segment_max - segment_min  # 计算数值范围宽度
        annotation_fractions = [0.25, 0.5, 0.75]  # 定义要标注的数值比例 (1/4, 1/2, 3/4)
        # 计算这些比例对应的实际数值
        annotation_values = [segment_min + frac * value_range for frac in annotation_fractions]
        # 使用 scale_value_to_radius 计算这些数值对应的内部半径 (用于绘制虚线和放置文本)
        internal_radii = [
            scale_value_to_radius(
                val, segment_min, segment_max, inner_radius, plotting_outer_radius
            )
            for val in annotation_values
        ]

        # 循环处理每个标注值 (1/4, 1/2, 3/4)
        for k in range(len(annotation_values)):
            current_val = annotation_values[k]  # 当前标注的数值
            current_internal_radius = internal_radii[k]  # 当前数值对应的内部半径

            # --- 绘制内部参考虚线 (弧形) ---
            if show_annotation_lines:  # 检查是否需要绘制虚线
                # 在扇区的起始和结束角度之间生成一系列角度点 (用于绘制平滑弧线)
                line_angles_deg = np.linspace(
                    segment_start_angle, segment_end_angle, annotation_line_points
                )
                line_angles_rad = np.radians(line_angles_deg)  # 转换为弧度
                # 计算这些角度点在当前内部半径上的 x, y 坐标
                line_x = current_internal_radius * np.cos(line_angles_rad)
                line_y = current_internal_radius * np.sin(line_angles_rad)
                # 使用 ax.plot 绘制虚线
                ax.plot(
                    line_x,
                    line_y,
                    linestyle=annotation_line_style,  # 线条样式
                    color=annotation_line_color,  # 线条颜色
                    linewidth=annotation_line_width,  # 线条宽度
                    zorder=3.5,
                )  # 绘制层级 (在条形图和误差棒之上，但在数值文本之下)

            # --- 放置数值标注文本 ---
            # 计算文本放置的角度 (扇区起始角度 + 用户设定的角度偏移量)
            target_text_angle_deg = segment_start_angle + annotation_outside_offset_deg
            target_text_angle_rad = np.radians(target_text_angle_deg)  # 转换为弧度
            # 计算文本放置的 x, y 坐标 (使用当前数值对应的内部半径 current_internal_radius)
            annot_x = current_internal_radius * np.cos(target_text_angle_rad)
            annot_y = current_internal_radius * np.sin(target_text_angle_rad)
            # 格式化要显示的数值文本 (保留一位小数)
            annot_text = f"{current_val:.1f}"

            # --- 计算数值标注文本的旋转角度和对齐方式 ---
            # 文本的基准旋转角度等于其放置角度
            base_rotation_deg = target_text_angle_deg
            angle_norm = base_rotation_deg % 360  # 标准化角度到 0-360
            # 根据文本在圆环上的位置调整旋转角度和垂直对齐方式，使其易于阅读
            if 90 < angle_norm < 270:  # 如果在图的左半部分
                final_rotation = base_rotation_deg + 180  # 旋转180度使文本朝外
                va_annot = "top"  # 垂直对齐方式
            else:  # 如果在图的右半部分
                final_rotation = (
                    base_rotation_deg + 180
                )  # 保持用户原来的设置 (可能是笔误，通常右侧也希望朝外，应该是 final_rotation = base_rotation_deg)
                # 如果希望右侧也朝外： final_rotation = base_rotation_deg
                # 如果希望所有文本都统一旋转（如都水平）： final_rotation = 0
                va_annot = "top"  # 保持用户原来的设置
            ha_annot = "center"  # 水平对齐方式: 居中

            # 绘制数值标注文本
            ax.text(
                annot_x,
                annot_y,
                annot_text,  # 坐标和文本
                ha=ha_annot,
                va=va_annot,  # 对齐方式
                fontsize=annotation_fontsize,  # 字体大小
                color=annotation_color,  # 颜色
                rotation=final_rotation,  # 旋转角度
                rotation_mode="anchor",  # 旋转模式
                zorder=4,
            )  # 绘制层级 (在虚线之上)
    # --- 4. 添加指标类别标签 (如 "SOC", "MAOC"...) 在 gap 中 ---
    # 标签放置在扇区的中心角度
    segment_mid_angle_deg_cat = segment_center_angles[i]
    segment_mid_angle_rad_cat = np.radians(segment_mid_angle_deg_cat)  # 转换为弧度
    # 计算标签放置的 x, y 坐标 (使用之前计算好的 category_label_radius)
    label_x = category_label_radius * np.cos(segment_mid_angle_rad_cat)
    label_y = category_label_radius * np.sin(segment_mid_angle_rad_cat)
    # 标准化角度到 0-360
    angle_deg_norm_cat = segment_mid_angle_deg_cat % 360
    # --- 计算类别标签的旋转角度和对齐方式，使其沿径向向外且易读 ---
    if 90 < angle_deg_norm_cat < 270:  # 如果在图的左半部分
        rotation_angle_cat = angle_deg_norm_cat - 90  # 旋转角度使其基线沿切线方向
        ha_cat = "right"  # 水平对齐方式
    else:  # 如果在图的右半部分
        rotation_angle_cat = angle_deg_norm_cat + 270  # 旋转角度使其基线沿切线方向
        ha_cat = "left"  # 水平对齐方式
    va_cat = "center"  # 垂直对齐方式: 居中
    # 绘制类别标签文本
    ax.text(
        label_x,
        label_y,
        metric_label,  # 坐标和文本
        ha=ha_cat,
        va=va_cat,  # 对齐方式
        rotation=rotation_angle_cat,  # 旋转角度
        rotation_mode="anchor",  # 旋转模式
        fontsize=label_fontsize,  # 字体大小
        zorder=5,
    )  # 绘制层级 (在数值标注之上，但在显著性标记之下)
# --- 添加图例 ---
# 创建图例句柄 (Patch 对象列表)，每个对象代表一种颜色和标签
# --- 添加图例 ---
# 创建图例句柄 (Line2D 对象列表)，使用正方形标记代表颜色
legend_handles = [
    plt.Line2D(
        [0],
        [0],  # 创建一个 Line2D 对象 (坐标不重要)
        marker="s",  # 设置标记形状为正方形 ('s' for square)
        color="w",  # 线条颜色设为白色或透明 (不重要，因为线条不显示)
        markerfacecolor=color,  # 设置标记填充颜色为对应的处理颜色
        markersize=legend_fontsize * 0.8,  # 设置标记大小 (可以根据字体大小调整)
        linestyle="None",  # 不显示线条本身
        label=label,
    )  # 设置标签
    for color, label in zip(treatment_colors, treatment_labels)
]  # 遍历颜色和标签
# 在轴域上显示图例
ax.legend(
    handles=legend_handles,  # 图例项列表 (现在是 Line2D 对象)
    loc="upper right",  # 图例位置: 右上角
    fontsize=legend_fontsize,  # 图例字体大小
    handletextpad=0.02,  # 图例标记与文字之间的水平间隔
    frameon=False,
)  # 不显示图例边框
plt.savefig(str(OUTPUT_DIR / "环形柱状图.png"), dpi=1080)
# --- 显示最终绘制好的图形 ---
plt.close("all")  # Interactive display removed; assets were exported above.

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

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

# --- 导入所需库 ---
# 导入 pandas 库，用于数据处理和读写 Excel 文件，通常简写为 pd
# 导入 os 库，用于与操作系统交互，如此处用于检查和创建文件夹
import os

# 导入 matplotlib 主库，以便设置其后端
import matplotlib

# 导入 matplotlib.pyplot 库，是 Python 最核心的绘图库，提供底层的绘图接口，通常简写为 plt
import matplotlib.pyplot as plt

# 导入 numpy 库，用于进行高效的数值计算，特别是数组操作，通常简写为 np
import numpy as np
import pandas as pd

# 导入 scipy.stats 库，用于科学计算和统计分析，通常简写为 stats
import scipy.stats as stats

# 导入 seaborn 库，一个基于 matplotlib 的高级可视化库，用于绘制美观的统计图形，通常简写为 sns
import seaborn as sns

# 从 matplotlib.lines 模块导入 Line2D 类，可用于创建自定义图例 (此脚本中未直接使用)
from matplotlib.lines import Line2D

# --- 后端设置与全局字体设置 ---
# 设置 matplotlib 的图形后端为 'TkAgg'。这是一个兼容性较好的后端，有助于在不同操作系统上正确弹出图形窗口。
matplotlib.use("Agg")
# 设置全局字体。通过 'font.family' 参数将图表中所有文本的默认字体设置为 'Times New Roman'。
plt.rcParams["font.family"] = "Times New Roman"
# 设置全局参数以正常显示负号。当使用某些字体时，负号可能会显示为方框，此设置可解决该问题。
plt.rcParams["axes.unicode_minus"] = False


# --- 辅助函数：将p值转换为显著性符号 ---
# 定义一个名为 p_to_symbol 的函数，它接收一个参数 p (p-value)
def p_to_symbol(p):
    """
    此函数接收一个p值，并根据其大小返回对应的学术惯用显著性符号。
    """
    # 如果 p 值大于 0.05，表示差异不显著，返回 'ns' (not significant)。
    if p > 0.05:
        return "ns"
    # 如果 p 值小于等于 0.0001，返回四个星号，表示极极显著。
    if p <= 0.0001:
        return "****"
    # 如果 p 值小于等于 0.001，返回三个星号，表示极显著。
    if p <= 0.001:
        return "***"
    # 如果 p 值小于等于 0.01，返回两个星号，表示非常显著。
    if p <= 0.01:
        return "**"
    # 如果 p 值小于等于 0.05，返回一个星号，表示显著。
    if p <= 0.05:
        return "*"
    # 如果 p 值不符合以上任何条件（例如为空），则返回一个空字符串。
    return ""


# --- 新增辅助函数：根据显著性符号返回颜色 ---
# 定义一个名为 get_significance_color 的函数，接收一个参数 symbol (符号字符串)
def get_significance_color(symbol):
    """
    根据输入的显著性符号返回一个特定的颜色。
    星号越多，颜色越醒目。
    """
    # 创建一个字典，将每个符号映射到一种颜色代码。
    color_map = {
        "****": "#d62728",  # 最显著：深红色
        "***": "#ff7f0e",  # 极显著：橙色
        "**": "#1f77b4",  # 非常显著：蓝色
        "*": "#9467bd",  # 显著：紫色
        "ns": "grey",  # 不显著：灰色
    }
    # 从字典中获取颜色。如果符号不在字典中（例如空字符串），则默认返回黑色。
    return color_map.get(symbol, "black")


# --- 1. 加载数据 ---
# 使用原始字符串(r"...")来定义文件路径，避免反斜杠被误解为转义字符。
file_path = str(DATA_DIR / "data.xlsx")
# 使用 pandas 的 read_excel 函数读取指定路径的 Excel 文件。
my_data = pd.read_excel(file_path)
# 在控制台打印消息，告知用户 Excel 文件已成功加载。
print("Excel文件加载成功。")

# --- 2. 准备数据 ---
# 定义感兴趣的数值列（因变量）的列名。
value_column_name = "HL(g/kg)"
# 定义分组列（自变量）的列名。
group_column_name = "Feature"
# 定义一个列表，指定两个分组在图表上显示的期望顺序。
group_order = ["clayconent", "organic"]
# 从原始 DataFrame 中选择需要的两列，并使用 .copy() 创建一个副本。
plot_data = my_data[[group_column_name, value_column_name]].copy()
# 使用 rename 方法重命名列，增强代码的可读性和复用性。
plot_data.rename(columns={group_column_name: "Group", value_column_name: "Value"}, inplace=True)
# 过滤数据，只保留 'group_order' 中定义的分组，以防文件中有多余的组。
plot_data = plot_data[plot_data["Group"].isin(group_order)]
# 将 'Group' 列的数据类型转换为 'Categorical'（分类），并指定其顺序。
plot_data["Group"] = pd.Categorical(plot_data["Group"], categories=group_order, ordered=True)
# 根据 'Group' 列的分类顺序对整个 DataFrame 进行排序。
plot_data.sort_values("Group", inplace=True)

# --- 3. 准备统计数据 ---
# 在控制台打印将要使用的检验方法。
print("\n--- 检验方法：秩和检验 (Mann-Whitney U test) ---")
# 创建一个字典，键是组名，值是该组对应的数据，以便后续调用。
groups_data = {g: plot_data["Value"][plot_data["Group"] == g].dropna() for g in group_order}

# --- 4. 绘图参数定义 ---
# 在控制台打印消息，告知用户程序正在开始生成图片。
print("\n--- 正在生成图片 ---")
# 定义坐标轴标题的字体大小。
axis_title_fontsize = 10
# 定义坐标轴刻度标签的字体大小。
axis_text_fontsize = 10
# 定义显著性标记（如星号）的字体大小。
signif_marker_fontsize = 10
# 为两个组定义两种颜色。
colors = ["#f0b87f", "#e59698"]
# 创建一个将组名映射到颜色的字典。
color_dict = {group: color for group, color in zip(group_order, colors)}
# 定义图表外边框（spines）的线条宽度。
panel_border_linewidth = 1
# 定义坐标轴刻度线的线条宽度。
tick_linewidth = 1
# 定义坐标轴刻度线的长度。
tick_length = 3
# 定义小提琴图轮廓的线条宽度。
violin_linewidth = 1
# 定义手动绘制的箱线图中，箱体边框的线条宽度。
box_border_linewidth = 1.0
# 定义手动绘制的箱线图中，中位数线的线条宽度。
box_median_linewidth = 1.5
# 定义手动绘制的箱线图中，上下须线的线条宽度。
box_whisker_linewidth = 1.0

# --- 5. 创建画布和绘图 ---
# 调用 plt.subplots() 创建一个图形（Figure, fig）和一个子图（Axes, ax），并设置合适的尺寸和紧凑布局。
fig, ax = plt.subplots(figsize=(4.6, 4), constrained_layout=True)

# A. 绘制小提琴图
# 调用 seaborn 的 violinplot 函数。
sns.violinplot(
    data=plot_data,
    x="Group",
    y="Value",
    hue="Group",
    inner=None,  # 不在内部绘制任何图形。
    ax=ax,  # 指定在哪个子图上绘制。
    legend=False,  # 不显示图例。
    order=group_order,  # 按指定顺序排列组。
    cut=0,  # 裁剪尾部到数据范围。
    density_norm="count",  # 宽度与样本量成正比。
)

# 自定义小提琴图样式
# 遍历小提琴图对象。
for i, violin in enumerate(ax.collections):
    # 确保只操作小提琴图。
    if i < len(group_order):
        # 设置轮廓颜色。
        violin.set_edgecolor(color_dict[group_order[i]])
        # 设置填充为透明。
        violin.set_facecolor("none")
        # 设置轮廓线宽。
        violin.set_linewidth(violin_linewidth)

# B. 绘制散点图
# 在图上叠加原始数据点。
sns.stripplot(
    data=plot_data,
    x="Group",
    y="Value",
    hue="Group",
    palette=colors,
    jitter=True,  # 添加水平抖动。
    size=5,  # 设置点的大小。
    alpha=0.3,  # 设置透明度。
    ax=ax,  # 在同一个子图上绘制。
    legend=False,  # 不显示图例。
    order=group_order,  # 确保顺序一致。
    marker="o",  # 设置标记形状为圆形。
)

# C. 手动绘制箱线图
# 定义箱体宽度。
box_width = 0.25
# 遍历每个组。
for i, group in enumerate(group_order):
    # 提取当前组的数据。
    group_data = plot_data[plot_data["Group"] == group]["Value"].dropna()
    # 如果组内无数据，则跳过，防止报错。
    if group_data.empty:
        continue
    # 计算四分位数。
    q1, median, q3 = np.percentile(group_data, [25, 50, 75])
    # 计算四分位距。
    iqr = q3 - q1
    # 计算须线末端位置。
    whisker_low_limit = q1 - 1.5 * iqr
    whisker_high_limit = q3 + 1.5 * iqr
    actual_whisker_low = group_data[group_data >= whisker_low_limit].min()
    actual_whisker_high = group_data[group_data <= whisker_high_limit].max()
    # 获取当前组的颜色。
    box_color = color_dict[group]
    # 创建矩形对象作为箱体。
    box = plt.Rectangle(
        (i - box_width / 2, q1),
        box_width,
        q3 - q1,
        edgecolor=box_color,
        facecolor=box_color,
        alpha=0.7,
        zorder=2,
        linewidth=box_border_linewidth,
    )
    # 将箱体添加到图上。
    ax.add_patch(box)
    # 绘制中位数线。
    ax.plot(
        [i - box_width / 2, i + box_width / 2],
        [median, median],
        color=box_color,
        lw=box_median_linewidth,
        zorder=3,
        solid_capstyle="butt",
    )
    # 绘制上须线。
    ax.plot(
        [i, i],
        [q3, actual_whisker_high],
        color=box_color,
        lw=box_whisker_linewidth,
        zorder=3,
        solid_capstyle="butt",
    )
    # 绘制下须线。
    ax.plot(
        [i, i],
        [q1, actual_whisker_low],
        color=box_color,
        lw=box_whisker_linewidth,
        zorder=3,
        solid_capstyle="butt",
    )

# --- 6. 主题美化 ---
# 设置X轴标签。
ax.set_xlabel("Group", fontsize=axis_title_fontsize, weight="bold")
# 设置Y轴标签。
ax.set_ylabel("Value (g/kg)", fontsize=axis_title_fontsize, weight="bold")
# 设置刻度线参数。
ax.tick_params(
    axis="both",
    which="major",
    labelsize=axis_text_fontsize,
    width=tick_linewidth,
    length=tick_length,
    direction="out",
    color="black",
)
# 遍历并设置所有刻度标签为粗体和黑色。
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
    label.set_color("black")
# 设置绘图区背景为白色。
ax.set_facecolor("white")
# 设置整个画布背景为白色。
fig.set_facecolor("white")
# 移除背景网格。
ax.grid(False)
# 遍历并设置所有边框的线宽和颜色。
for spine in ax.spines.values():
    spine.set_linewidth(panel_border_linewidth)
    spine.set_color("black")

# --- 7. 添加显著性标记 ---
# 在控制台打印标题。
print("\n--- 进行秩和检验并添加显著性标记 ---")

# 动态确定标记线的y轴位置：取所有数据中的最大值，再往上加一点距离。
y_max = plot_data["Value"].max()
# 在最大值上方5%的位置绘制标记线。
y_pos = y_max * 1.05
# 定义标记线的宽度。
line_width = 1

# 提取两组的数据。
group1_data = groups_data["clayconent"]
group2_data = groups_data["organic"]

# 确保两组都有数据才进行检验，避免报错。
if not group1_data.empty and not group2_data.empty:
    # 执行Mann-Whitney U检验（秩和检验），比较两组数据。
    u_stat, p_val = stats.mannwhitneyu(group1_data, group2_data, alternative="two-sided")
    # 打印p值结果。
    print(f"clayconent vs organic: Mann-Whitney U p-value = {p_val:.4g}")

    # 将p值转换为星号等显著性符号。
    symbol = p_to_symbol(p_val)

    # 根据显著性符号获取对应的颜色。
    significance_color = get_significance_color(symbol)

    # 使用获取到的颜色绘制连接线。
    ax.plot([0, 1], [y_pos, y_pos], color=significance_color, lw=line_width)
    # 使用获取到的颜色绘制显著性符号文本。
    ax.text(
        0.5,
        y_pos,
        symbol,
        ha="center",
        va="bottom",
        color=significance_color,
        fontsize=signif_marker_fontsize,
    )

    # 调整 Y 轴的上限，以确保所有显著性标记都能完整显示。
    ax.set_ylim(top=y_pos * 1.1)
else:
    # 如果有任何一组数据为空，则打印提示信息。
    print("至少有一个组没有数据，无法进行统计检验。")

# --- 8. 保存图片 ---
# 定义图片输出的目标文件夹路径。
output_folder = str(OUTPUT_DIR)
# 定义输出图片的文件名。
output_filename = str(OUTPUT_DIR / "py_plot_2_groups_colored_significance.png")
# 检查输出文件夹是否存在。
if not os.path.exists(output_folder):
    # 如果文件夹不存在，则创建它。
    os.makedirs(output_folder, exist_ok=True)
# 使用 os.path.join 将文件夹路径和文件名安全地拼接成一个完整的输出路径。
output_path = os.path.join(output_folder, output_filename)
# 调用 plt.savefig 函数保存图形，dpi=1080 设置高分辨率。
plt.savefig(output_path, dpi=1080)

# 在控制台打印最终的保存路径。
print(f"\n图片已成功保存至: {output_path}")

# --- 显示图形 ---
# 调用 plt.show()，在图形界面窗口中显示最终生成的图表。
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

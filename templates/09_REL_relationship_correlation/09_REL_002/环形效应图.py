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

# 导入matplotlib的绘图库，并简写为plt
# 导入matplotlib主库
import matplotlib

# 导入matplotlib的patches模块，用于创建图例中的色块
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# 导入numpy库，用于处理数值和数组，简写为np
import numpy as np

# 从matplotlib.lines模块导入Line2D，用于创建自定义的线型图例
from matplotlib.lines import Line2D

# --- 后端设置 ---
# 尝试设置matplotlib的后端为'TkAgg'，这有助于在某些环境下正确显示图形窗口
try:
    matplotlib.use("Agg")
# 如果导入TkAgg后端失败（例如，环境中没有安装Tkinter），则打印提示信息
except ImportError:
    print("TkAgg backend not available, using default.")
# --- 1. 数据准备 ---
# --- 全局参数设置 ---
# 设置matplotlib参数，允许坐标轴正确显示负号
plt.rcParams["axes.unicode_minus"] = False
# 设置全局默认字体为'serif'（衬线字体族）
plt.rcParams["font.family"] = "serif"
# 在'serif'字体族中，优先使用'Times New Roman'字体
plt.rcParams["font.serif"] = ["Times New Roman"]
# --- 定义绘图数据 ---
# 定义每个区域内变量的名称
variables = ["SWC", "TEM", "PRE", "ST"]
# 定义区域的名称
regions = ["R1", "R2", "R3", "R4", "R5"]
# 计算每个区域包含的变量数量
num_vars_per_region = len(variables)
# 计算区域的总数
num_regions = len(regions)
# 计算所有条目的总类别数（变量数 * 区域数）
num_categories = num_vars_per_region * num_regions
# --- 效应数据 ---
# 定义直接效应的示例数据，是一个一维numpy数组
direct_effects = np.array(
    [
        # R1区域的数据
        0.30,
        0.54,
        0.05,
        0.017,
        # R2区域的数据
        -0.21,
        0.42,
        -0.10,
        0.09,
        # R3区域的数据
        0.32,
        0.23,
        0.06,
        0.03,
        # R4区域的数据
        0.33,
        0.14,
        0.03,
        -0.15,
        # R5区域的数据
        0.20,
        0.37,
        0.06,
        -0.02,
    ]
)
# 定义间接效应的示例数据，是一个一维numpy数组
indirect_effects = np.array(
    [
        # R1
        0.15,
        -0.36,
        0.22,
        0.34,
        # R2
        0.23,
        0.16,
        0.51,
        0.33,
        # R3
        0.27,
        -0.34,
        0.12,
        -0.3,
        # R4
        0.12,
        -0.26,
        -0.23,
        0.44,
        # R5
        0.33,
        -0.12,
        0.24,
        -0.53,
    ]
)

# --- 2. 设置图像、角度和间距 ---
# 创建一个图形（fig）和一个子图（ax），设置图形大小为12x12英寸
# subplot_kw指定子图为极坐标投影（polar projection）
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

# 您可以在这里直接定义柱子、小间隙和大间隙的基础宽度（单位：弧度）
# 代码会自动将它们按比例缩放以填满整个圆周
bar_width_base = 0.04  # 每个柱子的基础宽度
small_gap_base = 0.02  # 区域内柱子间的小间隙的基础宽度
large_gap_base = 0.25  # 区域间的大间隙的基础宽度
# 设置整个图形的起始旋转角度（单位：度）
rotation_angle_degrees = 7.0
# --- 参数结束 ---

# --- 自动计算角度和宽度 (新方法) ---
# 计算总共有多少个柱子之间的小间隙
num_inner_gaps = num_categories - num_regions
# 计算总共有多少个区域之间的大间隙
num_region_gaps = num_regions
# 计算所有指定的基础宽度加起来的总弧度
total_base_width = (
    (num_categories * bar_width_base)
    + (num_inner_gaps * small_gap_base)
    + (num_region_gaps * large_gap_base)
)
# 计算缩放因子，用圆周的总弧度(2*pi)除以基础宽度总和
# 这样可以确保所有元素按比例缩放后恰好填满整个圆
scale_factor = (2 * np.pi) / total_base_width

# 根据缩放因子计算最终在图上使用的实际宽度
bar_width = bar_width_base * scale_factor
small_gap_rad = small_gap_base * scale_factor
large_gap_rad = large_gap_base * scale_factor

# --- 计算每个柱子的中心角度 ---
# 初始化一个空列表，用于存放每个柱子的中心角度
center_angles = []
# 设置起始位置，将旋转角度从度转换为弧度
current_pos = np.deg2rad(rotation_angle_degrees)
# 遍历每个区域
for i in range(num_regions):
    # 在每个区域内遍历每个变量
    for j in range(num_vars_per_region):
        # 计算当前柱子的中心角度
        angle = current_pos + bar_width / 2
        # 将计算出的角度添加到列表中
        center_angles.append(angle)
        # 更新当前位置，移过一个柱子的宽度
        current_pos += bar_width
        # 如果是区域内的最后一个变量
        if j == num_vars_per_region - 1:
            # 在位置上增加一个大间隙
            current_pos += large_gap_rad
        # 否则
        else:
            # 在位置上增加一个小间隙
            current_pos += small_gap_rad
# --- 间距计算结束 ---

# --- 定义颜色 ---
# 定义直接效应柱子的颜色
color_direct = "#e66101"
# 定义间接效应柱子的颜色
color_indirect = "#fdb863"

# --- 3. 核心条件绘图逻辑 ---
# 遍历所有类别（每个柱子）
for i in range(num_categories):
    # 获取当前柱子的直接效应值
    direct_val = direct_effects[i]
    # 获取当前柱子的间接效应值
    indirect_val = indirect_effects[i]
    # 获取当前柱子的中心角度
    angle = center_angles[i]

    # --- 绘制堆叠柱状图 ---
    # 如果直接效应和间接效应同号（或其中一个为0）
    if direct_val * indirect_val >= 0:
        # 绘制直接效应的柱子，设置图层顺序为3
        ax.bar(angle, direct_val, width=bar_width, color=color_direct, zorder=3)
        # 在直接效应柱子的顶部继续绘制间接效应的柱子（形成堆叠效果）
        ax.bar(
            angle, indirect_val, width=bar_width, bottom=direct_val, color=color_indirect, zorder=3
        )
    # 如果两者异号
    else:
        # 分别绘制直接效应的柱子
        ax.bar(angle, direct_val, width=bar_width, color=color_direct, zorder=3)
        # 分别绘制间接效应的柱子（此时不会堆叠，会各自从0点向不同方向延伸）
        ax.bar(angle, indirect_val, width=bar_width, color=color_indirect, zorder=3)

    # --- 绘制总效应标记线 ---
    # 计算总效应（直接效应 + 间接效应）
    total_effect = direct_val + indirect_val
    # 定义总效应标记线的角度宽度（为柱子宽度的70%）
    line_angular_width = bar_width * 0.65
    # 计算标记线的起始角度
    angle_start = angle - line_angular_width / 2
    # 计算标记线的结束角度
    angle_end = angle + line_angular_width / 2
    # 在总效应值的位置绘制一条深红色的粗横线作为标记
    ax.plot(
        [angle_start, angle_end],
        [total_effect, total_effect],
        color="darkred",  # 设置颜色为深红色
        linewidth=2.5,  # 设置线宽为2.5
        zorder=5,
    )  # 设置图层顺序为5，确保它在柱子和0值线之上

# --- 4. 自定义和美化图像 ---
# --- 坐标轴和网格线设置 ---
# 设置极坐标的0度位置在正北方（'N'）
ax.set_theta_zero_location("N")
# 设置极坐标的角度增长方向为顺时针（-1）
ax.set_theta_direction(-1)
# 设置y轴（半径轴）的显示范围
ax.set_ylim(-0.8, 0.7)
# 设置y轴（半径轴）刻度标签的显示角度位置（0度）
ax.set_rlabel_position(0)
# 设置y轴刻度标签的字体大小
ax.tick_params(axis="y", labelsize=10)
# 禁用x轴（角度轴）的网格线
ax.xaxis.grid(False)
# 隐藏极坐标图最外围的边框（spines）
ax.spines["polar"].set_visible(False)
# 设置y轴（半径轴）要显示的刻度位置
ax.set_yticks([-0.7, -0.5, -0.2, 0, 0.2, 0.5, 0.7])
# 启用y轴（半径轴）的网格线，并设置为灰色虚线
ax.yaxis.grid(True, linestyle="--", color="black", linewidth=0.8, zorder=1)

# --- 绘制0值参考线 ---
# 定义0值线圆形缺口的弧度大小
gap_radians = 0.03
# 创建一个带有缺口的角度数组（从缺口一端到另一端）
theta_with_gap = np.linspace(gap_radians, 2 * np.pi - gap_radians, 200)
# 绘制带有缺口的0值参考线，并设置图层顺序为4
ax.plot(
    theta_with_gap,
    np.zeros_like(theta_with_gap),
    color="black",
    linewidth=1.4,
    linestyle="--",
    zorder=1,
)

# --- 美化y轴刻度标签 ---
# 遍历所有的y轴刻度标签
for label in ax.get_yticklabels():
    # 设置标签垂直居中对齐
    label.set_verticalalignment("center")
    # 给标签添加一个无边框的白色背景，以遮挡下方的网格线
    label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.2))
    # 设置标签的图层顺序为10，确保它在最上层
    label.set_zorder(10)
    # ******************** 代码修改部分 开始 ********************
    # 设置数值标注的字体大小，例如设置为14
    label.set_fontsize(14)
    # 设置数值标注的字体粗细，例如设置为'bold'（粗体）或'normal'（常规）
    # label.set_fontweight('bold')
    # ******************** 代码修改部分 结束 ********************
# --- 添加文本标签 ---
# 禁用原始的、位于外圈的角度刻度标签
ax.set_xticks(center_angles)
ax.set_xticklabels([])

# --- 在柱子顶端添加变量标签 ---
# 创建一个包含所有变量标签的列表
labels_list = [var for region in regions for var in variables]
# 设置标签与柱子顶端的间距
padding = 0.035
# 遍历所有类别
for i in range(num_categories):
    # 获取当前标签的文本
    label_text = labels_list[i]
    # 获取当前类别的直接效应值
    direct_val = direct_effects[i]
    # 获取当前类别的间接效应值
    indirect_val = indirect_effects[i]
    # 获取当前类别的中心角度
    angle_rad = center_angles[i]

    # 计算标签应该放置的半径位置
    # 首先计算正方向上柱子的最外端位置
    outer_tip_radius = max(0, direct_val) + max(0, indirect_val)
    # 如果柱子在正方向上有长度
    if outer_tip_radius > 0:
        # 标签位置在最外端再加一个padding
        label_radius = outer_tip_radius + padding
    # 如果柱子都在负方向
    else:
        # 计算总效应
        total_effect_val = direct_val + indirect_val
        # 如果总效应为负
        if total_effect_val < 0:
            # 标签放在0半径位置外侧一点
            label_radius = 0 + padding
        # 如果总效应为0或正（但在外圈没有柱子），则不添加标签
        else:
            continue
    # 设置标签旋转角度为0（即不旋转）
    rotation = 0
    # 在计算好的位置添加文本标签
    ax.text(
        angle_rad,
        label_radius,
        label_text,
        ha="center",
        va="center",  # 设置水平和垂直对齐方式为居中
        fontsize=12,
        fontweight="bold",
        color="black",  # 设置字体大小和颜色
        rotation=rotation,
        zorder=11,
    )  # 设置图层顺序为11，确保在所有元素之上

# --- 添加内部的区域标签 ---
# 该部分用于在图内部添加区域标签（R1, R2等）
# 初始化一个列表存放每个区域标签的角度
region_label_angles = []
# 计算区域标签角度的偏移量，使其位于区域的中心
angle_offset_for_region_label = (
    num_vars_per_region * bar_width + (num_vars_per_region - 1) * small_gap_rad
) / 2
# 遍历每个区域
for i in range(num_regions):
    # 计算每个区域的起始角度
    start_of_region = np.deg2rad(rotation_angle_degrees) + i * (
        (num_vars_per_region * bar_width)
        + ((num_vars_per_region - 1) * small_gap_rad)
        + large_gap_rad
    )
    # 计算并存储区域标签的中心角度
    region_label_angles.append(start_of_region + angle_offset_for_region_label)

# 遍历计算好的区域标签角度
for i, angle in enumerate(region_label_angles):
    # 在半径0.5的位置添加区域标签文本
    ax.text(
        angle,
        -0.1,
        regions[i],
        horizontalalignment="center",
        verticalalignment="center",  # 设置水平和垂直居中
        fontsize=16,
        fontweight="bold",
        color="black",  # 设置字体大小、粗细和颜色
        zorder=0.5,  # 设置图层顺序为0.5，使其位于最底层
        # 为标签添加一个带圆角的、半透明的白色背景，以提高可读性
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
    )

# --- 创建和放置图例 ---
# 创建一个直接效应的图例（橙色方块）
direct_patch = mpatches.Patch(color=color_direct, label="Standardized direct effects")
# 创建一个间接效应的图例（浅橙色方块）
indirect_patch = mpatches.Patch(color=color_indirect, label="Standardized indirect effects")
# 为总效应创建一个线型图例（深红色横线），以匹配图中的标记
total_line_legend = Line2D([0], [0], color="darkred", lw=2.5, label="Standardized total effects")

# 将图例放置在图表下方
ax.legend(
    handles=[total_line_legend, direct_patch, indirect_patch],  # 定义图例中要显示的项目
    loc="upper center",  # 设置图例框内部的对齐方式
    bbox_to_anchor=(0.5, 0),  # 将图例的锚点定位在图表下方
    ncol=3,  # 设置图例分为3列
    prop={"size": 16, "weight": "bold"},  # 使用prop参数来设置字体大小和粗细
    frameon=False,
)  # 不显示图例的边框
# --- 添加标题 ---
# 在指定位置添加图表标题"a(1)"
# ax.set_title("a(1)", loc='left', x=0.1, y=0.9, fontsize=16, fontweight='bold')

# --- 显示图表 ---
# 自动调整布局，以防止标签或图例被裁切
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / "1.png"), dpi=300)
# 显示最终生成的图表
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

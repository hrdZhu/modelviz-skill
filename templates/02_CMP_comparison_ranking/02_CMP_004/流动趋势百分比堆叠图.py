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

# 导入matplotlib的pyplot模块，用于创建图表，通常简写为plt
# 导入os模块，用于操作系统相关功能，如此处的创建文件夹
import os

# 导入matplotlib的颜色处理模块，用于颜色转换等高级操作
import matplotlib.colors
import matplotlib.pyplot as plt

# 导入numpy库，用于进行高效的数值计算，特别是数组操作，通常简写为np
import numpy as np


# --- 颜色处理辅助函数 (代码不变) ---
# 定义一个名为lighten_color的函数，用于将输入的颜色变浅
def lighten_color(color_hex, factor=1.2):
    """
    接收一个matplotlib可识别的颜色，返回一个更亮的版本。
    factor > 1 会使颜色变亮。
    factor < 1 会使颜色变暗。
    """
    # 将十六进制或其他格式的颜色码转换为RGB元组（每个值的范围在0到1之间）
    rgb = matplotlib.colors.to_rgb(color_hex)
    # 将RGB色彩空间转换为HSV（色相、饱和度、亮度）色彩空间，便于调整亮度
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    # 提高HSV中的亮度值V（索引为2），并使用min函数确保其最大值不超过1
    hsv[2] = min(1, hsv[2] * factor)
    # 将调整亮度后的HSV颜色转换回RGB色彩空间
    lighter_rgb = matplotlib.colors.hsv_to_rgb(hsv)
    # 将最终的RGB元组转换回matplotlib通用的十六进制颜色码并返回
    return matplotlib.colors.to_hex(lighter_rgb)


# --- 新增功能 1: 定义10套科研风格配色方案 ---
# 创建一个字典来存储所有配色方案
scientific_palettes = {
    "Tableau_10": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "Colorblind_1": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"],
    "Viridis": [plt.cm.viridis(i) for i in np.linspace(0.1, 0.9, 6)],
    "Plasma": [plt.cm.plasma(i) for i in np.linspace(0.1, 0.9, 6)],
    "Paired": [plt.cm.Paired(i) for i in np.linspace(0, 1, 6)],
    "Set2": [plt.cm.Set2(i) for i in np.linspace(0, 1, 6)],
    "Muted_Tones": ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4", "#8c613c"],
    "Pastel_Tones": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b"],
    "Cividis": [plt.cm.cividis(i) for i in np.linspace(0.1, 0.9, 6)],
    "Original_Style": ["#d57d72", "#f9e097", "#65b2a8", "#bda8cd", "#a4cce3", "#e47369"],
}

# --- 新增功能 2: 设置并创建用于保存结果的文件夹 ---
# 定义一个字符串变量，作为保存输出图片的文件夹名称
output_folder = str(OUTPUT_DIR)
# 使用os模块创建这个文件夹；exist_ok=True表示如果文件夹已存在，则不执行任何操作，也不会报错
os.makedirs(output_folder, exist_ok=True)

# --- 1. 全局样式配置 (代码不变) ---
# 定义一个全局变量来统一控制所有色块的透明度
alpha_value = 0.8
# 设置matplotlib的全局字体族为衬线字体（serif），如Times New Roman
plt.rcParams["font.family"] = "serif"
# 在衬线字体族中，具体指定优先使用'Times New Roman'字体
plt.rcParams["font.serif"] = ["Times New Roman"]
# 设置全局字体默认粗细为粗体
plt.rcParams["font.weight"] = "bold"
# 再次明确设置坐标轴标签的字体为粗体，确保生效
plt.rcParams["axes.labelweight"] = "bold"
# 设置该参数以确保在使用某些字体时，图表中的负号可以正常显示，而不是一个方框
plt.rcParams["axes.unicode_minus"] = False

# --- 2. 准备数据 (代码不变) ---
# 定义一个列表，用作图表X轴上的分类标签
parts = ["Part1", "Part2", "Part3", "Part4", "Part5", "Part6"]
# 定义一个字典来存储绘图所需的核心数据
data = {
    "F": np.array([12, 12, 12, 12, 10, 5]),
    "E": np.array([10, 10, 26, 6, 8, 13]),
    "D": np.array([10, 23, 4, 22, 17, 17]),
    "C": np.array([18, 17, 8, 10, 5, 15]),
    "B": np.array([30, 18, 35, 30, 50, 20]),
    "A": np.array([20, 20, 15, 20, 10, 30]),
}

# --- 新增功能 3: 创建主循环，为每种配色方案生成并保存一张图表 ---
# 开始一个主循环，遍历我们预先定义的配色方案字典
for palette_name, colors in scientific_palettes.items():
    # 在每次循环开始时，打印当前正在处理的配色方案名称，以显示进度
    print(f"正在使用配色方案 '{palette_name}' 生成图表...")

    # --- 3. 开始绘图 (您的核心绘图代码被包裹在这个循环内) ---
    # 创建一个新的图形（fig）和一组子图/坐标系（ax），并设置整个图形的大小为10x7英寸
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- 3.1 绘制无边框的堆叠柱状图和内部数值 ---
    # 初始化一个全为0的numpy数组，用于记录每个堆叠柱状图的起始高度（即“底部”位置）
    bottom = np.zeros(len(parts))
    # 开始一个循环，使用zip函数同时遍历数据字典的项（(标签, 值)对）以及当前循环中的颜色列表
    for (label, values), color in zip(data.items(), colors):
        # 调用bar函数在坐标系ax上绘制一层柱状图
        ax.bar(
            parts,  # X轴位置，使用分类标签
            values,  # 当前层的高度
            label=label,  # 为当前层设置图例标签
            width=0.5,  # 设置柱子的宽度
            bottom=bottom,  # 设置当前层的起始Y坐标
            color=color,  # 设置填充颜色（来自当前循环的配色方案）
            linewidth=0,  # 设置边框线宽为0，即无边框
            alpha=alpha_value,  # 应用全局透明度
        )
        # 开始一个内层循环，遍历当前类别下的每一个数值，用于在其对应色块上添加文本标签
        for i, value in enumerate(values):
            # 判断数值是否大于0，大于0才添加标签，避免为0的色块也显示数字"0"
            if value > 0:
                # 计算文本标签的Y坐标，使其位于当前色块的垂直中心
                y_pos = bottom[i] + value / 2
                # 调用text函数在图上指定位置添加文本
                ax.text(i, y_pos, f"{value}", ha="center", va="center", color="white", fontsize=10)
        # 在绘制完一层后，将当前层的高度累加到bottom数组上，作为下一层的起始高度
        bottom += values

    # --- 3.2 绘制无边框的过渡区域 ---
    # 初始化一个新的起始高度数组，专门用于计算过渡区域的位置
    bottom_transition = np.zeros(len(parts))
    # 定义柱状图的宽度，必须与前面ax.bar中使用的值保持一致
    bar_width = 0.5
    # 再次循环遍历数据和颜色，以绘制过渡区域
    for (label, values), color in zip(data.items(), colors):
        # 调用之前定义的辅助函数，获取一个比原始颜色稍浅的颜色用于填充过渡区
        light_color = lighten_color(color, factor=1.15)
        # 开始一个内层循环，遍历所有柱子之间的间隙（从第0个到倒数第二个）
        for i in range(len(parts) - 1):
            # 定义过渡区域（梯形）的四个顶点的坐标
            v1 = (i + bar_width / 2, bottom_transition[i] + values[i])
            v2 = (i + 1 - bar_width / 2, bottom_transition[i + 1] + values[i + 1])
            v3 = (i + 1 - bar_width / 2, bottom_transition[i + 1])
            v4 = (i + bar_width / 2, bottom_transition[i])
            # 将四个顶点的x坐标和y坐标分别整理到列表中
            polygon_x = [v1[0], v2[0], v3[0], v4[0]]
            polygon_y = [v1[1], v2[1], v3[1], v4[1]]
            # 调用fill函数，根据顶点坐标填充一个多边形，不带任何边框，并应用透明度
            ax.fill(polygon_x, polygon_y, color=light_color, linewidth=0, alpha=alpha_value)
        # 更新过渡区域的起始高度数组，为下一层的计算做准备
        bottom_transition += values

    # --- 3.3 手动绘制所有水平分割线 ---
    # 初始化一个起始高度数组，用于计算水平分割线的位置
    bottom_for_lines = np.zeros(len(parts))
    # 循环遍历所有类别，但不包括最顶层（因为顶层上面没有分割线），所以用[:-1]切片
    for label in list(data.keys())[:-1]:
        # 获取当前类别的数据
        values = data[label]
        # 初始化两个空列表，用于存储将要绘制的分割线的x和y坐标点
        line_x, line_y = [], []
        # 循环遍历每个部分（每个柱子）
        for i in range(len(parts)):
            # 计算当前类别在当前部分的顶部Y坐标
            y_top = bottom_for_lines[i] + values[i]
            # 将当前柱子顶部的左、右两个点的x,y坐标依次添加到列表中
            line_x.extend([i - bar_width / 2, i + bar_width / 2])
            line_y.extend([y_top, y_top])
        # 使用plot函数将所有坐标点连接起来，绘制出一条连续的、跨越所有部分的白色分割线
        ax.plot(line_x, line_y, color="white", linewidth=1.5)
        # 更新分割线的起始高度数组，为绘制上一层的分割线做准备
        bottom_for_lines += values

    # --- 4. 坐标轴和图框样式调整 ---
    # 设置Y轴的显示范围从0到100
    ax.set_ylim(0, 100)
    # 设置Y轴的标题文本和字体大小
    ax.set_ylabel("Percentage (%)", fontsize=16)
    # 设置坐标轴刻度的样式
    ax.tick_params(axis="both", which="major", labelsize=12, length=8, width=2)
    # 隐藏顶部的坐标轴边框
    ax.spines["top"].set_visible(False)
    # 隐藏右侧的坐标轴边框
    ax.spines["right"].set_visible(False)
    # 设置左侧和底部坐标轴边框的线宽为2，使其加粗
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    # --- 5. 添加并精修图例 ---
    # 获取图表中所有已添加标签的元素的图例句柄（图形）和标签（文字）
    handles, labels = ax.get_legend_handles_labels()
    # 调用legend函数创建图例
    ax.legend(
        handles[::-1],
        labels[::-1],
        title="class",
        title_fontsize="14",
        fontsize="12",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=1.5,
        handleheight=1.2,
    )

    # --- 6. 自动调整布局并保存图表 ---
    # 自动调整图表布局，以防止标签、图例等元素显示不全或被裁切
    plt.tight_layout()
    # 【修改】构建每个图表的保存路径和文件名
    save_path = os.path.join(output_folder, f"chart_{palette_name}.png")
    # 【修改】保存图表到文件，而不是显示出来。dpi=300保证了较高的分辨率
    plt.savefig(save_path, dpi=300)
    # 【新增】关闭当前图形，释放内存，为绘制下一张图表做准备
    plt.close(fig)

# 在所有循环结束后，打印一条提示信息，告知用户所有操作已完成
print(f"\n任务完成！所有10张图表已保存至文件夹: '{output_folder}'")

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

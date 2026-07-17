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

import math
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "stix"
# 颜色库
color_schemes = {
    "1": ["#69b7a3", "#fde368", "#f69f98", "#99d2e7"],
    "2": ["#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"],
    "3": ["#fbb4ae", "#b3cde3", "#ccebc5", "#fed9a6"],
    "4": ["#8c564b", "#a9a9a9", "#d62728", "#bcbd22"],
    "5": ["#ccebc5", "#a8ddb5", "#7bccc4", "#43a2ca"],
    "6": ["#fee0d2", "#fc9272", "#ef3b2c", "#a50f15"],
    "7": ["#deebf7", "#9ecae1", "#4292c6", "#08519c"],
    "8": ["#e377c2", "#17becf", "#dbdb8d", "#7f7f7f"],
    "9": ["#efedf5", "#bcbddc", "#807dba", "#54278f"],
    "10": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
}
# 选择绘图方案
selected_scheme_name = "10"
colors = color_schemes[selected_scheme_name]
# 定义绘图所需要的数据、图名、图面的标注，可以根据自己的实际数据来添加或者是减少字典，像画几个就画几个
chart_data = [
    {  # 第一个图的数据
        "title": "MEM1 PESS",  # 图的标题
        "sizes": [52.59, 22.62, 19.36, 5.43],  # 各个扇区的百分比数据
        "center_text": "Demand:\n24592.4kW\nDemand response:\n5074.2kW",  # 显示在圆环中心的文本
    },
    {
        "title": "MEM2 PESS",
        "sizes": [50.92, 17.8, 16.89, 14.39],
        "center_text": "Demand:\n34618.8kW\nDemand response:\n7388.5kW",
    },
    {
        "title": "MEM3 PESS",
        "sizes": [30.27, 32.9, 29.67, 7.16],
        "center_text": "Demand:\n16928.9kW\nDemand response:\n4319.6kW",
    },
    {
        "title": "MEM1 SESS",
        "sizes": [52.51, 10.48, 35.07, 1.94],
        "center_text": "Demand:\n24592.4kW\nDemand response:\n2911.1kW",
    },
    {
        "title": "MEM2 SESS",
        "sizes": [49.76, 6.24, 31.28, 12.72],
        "center_text": "Demand:\n31618.8kW\nDemand response:\n4689.7kW",
    },
    {
        "title": "MEM3 SESS",
        "sizes": [28.59, 20.46, 45.74, 5.21],
        "center_text": "Demand:\n16928.9kW\nDemand response:\n2081.6kW",
    },
]
# 定义饼图各个扇区对应的标签文字
labels = ["Renewable energy", "Grid", "Storage", "Other devices"]
# 获取子图的总数量，即 chart_data 列表中的字典的数量
num_charts = len(chart_data)
# 设定每行的子图的数量
ncols = 3
# 使用 math.ceil() 函数向上取整，计算绘图需要的总行数
nrows = math.ceil(num_charts / ncols)
# 计算整个画布的高度
fig_height = 5 * nrows
# 创建一个大的画布figure和一组子图axes，指定行数、列数和整个画布的尺寸
fig, axes = plt.subplots(nrows, ncols, figsize=(16, fig_height))
# 如果只有一个图，axes 就不是数组，需要手动把它变成一个列表，以统一处理方式
if num_charts == 1:
    axes = [axes]  # 将单个 axes 对象放入列表中
else:
    axes = axes.flatten()  # 如果是多个图，将 axes 数组展平
# 初始化一个变量，用于存储最后一次绘制的饼图扇区对象，以便后续创建图例
wedges_for_legend = None
# 使用 enumerate 遍历展平后的 axes 数组，同时获取索引 i 和每个子图对象 ax
for i, ax in enumerate(axes):
    # 处理图表数量不是 ncols 的整数倍时多出来的空白子图
    if i >= num_charts:
        # 如果当前子图的索引超出了数据列表的范围（例如，创建了6个子图位置，但只有5个图的数据）
        ax.axis("off")  # 那么就将这个多余的子图的坐标轴和边框都隐藏掉
        continue  # 然后跳过本次循环的剩余部分，处理下一个子图
    data = chart_data[i]  # 根据当前索引 i，从 chart_data 列表中获取对应图表的数据
    # 在当前子图 (ax) 上绘制
    wedges, texts, autotexts = ax.pie(
        data["sizes"],  # 数值数据
        autopct="%1.2f%%",  # 设置扇区内百分比的显示格式，保留两位小数
        startangle=90,  # 设置饼图的起始角度，90度表示从顶部开始
        colors=colors,  # 使用我们之前选定的配色方案
        pctdistance=0.73,  # 设置百分比文本离圆心的距离
        wedgeprops={"edgecolor": "white", "linewidth": 1},  # 设置扇区属性，包括白色的描边和描边宽度
    )
    # 创建一个白色实心圆，用于覆盖饼图中心，从而形成圆环图效果
    center_circle = plt.Circle((0, 0), 0.50, fc="white")
    ax.add_artist(center_circle)  # 将这个白色圆添加到当前子图中
    wedges_for_legend = wedges  # 保存最后一次绘制的扇区对象，用于生成统一的图例
    # 遍历所有自动生成的百分比文本对象
    for autotext in autotexts:
        autotext.set_color("black")  # 将文本颜色设置为黑色
        autotext.set_fontsize(11)  # 设置文本字体大小
        autotext.set_fontweight("bold")  # 设置文本字体为粗体
    # 在圆环中心 (0,0) 添加文本
    ax.text(0, 0, data["center_text"], ha="center", va="center", fontsize=11)
    ax.set_title(
        data["title"], fontsize=16, pad=5
    )  # 设置当前子图的标题，并调整标题与图的间距 (pad)
# 动态计算图例的垂直位置，使其能适应不同行数的布局
legend_y_pos = 0.1 / nrows
# 在整个画布 (fig) 的底部中心位置添加一个统一的图例
fig.legend(
    wedges_for_legend,
    labels,  # 第一个参数是扇区对象，第二个是对应的标签文本
    loc="lower center",  # 设置图例位于底部中心
    bbox_to_anchor=(0.5, legend_y_pos),  # 精确控制图例位置，(0.5, y) 表示水平居中
    ncol=4,  # 设置图例分为4列显示
    fontsize=14,  # 设置图例的字体大小
    frameon=False,
)  # 不显示图例的边框
# 调整子图之间的间距，wspace为水平间距，hspace为垂直间距
plt.subplots_adjust(wspace=-0.8, hspace=-0.1)
# 自动调整整体布局，以防止标题、标签等重叠。rect参数用于在画布上留出边距，[左, 下, 右, 上]
plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # rect=[0, 0.1, 1, 0.95] 为底部图例留出空间
# 定义一个字符串变量，存储您希望保存图形的文件夹路径
output_folder = str(OUTPUT_DIR)
# 确保这个文件夹存在，如果不存在，os.makedirs 会自动创建它
os.makedirs(output_folder, exist_ok=True)
# 定义一个基础文件名，不包含文件扩展名
file_basename = "DZHXT"
# 使用 os.path.join 拼接文件夹路径和文件名，生成完整的文件保存路径
png_filepath = os.path.join(output_folder, f"{file_basename}{selected_scheme_name}.png")
pdf_filepath = os.path.join(output_folder, f"{file_basename}{selected_scheme_name}.pdf")
# 调用 savefig 函数保存图形
plt.savefig(png_filepath, dpi=300, bbox_inches="tight")
plt.savefig(pdf_filepath, bbox_inches="tight")
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

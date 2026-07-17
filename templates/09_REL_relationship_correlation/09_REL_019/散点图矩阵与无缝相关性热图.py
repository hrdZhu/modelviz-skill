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
# ======================================1.库的导入及全局设置===================================
# =========================================================================================
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# =========================================================================================
# ========================================2.数据的加载及预处理===================================
# =========================================================================================
file_path = str(DATA_DIR / "散点图矩阵/pairplot_data.xlsx")
df = pd.read_excel(file_path)
# 定义用于数据分箱的边界值
bins = [0, 3000, 4500, 6000, 7500, float("inf")]
# 定义每个分箱区间的标签
labels = ["3000", "4500", "6000", "7500", "9000"]
# 使用 pandas 的 cut 函数将 'Yield' 列的数据根据定义的 bins 和 labels 进行离散化，并将结果存入新列 'yield'
df["yield"] = pd.cut(df["Yield"], bins=bins, labels=labels, right=False)
# 定义要在散点图矩阵中分析的变量列表
variables = ["Yield", "TAGP", "GPP", "ET"]
# 计算需要的颜色数量，即 'yield' 列离散化后的类别数量
n_colors_needed = len(labels)
# =========================================================================================
# ======================================3.配色方案的设置及选择===================================
# =========================================================================================
color_schemes = {
    "1": {
        "categorical": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "heatmap": "YlGn",
    },
    "2": {"categorical": sns.color_palette("colorblind", 10), "heatmap": "cividis"},
    "3": {"categorical": sns.color_palette("deep", 10), "heatmap": "plasma"},
    "4": {"categorical": sns.color_palette("muted", 10), "heatmap": "magma"},
    "5": {"categorical": sns.color_palette("pastel", 10), "heatmap": "summer"},
    "6": {"categorical": sns.color_palette("bright", 10), "heatmap": "viridis"},
    "7": {"categorical": sns.color_palette("dark", 10), "heatmap": "inferno"},
    "8": {"categorical": sns.color_palette("Set2", 8), "heatmap": "Greens"},
    "9": {"categorical": sns.color_palette("Paired", 10), "heatmap": "coolwarm"},
    "10": {"categorical": sns.color_palette("tab10", 10), "heatmap": "RdYlGn"},
}
# 选择一个配色方案的名称
selected_scheme_name = "10"
# 根据名称从字典中获取对应的配色方案
selected_scheme = color_schemes[selected_scheme_name]
# 从选定的分类色板中获取所需数量的颜色
selected_palette = selected_scheme["categorical"][:n_colors_needed]
# 获取选定的热力图颜色映射（cmap）对象
selected_cmap = plt.get_cmap(selected_scheme["heatmap"])


# =========================================================================================
# ======================================4.热图绘制函数=========================================
# =========================================================================================
def calculated_corr_heatmap(x, y, g=None, cmap=None, **kwargs):
    ax = plt.gca()  # 获取当前的坐标轴对象
    if hasattr(ax, "_plotted"):  # 如果该坐标轴已经绘制过，则直接返回，避免重复绘制
        return
    ax.set_axis_off()  # 关闭坐标轴的刻度和边框
    expand_horizontal = 0.2  # 定义矩形在水平方向上的扩展比例
    expand_vertical = 0.1  # 定义矩形在垂直方向上的扩展比例
    rect_width = 1 + expand_horizontal  # 计算矩形的宽度（1代表整个坐标轴宽度），并加上扩展量
    rect_height = 1 + expand_vertical  # 计算矩形的高度（1代表整个坐标轴高度），并加上扩展量
    start_x = 0 - expand_horizontal / 2  # 计算矩形左下角的 x 坐标，向左移动半个扩展量
    start_y = 0 - expand_vertical / 2  # 计算矩形左下角的 y 坐标，向下移动半个扩展量
    corr = x.corr(y)  # 计算 x 和 y 两个变量之间的皮尔逊相关系数
    # 创建一个颜色归一化对象，将相关系数的值映射到 [-1, 1.0] 的范围
    norm = mcolors.Normalize(vmin=-1, vmax=1.0)
    # 使用统一的尺寸和位置来绘制矩形，填充颜色根据相关系数决定
    rect = Rectangle(
        (start_x, start_y),
        rect_width,
        rect_height,
        facecolor=cmap(norm(corr)),
        transform=ax.transAxes,
        zorder=-1,
        clip_on=False,
    )
    # 将创建的矩形添加到坐标轴上
    ax.add_patch(rect)
    # 根据相关系数的大小和色板类型决定文本颜色
    text_color = (
        "white"
        if abs(corr) > 0.55 and (cmap.name not in ["summer", "YlGn", "cividis"])
        else "black"
    )
    # 在矩形中心位置显示相关系数的数值，保留两位小数
    ax.text(
        0.5,
        0.5,
        f"{corr:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=16,
        color=text_color,
    )
    # 给坐标轴对象添加一个 '_plotted' 属性，标记为已绘制
    ax._plotted = True


# =========================================================================================
# ======================================5.散点图主题绘制========================================
# =========================================================================================
# 创建 PairGrid 对象，用于构建散点图矩阵的网格
g = sns.PairGrid(df, vars=variables, hue="yield", palette=selected_palette)
# 定义散点的大小
scatter_size = 35
# 将散点图函数 sns.scatterplot 映射到矩阵的下三角部分
g.map_lower(sns.scatterplot, alpha=0.7, s=scatter_size)
# 将核密度估计图函数 sns.kdeplot 映射到矩阵的对角线部分
g.map_diag(sns.kdeplot, fill=True)
# 将自定义的热力图函数 calculated_corr_heatmap 映射到矩阵的上三角部分
g.map_upper(calculated_corr_heatmap, g=g, cmap=selected_cmap)
# 获取变量的数量
num_vars = len(variables)
# =========================================================================================
# ======================================6.图面细节调整===================================
# =========================================================================================
# 遍历 PairGrid 中的每一个子图
for i, ax_row in enumerate(g.axes):
    for j, ax in enumerate(ax_row):
        # 默认隐藏所有子图的 x 和 y 轴标签
        ax.tick_params(labelleft=False, labelbottom=False)
        # 如果子图位于第一列，则显示 y 轴标签
        if j == 0:
            ax.tick_params(axis="y", labelleft=True)
        # 如果子图位于最后一行，则显示 x 轴标签，并不进行旋转
        if i == num_vars - 1:
            ax.tick_params(axis="x", labelbottom=True, rotation=0)
# 为图表添加图例
g.add_legend(title="yield")
# 获取图例对象
legend = g._legend
# 获取右下角的子图坐标轴，用于定位图例
ax_legend = g.axes[-1, -1]
# 将图例放置在右下角子图的右上角位置
legend.set_bbox_to_anchor((1.02, 0.95), transform=ax_legend.transAxes)
# 设置图例的具体位置为 'upper right'
legend.set_loc("upper right")
# 去掉图例的边框
legend.set_frame_on(False)
# 设置图例中文本的字体大小
plt.setp(legend.get_texts(), fontsize=10)
# 设置图例中标题的字体大小
plt.setp(legend.get_title(), fontsize=10)
# 计算图例中标记点的大小
legend_marker_size = np.sqrt(scatter_size)
# 遍历图例中的所有句柄（handles）
for handle in legend.legend_handles:
    # 判断句柄是否为 Line2D 对象（散点图的图例项）
    if isinstance(handle, plt.Line2D):
        # 设置标记点的边缘颜色与其填充颜色相同
        handle.set_markeredgecolor(handle.get_markerfacecolor())
        # 设置标记点的大小
        handle.set_markersize(legend_marker_size)
# =========================================================================================
# ======================================7.颜色条设置及结果保存===================================
# =========================================================================================
# 调整子图布局，为右侧的颜色条留出空间
g.fig.subplots_adjust(right=0.85)
# 在图表上创建一个新的坐标轴，用于放置颜色条
cbar_ax = g.fig.add_axes([0.88, 0.25, 0.02, 0.5])
# 创建一个与热力图一致的颜色归一化对象
norm = mcolors.Normalize(vmin=-0.2, vmax=1.0)
# 创建一个 ScalarMappable 对象，它将数据值映射到颜色
sm = ScalarMappable(norm=norm, cmap=selected_cmap)
# 在指定的坐标轴上绘制颜色条
cbar = g.fig.colorbar(sm, cax=cbar_ax)
# 设置颜色条上的刻度位置
cbar.set_ticks(np.arange(-0.2, 1.1, 0.2))
cbar.outline.set_visible(False)  # 隐藏颜色条的外框
cbar.ax.tick_params(width=0)  # 隐藏颜色条刻度线
cbar.ax.tick_params(labelsize=10)  # 设置颜色条刻度标签的字体大小
plt.savefig(str(OUTPUT_DIR / f"pairplot_{selected_scheme_name}.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(OUTPUT_DIR / f"pairplot_{selected_scheme_name}.pdf"), bbox_inches="tight")
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

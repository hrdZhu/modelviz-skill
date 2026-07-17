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

import matplotlib  # 导入matplotlib库本身
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于创建图表，通常简写为plt
import numpy as np  # 导入numpy库，用于进行高效的数值计算，特别是数组操作
import pandas as pd  # 导入pandas库，用于数据处理和分析，特别是DataFrame对象
import seaborn as sns  # 导入seaborn库，这是一个基于matplotlib的高级可视化库，用于美化图表
from scipy.stats import (  # 从scipy库的stats模块中导入高斯核密度估计和偏态分布函数
    gaussian_kde,
    skewnorm,
)

matplotlib.use(
    "Agg"
)  # 设置matplotlib的后端，'TkAgg'是一个常用的图形界面后端，有助于在某些环境中正确显示图表窗口
# 定义Excel文件路径 (请确保这个路径是正确的)
file_path = str(DATA_DIR / "simulated_mpl_data.xlsx")
# 定义图表中从上到下的类别顺序
cat_order = ["Polar", "Cold", "Arid", "Temperate", "Tropical"]
# 读取Excel文件到DataFrame
df = pd.read_excel(file_path)
print(f"成功从 '{file_path}' 读取数据。")
# 确保'Category'列是带有正确顺序的类别类型，这对绘图顺序至关重要
df["Category"] = pd.Categorical(df["Category"], categories=cat_order, ordered=True)
# =================================================================================
# 在这里定义了10个不同的颜色字典，每个字典代表一种配色方案。
# 您可以方便地在下方选择要使用的方案。
# =================================================================================
color_schemes = {
    "方案1": {
        "Polar": "#8e6d9d",
        "Cold": "#0072b2",
        "Arid": "#e69f00",
        "Temperate": "#77ab59",
        "Tropical": "#d95f5f",
    },
    "方案2": {
        "Polar": "#a2cffe",
        "Cold": "#66b3ff",
        "Arid": "#ffb366",
        "Temperate": "#99ff99",
        "Tropical": "#ff9999",
    },
    "方案3": {
        "Polar": "#E6A0C4",
        "Cold": "#C6D8D3",
        "Arid": "#F3D4A0",
        "Temperate": "#A8D5BA",
        "Tropical": "#F5B99A",
    },
    "方案4": {
        "Polar": "#3B5998",
        "Cold": "#8B9DC3",
        "Arid": "#F7C873",
        "Temperate": "#E69138",
        "Tropical": "#DD4B39",
    },
    "方案5": {
        "Polar": "#b2b2b2",
        "Cold": "#6a8d92",
        "Arid": "#c9a66b",
        "Temperate": "#8c9d82",
        "Tropical": "#b77b72",
    },
    "方案6": {
        "Polar": "#9b59b6",
        "Cold": "#3498db",
        "Arid": "#f1c40f",
        "Temperate": "#2ecc71",
        "Tropical": "#e74c3c",
    },
    "方案7": {
        "Polar": "#0d47a1",
        "Cold": "#1976d2",
        "Arid": "#42a5f5",
        "Temperate": "#90caf9",
        "Tropical": "#e3f2fd",
    },
    "方案8": {
        "Polar": "#5D4037",
        "Cold": "#8D6E63",
        "Arid": "#D4A76A",
        "Temperate": "#A1887F",
        "Tropical": "#BF360C",
    },
    "方案9": {
        "Polar": "#c7ddef",
        "Cold": "#a1c4e9",
        "Arid": "#fadfad",
        "Temperate": "#c1e1c1",
        "Tropical": "#f7caca",
    },
    "方案10": {
        "Polar": "#f026ff",
        "Cold": "#00f6ff",
        "Arid": "#ffea00",
        "Temperate": "#5bff57",
        "Tropical": "#ff479c",
    },
}

# --- 在这里选择您想使用的配色方案 ---
# 只需将下面的方案名称替换为 '方案1：原始经典' 到 '方案10：霓虹科技' 中的任意一个即可
selected_scheme_name = "方案10"
categories = color_schemes[selected_scheme_name]
print(f"已选择配色方案: {selected_scheme_name}")
sns.set_theme(
    style="white", rc={"axes.facecolor": (0, 0, 0, 0)}
)  # 设置seaborn的主题样式，背景为白色，坐标轴区域透明
plt.rcParams["font.sans-serif"] = ["Times New Roman"]  # 设置字体参数，以支持Times New Roman字体
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示负号
fig, ax = plt.subplots(
    figsize=(8, 5), dpi=120
)  # 创建一个图窗(fig)和一组子图(ax)，并设置尺寸和分辨率
overlap = 1  # 定义山脊图之间垂直方向的重叠程度
num_cats = len(cat_order)  # 获取类别的总数
for i, category in enumerate(cat_order):  # 遍历每个类别及其索引，用于逐一绘制
    y_base = (num_cats - 1 - i) * overlap  # 计算当前类别图表的垂直基线位置，实现从上到下排列
    data_cat = df[(df["Category"] == category)]["MPL"]  # 从总数据中筛选出当前类别的数据
    color = categories[(category)]  # 获取当前类别对应的颜色
    kde = gaussian_kde(data_cat)  # 使用高斯核密度估计计算当前类别数据的概率密度函数
    x_range = np.linspace(
        df["MPL"].min(), df["MPL"].max(), 500
    )  # 创建一个x轴的坐标点范围，用于绘制平滑的密度曲线
    y_kde = kde(x_range)  # 计算在x_range上每个点的概率密度值
    y_kde_scaled = y_kde * 1.8  # 对密度曲线的高度进行缩放，使其在视觉上更明显
    ax.fill_between(
        x_range, y_base, y_kde_scaled + y_base, color=color, alpha=0.7, zorder=i * 3 + 1
    )  # 填充密度曲线下方的区域，形成山脊效果
    ax.plot(
        x_range, y_kde_scaled + y_base, color=color, lw=0.7, zorder=i * 3 + 2
    )  # 绘制密度曲线的轮廓线
    mean_val = data_cat.mean()  # 计算当前类别数据的均值
    median_val = data_cat.median()  # 计算当前类别数据的中位数
    mean_y_on_curve = kde(mean_val)[0] * 1.8  # 计算均值在缩放后的密度曲线上的高度
    median_y_on_curve = kde(median_val)[0] * 1.8  # 计算中位数在缩放后的密度曲线上的高度
    ax.vlines(
        median_val,
        y_base,
        y_base + median_y_on_curve,
        color=color,
        lw=1.2,
        ls="-",
        zorder=i * 3 + 3,
    )  # 在中位数位置绘制一条垂直线，颜色与填充色相同
    label_y_offset = 0.35  # 定义文本标签的垂直偏移量
    if category == "Polar":  # 如果类别是'Polar'，使用特殊的文本格式
        ax.text(
            -2.5, y_base + 0.1, f"{mean_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加均值标签
        ax.text(
            3, y_base + 0.15, f"{median_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加中位数标签
        ax.text(-2.5, y_base + 0.5, f"Mean:", ha="center", va="center", fontsize=9)  # 添加均值标签
        ax.text(3, y_base + 0.5, f"Median:", ha="center", va="center", fontsize=9)  # 添加中位数标签
    elif category == "Cold":  # 如果类别是'Cold'
        ax.text(
            -2.5, y_base + 0.1, f"{mean_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加均值标签
        ax.text(
            3, y_base + 0.2, f"{median_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加中位数标签
    elif category == "Arid":  # 如果类别是'Arid'
        ax.text(
            -2.5, y_base + 0.25, f"{mean_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加均值标签
        ax.text(
            3, y_base + 0.3, f"{median_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加中位数标签
    elif category == "Temperate":  # 如果类别是'Temperate'
        ax.text(
            -2.5, y_base + 0.1, f"{mean_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加均值标签
        ax.text(
            3, y_base + 0.4, f"{median_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加中位数标签
    elif category == "Tropical":  # 如果类别是'Tropical'
        ax.text(
            -2.5, y_base + 0.1, f"{mean_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加均值标签
        ax.text(
            3, y_base + 0.32, f"{median_val:.2f}", ha="center", va="center", fontsize=9
        )  # 添加中位数标签
ax.axvline(
    0, color="grey", linestyle="--", lw=1, zorder=0
)  # 在x=0的位置绘制一条全局的灰色垂直虚线作为参考
ax.set_xlabel("MPL", fontsize=12)  # 设置x轴的标签
ax.set_xlim(-4, 6.5)  # 设置x轴的显示范围
y_tick_positions = [
    (num_cats - 1 - i) * overlap for i in range(num_cats)
]  # 计算y轴上每个类别标签的精确位置
ax.set_yticks(y_tick_positions)  # 设置y轴的刻度位置
ax.set_yticklabels(cat_order, fontsize=11, ha="right")  # 设置y轴的刻度标签，并使其右对齐
ax.tick_params(axis="y", length=0)  # 自定义y轴刻度样式，这里将刻度线的长度设置为0，即隐藏刻度线
ax.spines["right"].set_visible(False)  # 隐藏右侧的轴线
ax.spines["top"].set_visible(False)  # 隐藏顶部的轴线
ax.spines["bottom"].set_color("black")  # 设置底部轴线的颜色为黑色
ax.spines["bottom"].set_linewidth(1)  # 设置底部轴线的宽度
ax.grid(False)  # 禁用整个图表的网格线
ax.yaxis.grid(False)  # 再次确认禁用y轴的网格线
plt.tight_layout(pad=1.5)  # 自动调整子图参数，使之填充整个图像区域，防止标签被裁切
plt.savefig(str(OUTPUT_DIR / "10.png"), dpi=300)
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

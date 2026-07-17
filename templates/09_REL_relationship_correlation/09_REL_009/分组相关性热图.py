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

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "stix"
# 颜色库
color_schemes = {
    "1": sns.diverging_palette(240, 10, s=85, l=65, center="light", as_cmap=True),
    "2": sns.diverging_palette(275, 130, s=85, l=65, center="light", as_cmap=True),
    "3": sns.diverging_palette(180, 35, s=85, l=65, center="light", as_cmap=True),
    "4": sns.diverging_palette(320, 145, s=85, l=65, center="light", as_cmap=True),
    "5": sns.diverging_palette(250, 45, s=85, l=65, center="light", as_cmap=True),
    "6": "coolwarm",
    "7": "bwr",
    "8": "RdYlBu_r",
    "9": "viridis",
    "10": "icefire",
}
selected_scheme_name = "9"  # 选择要使用的配色
excel_path = str(DATA_DIR / "分组data.xlsx")  # 定义输入文件的路径
# 读取名为 'Microbe_Data' 的工作表，并将第一列作为行索引
microbe_abundance_df = pd.read_excel(excel_path, sheet_name="Microbe_Data", index_col=0)
# 读取名为 'Env_Factors_Data' 的工作表，并将第一列作为行索引
env_factors_df = pd.read_excel(excel_path, sheet_name="Env_Factors_Data", index_col=0)
# 获取Microbe_Data数据的所有列名，并转换为列表
microbe_labels = microbe_abundance_df.columns.tolist()
# 获取Env_Factors_Data数据的所有列名，并转换为列表
env_labels = env_factors_df.columns.tolist()
# 创建一个空的DataFrame，用于相关性系数值
corr_matrix = pd.DataFrame(index=microbe_labels, columns=env_labels)
# 创建一个空的DataFrame，用于存储相关性分析的p值
p_value_matrix = pd.DataFrame(index=microbe_labels, columns=env_labels)
# 找出两个数据共有的样本
common_samples = microbe_abundance_df.index.intersection(env_factors_df.index)
# 使用共有的样本索引来筛选数据，确保样本一一对应
microbe_abundance_df = microbe_abundance_df.loc[common_samples]
env_factors_df = env_factors_df.loc[common_samples]
# 开始双重循环，计算相关性
for i in microbe_labels:
    for j in env_labels:
        # 使用 pearsonr 函数计算皮尔逊相关系数和p值
        corr_coeff, p_value = pearsonr(microbe_abundance_df[i], env_factors_df[j])
        corr_matrix.loc[i, j] = corr_coeff  # 将计算出的相关系数存入相关性矩阵
        p_value_matrix.loc[i, j] = p_value  # 将计算出的p值存入p值矩阵
# 将相关性矩阵的数据类型转换为浮点数
corr_matrix = corr_matrix.astype(float)
# 将p值矩阵的数据类型转换为浮点数
p_value_matrix = p_value_matrix.astype(float)
# 定义p值的判断条件，用于标记显著性水平
conditions = [p_value_matrix < 0.001, p_value_matrix < 0.01, p_value_matrix < 0.05]
# 定义与上述条件对应的显著性标记
choices = ["***", "**", "*"]
# 使用 np.select 根据p值选择对应的标记，不满足任何条件的默认为空字符串''
annot_data = np.select(conditions, choices, default="")
# 创建一个新的图形和坐标轴，设置图形的尺寸为12x15英寸
fig, ax = plt.subplots(figsize=(12, 15))
# 使用 seaborn 绘制热图
sns.heatmap(
    corr_matrix,  # 数据为计算出的相关性矩阵
    ax=ax,  # 在指定的坐标轴上绘制
    cmap=color_schemes[selected_scheme_name],  # 设置颜色映射方案
    annot=annot_data,  # 显著性星号
    fmt="",  # 由于注释数据已经是字符串，所以格式化字符串为空
    annot_kws={
        "size": 20,
        "color": "black",
        "ha": "center",
        "va": "center",
    },  # 设置注释文本的属性（大小、颜色、水平和垂直对齐方式）
    linewidths=2,  # 设置单元格之间的网格线宽度
    linecolor="white",  # 设置网格线的颜色
    vmin=-1,
    vmax=1,  # 设置颜色条的范围，从-1到1
    square=True,  # 将每个单元格设置为正方形
    cbar=True,  # 显示颜色条
    cbar_kws={  # 设置颜色条的属性
        "orientation": "horizontal",  # 方向为水平
        "location": "top",  # 位置在图的顶部
        "shrink": 0.68,  # 缩小颜色条的长度为其原始尺寸的68%
        "aspect": 25,  # 设置颜色条的长宽比
        "pad": 0.01,  # 设置颜色条与热图之间的间距
    },
)
# 获取热图的颜色条对象
cbar = ax.collections[0].colorbar
# 设置颜色条刻度标签的字体大小为12，并移除刻度线
cbar.ax.tick_params(labelsize=12, length=0)
# 将Y轴的刻度标签显示在图表的右侧
ax.yaxis.tick_right()
# 设置X轴刻度的属性：旋转90度，字体大小14，移除刻度线
ax.tick_params(axis="x", rotation=90, labelsize=14, length=0)
# 设置Y轴刻度的属性：不旋转，字体大小14，移除刻度线
ax.tick_params(axis="y", rotation=0, labelsize=14, length=0)
# 将X轴的标题设置为空
ax.set_xlabel("")
# 将Y轴的标题设置为空
ax.set_ylabel("")
# 定义一个字典，用于存储Y轴上不同分组的名称及其在Y轴上的起始和结束索引
groups = {"S-strategies": (0, 7), "Y-strategies": (7, 12), "A-strategies": (12, 19)}
# 设置Y轴的范围，使其从上到下为0到总数，这样热图的(0,0)点在左上角
ax.set_ylim(len(microbe_labels), 0)
# 定义分组标签在X轴方向上的位置
label_x_pos = -0.8
# 循环遍历每个分组，在图表左侧添加灰色矩形背景和分组标签
for label, (start, end) in groups.items():
    # 创建一个矩形对象来标识分组区域
    rect = patches.Rectangle(
        (label_x_pos - 0.5, start),
        1,
        end - start,
        linewidth=1,
        edgecolor="white",
        facecolor="#cccccc",
        clip_on=False,
    )
    ax.add_patch(rect)  # 将矩形添加到图表中
    # 在矩形中央添加分组的文本标签
    ax.text(
        label_x_pos,
        (start + end) / 2,
        label,
        ha="center",
        va="center",
        rotation=90,
        fontsize=16,
        color="black",
        clip_on=False,
    )
# 在Y轴位置为7的地方画一条白色的粗横线，用于分隔不同的组
ax.axhline(y=7, color="white", linewidth=5)
# 在Y轴位置为12的地方画一条白色的粗横线，用于分隔不同的组
ax.axhline(y=12, color="white", linewidth=5)
# 自动调整子图布局以适应图形元素，并设置边界框来微调边距
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
plt.savefig(str(OUTPUT_DIR / f"分组相关性{selected_scheme_name}.png"), dpi=300, bbox_inches="tight")
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

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

# -*- coding: utf-8 -*-
import matplotlib  # 导入 matplotlib 库
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图模块，并简写为 plt
import numpy as np  # 导入 numpy科学计算库，并简写为 np
import pandas as pd  # 导入 pandas 数据处理库，并简写为 pd
import seaborn as sns  # 导入 seaborn 绘图库，并简写为 sns
from matplotlib.patches import (
    FancyBboxPatch,  # 从 matplotlib.patches 模块导入 FancyBboxPatch 类，用于绘制圆角矩形
    Rectangle,  # 从 matplotlib.patches 模块导入 Rectangle 类，用于绘制矩形
)
from scipy import stats  # 从 scipy库中导入 stats 模块，用于统计分析
from statsmodels.stats.multitest import (
    multipletests,  # 从 statsmodels 库中导入 multipletests 函数，用于多重检验校正
)

matplotlib.use("Agg")  # 设置 matplotlib 使用的后端为 TkAgg，这有助于在某些环境下正确显示图形窗口
plt.rcParams["font.family"] = "Times New Roman"  # 设置 matplotlib 的默认字体族为 'Times New Roman'
# 数据准备和标签定义 (修改部分：从文件加载真实数据)
# -------------------- START OF MODIFIED SECTION --------------------
# 文件格式要求：行是样本，列是变量(特征)，第一行应为变量名(如 "Albumin", "Hgb" 等)。
data_df = pd.read_excel(str(DATA_DIR / "correlation_data.xlsx"))
print(
    f"成功加载数据文件 'your_data_file.csv'。 数据维度: {data_df.shape[0]} 个样本, {data_df.shape[1]} 个变量。"
)
# 从加载的数据中自动获取标签和原始数据
labels = data_df.columns.tolist()  # 从文件的列名自动获取标签
raw_data = data_df.values  # 将DataFrame转换为Numpy数组以供后续计算
num_vars = len(labels)  # 变量（特征）的数量
# -------------------- END OF MODIFIED SECTION --------------------
corr_matrix = np.ones((num_vars, num_vars))  # 初始化相关系数矩阵，所有元素为1
pval_matrix_raw = np.ones((num_vars, num_vars))  # 初始化原始P值矩阵，所有元素为1
p_values_list_for_fdr = []  # 初始化一个列表，用于存储进行FDR校正的P值
for i in range(num_vars):  # 外层循环遍历每个变量
    pval_matrix_raw[i, i] = 0.0  # 对角线上的P值设为0 (自身与自身的相关性P值为0)
    for j in range(i + 1, num_vars):  # 内层循环遍历当前变量之后的其他变量 (避免重复计算和自身比较)
        r, p = stats.pearsonr(
            raw_data[:, i], raw_data[:, j]
        )  # 计算变量i和变量j之间的皮尔逊相关系数r和P值p
        corr_matrix[i, j] = r
        corr_matrix[j, i] = r  # 将相关系数存入相关系数矩阵 (对称矩阵)
        pval_matrix_raw[i, j] = p
        pval_matrix_raw[j, i] = p  # 将原始P值存入P值矩阵 (对称矩阵)
        if p is not np.nan:
            p_values_list_for_fdr.append(p)  # 如果P值不是NaN，则添加到FDR校正列表中
pvals_corrected_flat = []  # 初始化一个列表，用于存储校正后的P值 (扁平化)
if p_values_list_for_fdr:  # 如果有P值需要校正
    reject, pvals_corrected_flat, _, _ = multipletests(
        p_values_list_for_fdr, alpha=0.05, method="fdr_bh"
    )  # 使用'fdr_bh'方法进行多重检验校正
else:  # 如果没有P值需要校正
    pvals_corrected_flat = []  # 校正后的P值列表为空
pval_matrix_corrected = np.ones_like(
    pval_matrix_raw
)  # 初始化校正后的P值矩阵，结构同原始P值矩阵，元素为1
k = 0  # 初始化一个计数器，用于从扁平化的校正P值列表中取值
for i in range(num_vars):  # 外层循环遍历每个变量
    pval_matrix_corrected[i, i] = 0.0  # 校正后P值矩阵的对角线元素设为0
    for j in range(i + 1, num_vars):  # 内层循环遍历当前变量之后的其他变量
        if k < len(pvals_corrected_flat):  # 如果计数器未超出校正P值列表的长度
            pval_matrix_corrected[i, j] = pvals_corrected_flat[k]  # 从列表中取校正后的P值
            pval_matrix_corrected[j, i] = pvals_corrected_flat[k]  # 存入校正P值矩阵 (对称)
            k += 1  # 计数器加1
        else:  # 如果校正P值列表已用完 (理论上不应发生，除非原始P值列表为空)
            pval_matrix_corrected[i, j] = pval_matrix_raw[i, j]  # 使用原始P值填充
            pval_matrix_corrected[j, i] = pval_matrix_raw[j, i]  # (对称)
pval_stars = np.empty((num_vars, num_vars), dtype=object)
pval_stars.fill("")  # 初始化一个对象类型的Numpy数组用于存储P值显著性星号，并用空字符串填充
for r_idx in range(num_vars):  # 遍历校正P值矩阵的行
    for c_idx in range(num_vars):  # 遍历校正P值矩阵的列
        if r_idx == c_idx:
            continue  # 如果是对角线元素，则跳过
        p_corr = pval_matrix_corrected[r_idx, c_idx]  # 获取校正后的P值
        if p_corr < 0.001:
            pval_stars[r_idx, c_idx] = "***"  # P值小于0.001，标记为三个星号
        elif p_corr < 0.01:
            pval_stars[r_idx, c_idx] = "**"  # P值小于0.01，标记为两个星号
        elif p_corr < 0.05:
            pval_stars[r_idx, c_idx] = "*"  # P值小于0.05，标记为一个星号
        else:
            pval_stars[r_idx, c_idx] = ""  # P值不显著，标记为空字符串
annot_texts = np.empty((num_vars, num_vars), dtype=object)
annot_texts.fill("")  # 初始化一个对象类型的Numpy数组用于存储热图单元格的注释文本
for i in range(num_vars):  # 遍历行
    for j in range(num_vars):  # 遍历列
        if i > j:  # 只处理下三角部分 (因为热图只显示下三角)
            val = corr_matrix[i, j]
            stars = pval_stars[i, j]  # 获取相关系数和对应的星号
            if val == 0.0:
                val_str = "0"  # 如果相关系数为0，格式化为"0"
            elif abs(round(val * 10) - (val * 10)) < 1e-5 and abs(val) < 1:
                val_str = f"{val:.1f}"  # 如果相关系数乘以10后与四舍五入到一位小数的结果非常接近 (处理像0.20这样的情况)，且绝对值小于1，则格式化为一位小数
            else:
                val_str = f"{val:.2f}"  # 其他情况，格式化为两位小数
            annot_texts[i, j] = (
                f"{val_str}\n{stars}"  # 组合相关系数值和星号作为注释文本，用换行符分隔
            )
corr_df = pd.DataFrame(
    corr_matrix, index=labels, columns=labels
)  # 将相关系数矩阵转换为Pandas DataFrame，并设置行列标签
annot_df = pd.DataFrame(
    annot_texts, index=labels, columns=labels
)  # 将注释文本矩阵转换为Pandas DataFrame
mask = np.zeros_like(
    corr_df, dtype=bool
)  # 创建一个与corr_df形状相同，类型为布尔值的掩码矩阵，并用False填充
mask[np.triu_indices_from(mask, k=0)] = (
    True  # 将掩码矩阵的上三角部分（包括对角线，k=0）设置为True，用于隐藏热图的这部分
)
# 开始绘图
fig, ax = plt.subplots(
    figsize=(10, 10)
)  # 创建一个新的图形(fig)和一组子图(ax)，设置图形大小为10x10英寸
fig.patch.set_facecolor("white")
ax.set_facecolor("white")  # 设置图形和子图的背景颜色为白色
plt.subplots_adjust(
    left=0.15, right=0.80, top=0.93, bottom=0.07
)  # 调整子图在图形中的布局位置和间距
current_cmap = plt.get_cmap(
    "RdBu_r"
).copy()  # 获取名为 'RdBu_r' 的颜色映射方案副本 (红-白-蓝，r表示反转)
current_cmap.set_bad(color="white")  # 设置颜色映射中无效值（例如NaN）的颜色为白色
sns.heatmap(
    corr_df,
    mask=mask,
    annot=annot_df,
    fmt="s",
    cmap=current_cmap,  # 使用seaborn绘制热图
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0,  # 设置颜色映射的最小值、最大值、中心值，确保单元格为方形，单元格间线条宽度为0
    cbar=False,
    ax=ax,
    annot_kws={"fontsize": 8, "fontweight": "normal", "va": "center"},
)  # 不自动生成颜色条，在指定的ax上绘制，设置注释文本的字体大小、字重和垂直对齐方式
# 手动为下三角数据单元格添加边框
for r_idx in range(num_vars):  # 遍历行索引
    for c_idx in range(num_vars):  # 遍历列索引
        if not mask[r_idx, c_idx]:  # 如果当前单元格没有被掩码隐藏 (即属于下三角部分)
            ax.add_patch(
                Rectangle((c_idx, r_idx), 1, 1, fill=False, edgecolor="lightgray", lw=0.5)
            )  # 在该单元格位置添加一个无填充、浅灰色边框、线宽为0.5的矩形
# 调整坐标轴标签和刻度
ax.set_xticks([])
ax.set_xticklabels([])
ax.xaxis.tick_top()  # 清空X轴刻度及标签，并将X轴刻度线置于顶部 (虽然此处无刻度)
y_tick_positions = np.arange(1, num_vars) + 0.5  # 计算Y轴刻度的位置 (对应于热图单元格的中心)
y_tick_labels = labels[1:]  # 获取Y轴的标签文本 (不包括第一个标签，因为第一行被隐藏)
y_axis_label_fontsize = 9  # 设置Y轴标签的字体大小
ax.set_yticks(y_tick_positions)  # 设置Y轴的刻度位置
ax.set_yticklabels(
    y_tick_labels, rotation=0, va="center", ha="right", fontsize=y_axis_label_fontsize
)  # 设置Y轴的刻度标签，旋转角度为0，垂直居中对齐，水平右对齐
ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")
ax.tick_params(axis="y", length=0)  # 将Y轴刻度线置于左侧，标签位置也置于左侧，Y轴刻度线长度设为0
ax.set_xlabel("")
ax.set_ylabel("")  # 清空X轴和Y轴的轴标签文本
ax.set_xlim(0, num_vars)
ax.set_ylim(num_vars, 0)  # 设置X轴和Y轴的显示范围，Y轴反向 (顶部为0，底部为num_vars)
# 在对角线单元格添加文字标注
diagonal_label_fontsize = y_axis_label_fontsize  # 对角线标签的字体大小与Y轴标签一致
for i in range(
    num_vars - 1 if num_vars > 0 else 0
):  # 循环遍历变量索引 (不包括最后一个，以去掉最后一个对角线标签)
    ax.text(
        i + 0.5,
        i + 0.5,
        labels[i],  # 在对角线单元格中心添加文本
        ha="center",
        va="center",
        fontsize=diagonal_label_fontsize,  # 水平居中、垂直居中对齐
        rotation=0,
        color="black",
    )  # 文本不旋转，颜色为黑色
# 添加左上角的 "a" 标签
# fig.text(0.13, 0.76, "a", fontsize=18, fontweight='bold', va='center_baseline', ha='left', transform=fig.transFigure) # 在图形的指定位置( фигура坐标系)添加文本"a"
# 手动创建和控制颜色条
cbar_left = 0.73  # 颜色条左下角x坐标 (图形坐标系，用户设定值)
cbar_bottom = 0.28  # 颜色条左下角y坐标 (图形坐标系，用户设定值)
cbar_width = 0.02  # 颜色条宽度 (图形坐标系，用户设定值)
cbar_height = 0.5  # 颜色条高度 (图形坐标系，用户设定值)
cax_colorbar = fig.add_axes(
    [cbar_left, cbar_bottom, cbar_width, cbar_height]
)  # 在图形中为颜色条创建一个新的Axes对象
mappable = ax.collections[0]  # 获取热图对象中用于颜色映射的集合 (通常是第一个)
cbar = fig.colorbar(
    mappable, cax=cax_colorbar, orientation="vertical", ticks=np.arange(-1.0, 1.1, 0.2)
)  # 创建颜色条，指定其在cax_colorbar上，方向为垂直，刻度从-1.0到1.0，步长0.2
cbar.ax.tick_params(labelsize=8)  # 设置颜色条刻度标签的字体大小
cbar.set_label("Pearson correlation coefficient", size=9)  # 设置颜色条的标签文本及其字体大小
for spine in cax_colorbar.spines.values():  # 遍历颜色条Axes的所有边框 (spines)
    spine.set_edgecolor("black")
    spine.set_visible(True)
    spine.set_linewidth(1.0)  # 设置边框颜色为黑色，可见，线宽为1.0
# 创建并定位P值图例
legend_ax_width = 0.12  # P值图例的宽度 (用户设定值，图形坐标系中占总宽度的百分比)
legend_ax_height = 0.11  # P值图例的高度 (用户设定值，图形坐标系中占总高度的百分比)
legend_ax_left = 0.48  # P值图例左下角的X坐标 (例如，0.68 表示从图形左边缘起68%的位置)
legend_ax_bottom = 0.62  # P值图例左下角的Y坐标 (例如，0.82 表示从图形下边缘起82%的位置)
# 创建P值图例的Axes对象
legend_ax = fig.add_axes([legend_ax_left, legend_ax_bottom, legend_ax_width, legend_ax_height])
legend_ax.set_axis_off()  # 关闭P值图例Axes的默认坐标轴、刻度、标签和背景
# 创建并添加FancyBboxPatch作为P值图例的圆角背景和边框
fancy_bbox_style = FancyBboxPatch(  # 创建圆角矩形对象
    (0, 0),  # 圆角矩形左下角在legend_ax坐标系中的位置 (0,0)
    1,  # 圆角矩形的宽度 (1表示占满legend_ax的100%宽度)
    1,  # 圆角矩形的高度 (1表示占满legend_ax的100%高度)
    boxstyle="round,pad=0.0",  # 设置框样式为圆角，pad=0表示在名义边界处进行圆角处理
    fc="white",  # 填充颜色为白色
    ec="black",  # 边框颜色为黑色
    lw=2.0,  # 边框线宽为2.0
    alpha=1,  # 透明度为0.85
    transform=legend_ax.transAxes,  # 指定坐标变换基于legend_ax的坐标系
    zorder=0.5,  # 设置堆叠顺序，使其位于文本元素之下
)
legend_ax.add_patch(fancy_bbox_style)  # 将创建的圆角矩形添加到legend_ax中
# --- P值图例内容 ---
# 在FancyBboxPatch之上绘制文本和线条
# 文本坐标相对于legend_ax (0到1)
# 标题文本
title_x_pos = 0.05  # 标题文本在legend_ax中的x位置 (左边距5%)
title_y_pos = 0.85  # 标题文本在legend_ax中的y位置 (用户设定值，va='top'将以此为顶部对齐)
legend_ax.text(
    title_x_pos,
    title_y_pos,
    "FDR-adjusted\nP value",  # 绘制标题文本
    fontsize=9,  # 字体大小
    va="top",  # 垂直方向顶部对齐
    ha="left",  # 水平方向左对齐
    fontweight="bold",  # 字体加粗
    linespacing=1.3,  # 行间距
    transform=legend_ax.transAxes,  # 指定坐标变换基于legend_ax
    zorder=1,
)  # 设置堆叠顺序，确保文本在圆角矩形之上
# 水平分隔线
y_hline = 0.6  # 水平线在legend_ax中的y位置 (用户设定值)
line_xmin_rel = 0.05  # 水平线在legend_ax中的起始x位置 (相对legend_ax宽度)
line_xmax_rel = 0.3  # 水平线在legend_ax中的结束x位置 (用户设定值，相对legend_ax宽度)
legend_ax.axhline(
    y_hline,
    xmin=line_xmin_rel,
    xmax=line_xmax_rel,  # 绘制水平线
    color="black",
    linewidth=0.7,
    zorder=1,
)  # 颜色黑色，线宽0.7，堆叠顺序为1
# 星号和P值条目
text_entry_y_positions = [0.50, 0.33, 0.16]  # 定义每行条目的y位置 (用户设定值)
star_x_entry_pos = 0.1  # 星号的x位置
pval_text_x_entry_pos = 0.3  # P值文本的x位置
if legend_ax_width >= 0.3:  # 如果图例框较宽，则调整星号和文本的x位置以分散内容
    star_x_entry_pos = 0.14  # 调整星号x位置
    pval_text_x_entry_pos = 0.38  # 调整P值文本x位置
p_value_display_texts = ["< 0.05", "< 0.01", "< 0.001"]  # 定义P值区间的显示文本
star_display_texts = ["*", "**", "***"]  # 定义对应的星号显示文本
for i in range(len(star_display_texts)):  # 循环遍历每个显著性级别
    legend_ax.text(
        star_x_entry_pos,
        text_entry_y_positions[i],
        star_display_texts[i],  # 绘制星号
        fontsize=10,
        ha="left",
        va="center",  # 字体大小，左对齐，垂直居中
        transform=legend_ax.transAxes,
        zorder=1,
    )  # 基于legend_ax坐标系，堆叠顺序为1
    legend_ax.text(
        pval_text_x_entry_pos,
        text_entry_y_positions[i],
        p_value_display_texts[i],  # 绘制P值文本
        fontsize=9,
        ha="left",
        va="center",  # 字体大小，左对齐，垂直居中
        transform=legend_ax.transAxes,
        zorder=1,
    )  # 基于legend_ax坐标系，堆叠顺序为1

plt.savefig(str(OUTPUT_DIR / "dcyzjz.png"), dpi=300, bbox_inches="tight", pad_inches=0.2)
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

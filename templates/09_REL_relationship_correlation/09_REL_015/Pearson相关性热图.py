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

import matplotlib  # 导入matplotlib库本身，用于一些底层设置
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块，通常简写为plt
import numpy as np  # 导入numpy库，用于科学计算，特别是数组操作，通常简写为np
import pandas as pd  # 导入pandas库，用于数据处理和读取Excel文件，通常简写为pd
from matplotlib.patches import Rectangle  # 从matplotlib中导入用于绘制矩形(正方形)的工具
from scipy import stats  # 导入scipy库中的stats模块，用于进行统计计算，如相关性和p值

matplotlib.use("Agg")  # 设置matplotlib的后端，'TkAgg'是一个常用的图形界面后端
plt.rcParams["font.sans-serif"] = [
    "Times New Roman"
]  # 设置全局字体为'Times New Roman'，用于图表中的所有文本
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题
print("--- Part 1: 开始从真实数据计算相关性和显著性 ---")
# --- 【请在此处配置您的数据】 ---
# 1. 指定您的数据文件路径
#    这个文件应该是一个Excel，其中包含了下面 x_labels 和 y_labels 中的所有列
data_file_path = str(
    DATA_DIR / "simulated_correlation_data.xlsx"
)  # <--- 请将这里替换为您的文件路径
# 2. 定义您要分析的两组变量 (行和列)
# y_labels 定义了热图Y轴（行）的标签，代表一组变量
y_labels = [
    "Protist_richness",
    "Fungi_richness",
    "Bacteria_richness",
    "Verrucomicrobia",
    "Planctomycetes",
    "Nitrospirae",
    "Gemmatimonadetes",
    "Gammaproteobacteria",
    "Firmicutes",
    "Deltaproteobacteria",
    "Chloroflexi",
    "Betaproteobacteria",
    "Bacteroidetes",
    "Alphaproteobacteria",
    "Actinobacteria",
    "Acidobacteria",
    "Mortierellomycota",
    "Basidiomycota",
    "Ascomycota",
    "Saprotroph",
    "Pathogen",
    "AM_Fungi",
    "Ochrophyta",
    "Lobosa",
    "Conosa",
    "Ciliophora",
    "Cercozoa",
    "Phototroph",
    "Parasite",
    "Consumer",
]
# x_labels 定义了热图X轴（列）的标签，代表另一组变量
x_labels = ["Bacterial", "F:B ratio", "Fungi", "Microbial", "ER", "GPP", "HR", "NEE", "SR"]
# 读取源数据
# 尝试使用pandas的read_excel函数读取指定路径的Excel文件
source_df = pd.read_excel(data_file_path)
# 如果成功，打印确认信息
print(f"成功读取数据文件: {data_file_path}")
# 初始化用于存储结果的两个空的DataFrame
# corr_df 用于存储相关系数(r值)
corr_df = pd.DataFrame(index=y_labels, columns=x_labels, dtype=float)
# sig_df 用于存储显著性标记(如***, **, *)
sig_df = pd.DataFrame("", index=y_labels, columns=x_labels)
print("--- 开始计算相关性和p值... ---")
# 使用双重循环遍历y_labels和x_labels中的每一对变量
for y_var in y_labels:
    for x_var in x_labels:
        # 检查两个变量名是否存在于源数据DataFrame的列名中
        if y_var in source_df.columns and x_var in source_df.columns:
            # 提取这两列数据，并使用.dropna()去除任何包含缺失值(NaN)的行，以确保数据对齐
            temp_df = source_df[[y_var, x_var]].dropna()
            # 只有当清理缺失值后剩余的数据点数量大于等于3时，才进行计算
            if len(temp_df) >= 3:
                # 使用scipy.stats.pearsonr计算皮尔逊相关系数(corr)和对应的p值(p_value)
                corr, p_value = stats.pearsonr(temp_df[y_var], temp_df[x_var])
                # 将计算出的相关系数填充到结果矩阵的对应位置
                corr_df.loc[y_var, x_var] = corr
                # 根据p值的大小，判断显著性水平，并填充对应的星号标记
                if p_value < 0.001:
                    sig_df.loc[y_var, x_var] = "***"  # 极显著
                elif p_value < 0.01:
                    sig_df.loc[y_var, x_var] = "**"  # 显著
                elif p_value < 0.05:
                    sig_df.loc[y_var, x_var] = "*"  # 存在相关性
                # else: p值大于等于0.05时，保持为空字符串 ''，表示不显著
            else:
                # 如果有效数据点不足，则将相关性设为NaN（Not a Number）
                corr_df.loc[y_var, x_var] = np.nan
                # 显著性标记为空
                sig_df.loc[y_var, x_var] = ""
        else:
            # 如果源数据中不存在某个变量列，也将相关性设为NaN
            corr_df.loc[y_var, x_var] = np.nan
            # 显著性标记为空
            sig_df.loc[y_var, x_var] = ""
# 使用.fillna(0, inplace=True)将计算中可能出现的空值(NaN)替换为0，以避免后续绘图出错
corr_df.fillna(0, inplace=True)
print("--- 计算完成 ---")
print("\n\n" + "=" * 50)
print("--- 相关性分析结果 ---")
print("\n--- 相关性系数矩阵 (Correlation Coefficients) ---")
# 设置pandas的显示选项，以便在控制台能完整地看到所有行和列
pd.set_option("display.max_rows", None)  # 显示所有行
pd.set_option("display.max_columns", None)  # 显示所有列
pd.set_option("display.width", 1000)  # 设置显示宽度
# 打印相关系数矩阵
print(corr_df)
print("\n--- 显著性水平矩阵 (Significance Stars) ---")
# 打印显著性标记矩阵
print(sig_df)
# 定义结果输出文件的完整路径
output_results_path = str(OUTPUT_DIR / "correlation_results.xlsx")

# 使用 pd.ExcelWriter 创建一个Excel文件写入器，可以向同一个文件写入多个工作表
with pd.ExcelWriter(output_results_path) as writer:
    # 将相关性系数矩阵写入名为 'Correlation_Coefficients' 的工作表
    corr_df.to_excel(writer, sheet_name="Correlation_Coefficients")
    # 将显著性标记矩阵写入名为 'Significance_Stars' 的工作表
    sig_df.to_excel(writer, sheet_name="Significance_Stars")
    # 如果成功，打印保存确认信息
print(f"\n--- 分析结果已成功保存到Excel文件: {output_results_path} ---")
print("=" * 50 + "\n")
# 恢复pandas的默认显示选项，避免影响其他代码的输出格式
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.reset_option("display.width")
# a. 初始化画布和坐标轴
# figsize=(10, 20) 设置了图像的尺寸，宽度为10英寸，高度为20英寸
fig, ax = plt.subplots(figsize=(10, 20))
# b. 设置坐标轴范围，并强制单元格为正方形
# xlim 和 ylim 设置了x轴和y轴的显示范围，-0.5到len-0.5是为了让单元格居中
ax.set_xlim(-0.5, len(x_labels) - 0.5)
ax.set_ylim(len(y_labels) - 0.5, -0.5)  # y轴是反向的，从上到下递增
# ax.set_aspect('equal') 确保x轴和y轴的单位长度相同，从而使单元格成为正方形
ax.set_aspect("equal")
# c. 定义颜色映射
vmax = 1  # 定义相关性绝对值的最大值，用于颜色映射的标准化
# 创建一个从-vmax到+vmax的线性颜色标准化器
norm = plt.Normalize(-vmax, vmax)
# 获取一个预定义的颜色映射方案'RdBu_r'（红-白-蓝，反向），红色代表正相关，蓝色代表负相关
cmap = plt.get_cmap("RdBu_r")
# d. 遍历所有单元格，绘制图形
for i in range(len(y_labels)):  # 遍历行
    for j in range(len(x_labels)):  # 遍历列
        # 获取当前单元格的相关性值和显著性标记
        corr_val = corr_df.iloc[i, j]
        sig_val = sig_df.iloc[i, j]
        # 1. 绘制按面积缩放的居中正方形
        # 只有当相关性值的绝对值大于一个很小的阈值时才绘制正方形，避免绘制几乎看不见的点
        if abs(corr_val) > 0.01:
            # 【核心修正】计算正方形的边长。面积与相关性绝对值成正比，所以边长与相关性绝对值的平方根成正比
            # np.sqrt(abs(corr_val) / vmax) 计算缩放后的尺寸
            # min(..., 1.0) 确保尺寸最大不超过1.0，防止正方形超出单元格
            size = min(np.sqrt(abs(corr_val) / vmax), 1.0)
            # 使用之前定义的颜色映射和标准化器，根据相关性值获取颜色
            color = cmap(norm(corr_val))
            # 计算正方形左下角的坐标，使其在单元格中居中
            x_pos = j - size / 2
            y_pos = i - size / 2
            # 创建一个矩形（正方形）对象
            rect = Rectangle(
                (x_pos, y_pos), size, size, facecolor=color, edgecolor="none"
            )  # edgecolor='none'表示无边框
            # 将这个正方形添加到坐标轴上
            ax.add_patch(rect)
        # 2. 在单元格中心绘制显著性星号
        # 如果当前单元格有显著性标记（即不为空字符串）
        if sig_val:
            # 在(j, i)坐标处（单元格中心）添加文本
            ax.text(j, i, sig_val, ha="center", va="center", color="black", fontsize=14)
# e. 重新添加手动绘制的网格线
# 设置次要刻度线的位置，使其位于单元格的边界上
ax.set_xticks(np.arange(len(x_labels) + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(len(y_labels) + 1) - 0.5, minor=True)
# 在次要刻度线的位置绘制网格线
ax.grid(which="minor", color="black", linestyle="-", linewidth=1.5)
# 隐藏次要刻度线本身的小标记
ax.tick_params(which="minor", size=0)
# 设置主刻度和标签
# 设置主刻度的位置在每个单元格的中心
ax.set_xticks(np.arange(len(x_labels)))
# 设置x轴的标签，并旋转45度，ha='left'表示标签左对齐
ax.set_xticklabels(x_labels, rotation=45, ha="left")
# 设置y轴主刻度的位置
ax.set_yticks(np.arange(len(y_labels)))
# 设置y轴的标签
ax.set_yticklabels(y_labels)
# 将x轴的刻度和标签移动到图表的顶部
ax.xaxis.tick_top()
# 设置y轴标签的字体大小
ax.tick_params(axis="y", labelsize=20)
# 设置x轴标签的字体大小和与轴线的间距
ax.tick_params(axis="x", labelsize=20, pad=5)
# 隐藏主刻度线本身的小标记
ax.tick_params(which="major", size=0)
# 移除默认的坐标轴外框线（spines）
for spine in ax.spines.values():
    spine.set_visible(False)
# 定义主图区域在整个画布中的位置和大小（边距调整）
# [左边距, 底边距, 宽度, 高度]，所有值都是相对于画布尺寸的比例
left, bottom, width, height = 0.25, 0.05, 0.65, 0.8
# 应用这个位置设置
ax.set_position([left, bottom, width, height])
# 定义颜色条（colorbar）在画布中的位置
cbar_ax = fig.add_axes([left + width - 0.08, bottom, 0.02, height])
# 创建一个可映射对象，用于生成颜色条
mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
# 在指定的位置(cbar_ax)绘制颜色条
cb = plt.colorbar(mappable, cax=cbar_ax)
# 设置颜色条刻度标签的字体大小
cb.ax.tick_params(labelsize=20)
# 将最终的图像保存到文件，dpi=300表示设置高分辨率
plt.savefig(
    str(OUTPUT_DIR / "矩阵热图.png"),
    dpi=300,
    bbox_inches="tight",  # <--- 添加这个参数
)
# 显示图像窗口
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

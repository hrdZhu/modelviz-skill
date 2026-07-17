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

# 导入numpy库，用于进行科学计算，特别是数组操作
# 导入matplotlib主库
import matplotlib

# 导入matplotlib的gridspec模块，用于创建复杂的子图布局
import matplotlib.gridspec as gridspec

# 导入matplotlib的patches模块，用于绘制如图例中的色块或椭圆等形状
import matplotlib.patches as mpatches

# 导入matplotlib的pyplot模块，用于创建图形和可视化
import matplotlib.pyplot as plt
import numpy as np

# 导入pandas库，用于数据处理和分析，特别是DataFrame操作
import pandas as pd

# 导入seaborn库，用于绘制更美观的统计图形
import seaborn as sns

# 从matplotlib.lines模块导入Line2D，用于创建自定义的线型图例
from matplotlib.lines import Line2D

# 从scipy库中导入计算距离和转换距离矩阵格式的函数
from scipy.spatial.distance import pdist, squareform

# 从scipy库中导入卡方分布函数，用于计算置信椭圆
from scipy.stats import chi2

# 从skbio库中导入DistanceMatrix（距离矩阵）对象和permanova函数
# 恢复为导入permanova，因为adonis在您的版本中不存在
from skbio.stats.distance import DistanceMatrix, permanova

# 从skbio库中导入主坐标分析（PCoA）函数
from skbio.stats.ordination import pcoa

# 设置matplotlib的后端为'TkAgg'，这是一种图形用户界面工具包，有助于在某些操作系统或环境中正确弹出显示图形窗口
matplotlib.use("Agg")
# --- 1. 从Excel文件加载实际数据 ---
# 定义包含您实际数据的Excel文件的完整路径
# !!! 请确保将此路径修改为您的实际文件路径 !!!
input_excel_path = str(DATA_DIR / "simulated_abundance_data.xlsx")
# 使用try-except结构来尝试读取文件，这样如果文件不存在，程序会给出提示而不是直接崩溃
# index_col=0 参数告诉pandas将文件中的第一列作为DataFrame的行索引（即样本名称）
df_loaded_data = pd.read_excel(input_excel_path, index_col=0)
# 如果文件成功读取，打印一条成功信息
print(f"成功从文件: {input_excel_path} 加载数据。")
# 从加载的数据中分离出分组信息和丰度数据
# 将'Group'列提取出来，作为独立的分组信息系列（Series）
df_grouping = df_loaded_data["Group"]
# 从原始加载的DataFrame中删除'Group'列，剩下的就是物种丰度数据
df_abundance = df_loaded_data.drop("Group", axis=1)
# 从加载的数据中动态地获取关键信息，以确保后续代码的兼容性
# 获取所有唯一的分组名称，并转换为列表
groups = df_grouping.unique().tolist()
# 获取样本总数
total_samples = len(df_loaded_data)
# --- 2. 计算Bray-Curtis距离矩阵 ---
# 使用pdist函数计算丰度数据中每对样本之间的Bray-Curtis相异性指数，返回一个压缩后的一维数组
bc_dist_matrix = squareform(pdist(df_abundance, metric="braycurtis"))
# 使用squareform函数将一维距离数组转换为对称的二维方阵（DataFrame）
df_dist_matrix = pd.DataFrame(bc_dist_matrix, index=df_abundance.index, columns=df_abundance.index)
# 将pandas的DataFrame转换为skbio库专用的DistanceMatrix对象，以满足后续函数的要求
skbio_dist_matrix = DistanceMatrix(
    np.ascontiguousarray(df_dist_matrix.to_numpy(dtype=float)), ids=df_dist_matrix.index
)
# --- 3. 执行PERMANOVA检验并手动计算R2 ---
# 使用转换后的距离矩阵和分组信息进行PERMANOVA（多元方差分析）检验
permutation_count = 9999 if _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1" else 999
adonis_results = permanova(skbio_dist_matrix, df_grouping, permutations=permutation_count)

# 打印一个分隔符，使输出更清晰
print("\n" + "=" * 30)
# 打印标题
print("PERMANOVA (Adonis) 检验结果:")
# 打印permanova函数返回的完整结果对象
print(adonis_results)
# 打印一个分隔符
print("=" * 30 + "\n")

# 从检验结果中提取F统计量（test statistic）
f_stat = adonis_results["test statistic"]
# 从检验结果中提取p值（p-value）
p_val = adonis_results["p-value"]
# 获取分组的数量
num_groups = len(groups)
# 计算组间自由度 (分组数 - 1)
df_between = num_groups - 1
# 计算组内自由度 (样本总数 - 分组数)
df_within = total_samples - num_groups
# 根据公式手动计算R2值
r2_val = (f_stat * df_between) / (f_stat * df_between + df_within)

# --- 4. 执行PCoA分析 ---
# 对距离矩阵进行主坐标分析（PCoA）
pcoa_results = pcoa(skbio_dist_matrix)

# 打印一个分隔符
print("\n" + "=" * 30)
# 打印标题
print("PCoA 分析结果:")
# 打印一个子标题
print("\n每个主坐标轴的解释度 (Proportion Explained):")
# 打印PCoA结果中每个轴的解释度系列
print(pcoa_results.proportion_explained)
# 打印一个子标题
print("\n样本在前几个主坐标轴上的得分 (前5行预览):")
# 打印PCoA结果中样本得分DataFrame的前5行
print(pcoa_results.samples.head())
# 打印一个分隔符
print("=" * 30 + "\n")

# 从PCoA结果中提取前两个主坐标（PC1, PC2）的样本得分
df_pcoa_coords = pcoa_results.samples[["PC1", "PC2"]]
# 将列名重命名为更具可读性的'PCoA1'和'PCoA2'
df_pcoa_coords.columns = ["PCoA1", "PCoA2"]
# 将PCoA坐标与分组信息合并到一个DataFrame中，方便后续绘图
df_pcoa_coords = pd.concat([df_pcoa_coords, df_grouping], axis=1)
# 从PCoA结果中获取每个主坐标轴的解释变异度
proportion_explained = pcoa_results.proportion_explained
# 计算第一主坐标的解释度百分比
pcoa1_exp = proportion_explained["PC1"] * 100
# 计算第二主坐标的解释度百分比
pcoa2_exp = proportion_explained["PC2"] * 100

# --- 5. 可视化绘图 ---
# 注意：这里的颜色和标记映射是预先设定的。如果您的组名不同，需要在此处修改。
# 为每个组定义颜色
color_map = {"KC": "#4D4D4D", "MC": "#A6564B", "WB": "#E8A243"}
# 为每个组定义标记形状
marker_map = {"KC": "o", "MC": "^", "WB": "s"}
# 创建一个10x10英寸的图形，并启用constrained_layout以更好地自动调整布局
fig = plt.figure(figsize=(12, 10))
# 使用GridSpec创建一个4x4的网格来布局子图
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.1, wspace=0.1)
# 将中心散点图放置在网格的特定区域（第1到3行，第0到2列）
ax_scatter = fig.add_subplot(gs[1:4, 0:3])
# 将顶部密度图放置在网格的第0行，并与散点图共享x轴
ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
# 将右侧密度图放置在网格的第3列，并与散点图共享y轴
ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

# 遍历按'Group'分组后的数据，为每个组绘图
for group_name, group_data in df_pcoa_coords.groupby("Group"):
    # 获取当前组的颜色，如果组名不在预设中，则使用默认的灰色
    color = color_map.get(group_name, "gray")
    # 获取当前组的标记形状，如果组名不在预设中，则使用默认的'x'
    marker = marker_map.get(group_name, "x")

    # 在中心图上绘制当前组的散点
    ax_scatter.scatter(
        group_data["PCoA1"],
        group_data["PCoA2"],
        marker=marker,
        s=100,
        color=color,
        alpha=0.7,  # 设置标记、大小、颜色、透明度
        edgecolor="white",
        linewidth=0.5,
        label=group_name,
    )  # 设置边缘颜色和线宽
    # --- 计算并绘制置信椭圆 ---
    # 获取当前组的坐标数据
    coords = group_data[["PCoA1", "PCoA2"]].values
    # 计算当前组的质心（均值）
    centroid = coords.mean(axis=0)
    # 计算当前组坐标的协方差矩阵
    cov = np.cov(coords, rowvar=False)
    # 计算协方差矩阵的特征值和特征向量
    vals, vecs = np.linalg.eigh(cov)
    # 对特征值和特征向量进行排序，以确保椭圆长轴对应最大的特征值
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    # 计算椭圆的旋转角度
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # 计算95%置信椭圆的半径缩放因子
    s = np.sqrt(chi2.ppf(0.95, 2))
    # 计算椭圆的长轴和短轴长度
    width, height = 2 * s * np.sqrt(vals)
    # 创建一个椭圆对象
    ellipse = mpatches.Ellipse(
        xy=centroid,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=0.2,
        edgecolor=color,  # 设置填充色、透明度和边缘色
        linestyle="--",
        linewidth=2,
        fill=True,
    )  # 设置边缘线型和线宽
    # 将创建好的椭圆添加到散点图中
    ax_scatter.add_patch(ellipse)

# --- 美化中心散点图 ---
# 绘制y=0的水平虚线
ax_scatter.axhline(0, color="gray", linestyle="--", linewidth=1)
# 绘制x=0的垂直虚线
ax_scatter.axvline(0, color="gray", linestyle="--", linewidth=1)
# 设置x轴标签，并包含解释度信息
ax_scatter.set_xlabel(f"PCoA 1 ({pcoa1_exp:.1f}%)", fontsize=14)
# 设置y轴标签，并包含解释度信息
ax_scatter.set_ylabel(f"PCoA 2 ({pcoa2_exp:.1f}%)", fontsize=14)
# 设置坐标轴刻度标签的字体大小
ax_scatter.tick_params(axis="both", which="major", labelsize=12)

# --- 绘制边缘密度图 ---
# 注意：如果您的组名与'KC', 'MC', 'WB'不同，这里的hue_order需要更新，或者直接移除
# 为了代码通用性，可以不指定hue_order，让seaborn自动处理
# 使用seaborn绘制顶部的核密度估计图（KDE）
sns.kdeplot(
    data=df_pcoa_coords,
    x="PCoA1",
    hue="Group",
    palette=color_map,
    fill=True,
    alpha=0.5,
    linewidth=2,
    ax=ax_histx,
    legend=False,
)
# 使用seaborn绘制右侧的核密度估计图
sns.kdeplot(
    data=df_pcoa_coords,
    y="PCoA2",
    hue="Group",
    palette=color_map,
    fill=True,
    alpha=0.5,
    linewidth=2,
    ax=ax_histy,
    legend=False,
)

# --- 美化顶部密度图 ---
# 隐藏上、右、左的坐标轴线
ax_histx.spines[["top", "right", "left"]].set_visible(False)
# 隐藏x轴的刻度和标签
ax_histx.tick_params(axis="x", bottom=False, labelbottom=False)
# 隐藏y轴的刻度和标签
ax_histx.tick_params(axis="y", left=False, labelleft=False)
# 清空y轴标签
ax_histx.set_ylabel("")
# 清空x轴标签
ax_histx.set_xlabel("")

# --- 美化右侧密度图 ---
# 隐藏上、右、下的坐标轴线
ax_histy.spines[["top", "right", "bottom"]].set_visible(False)
# 隐藏x轴的刻度和标签
ax_histy.tick_params(axis="x", bottom=False, labelbottom=False)
# 隐藏y轴的刻度和标签
ax_histy.tick_params(axis="y", left=False, labelleft=False)
# 清空y轴标签
ax_histy.set_ylabel("")
# 清空x轴标签
ax_histy.set_xlabel("")

# --- 创建和添加图例与注释 ---
# 动态创建图例元素列表，以适应不同的组名
legend_elements = [
    # 为每个组创建一个Line2D对象作为图例项
    Line2D(
        [0],
        [0],
        marker=marker_map.get(g, "x"),
        color="w",
        label=g,  # 设置标记形状、颜色和标签文本
        markerfacecolor=color_map.get(g, "gray"),
        markersize=10,
        markeredgecolor="black",
    )  # 设置标记填充色、大小和边缘色
    for g in groups  # 遍历所有组
]
# 在散点图上显示图例
ax_scatter.legend(handles=legend_elements, loc="best", edgecolor="black", prop={"size": 12})

# 准备Adonis检验结果的文本
adonis_text = f"adonis R2: {r2_val:.2f}; P-value: {p_val:.3f}"
# 在图形的指定位置（底部中心）添加文本注释
fig.text(
    0.5,
    0.03,
    adonis_text,
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=1, edgecolor="none"),
)  # 给文本添加半透明白色背景
# ******************** 新增代码：保存分析结果到Excel 开始 ********************
# 定义要保存的结果Excel文件路径
results_excel_path = str(OUTPUT_DIR / "analysis_results.xlsx")

# 使用ExcelWriter，可以将多个DataFrame写入到同一个Excel文件的不同工作表中
with pd.ExcelWriter(results_excel_path) as writer:
    # 将PERMANOVA结果写入名为'PERMANOVA_Results'的sheet
    adonis_results.to_excel(writer, sheet_name="PERMANOVA_Results")
    # 将PCoA样本得分写入名为'PCoA_Sample_Scores'的sheet
    pcoa_results.samples.to_excel(writer, sheet_name="PCoA_Sample_Scores")
    # 将PCoA每个轴的解释度写入名为'PCoA_Proportion_Explained'的sheet
    pcoa_results.proportion_explained.to_excel(writer, sheet_name="PCoA_Proportion_Explained")
    # 将Bray-Curtis距离矩阵写入名为'Bray_Curtis_Distance_Matrix'的sheet
    df_dist_matrix.to_excel(writer, sheet_name="Bray_Curtis_Distance_Matrix")

print(f"所有分析结果已成功保存到: {results_excel_path}")
# ******************** 新增代码：保存分析结果到Excel 结束 ********************
# --- 保存并显示图形 ---
# 将最终的图形保存为PNG文件，并设置分辨率为300dpi
plt.savefig(str(OUTPUT_DIR / "PCoA.png"), dpi=300)
# 在屏幕上显示图形
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

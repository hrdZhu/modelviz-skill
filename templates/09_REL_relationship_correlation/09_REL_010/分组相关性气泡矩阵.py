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

# 导入所需的库
import os  # 导入os库，用于处理文件和目录路径

import matplotlib  # Matplotlib主库
import matplotlib.lines as mlines  # 用于在图例中创建自定义的线或标记
import matplotlib.pyplot as plt  # 用于数据可视化，创建图表
import numpy as np  # 用于数值计算，例如处理NaN值
import pandas as pd  # 用于数据处理和分析，特别是DataFrame的操作
from scipy.stats import pearsonr  # 从Scipy库中导入pearsonr函数，用于计算皮尔逊相关系数和p值
from statsmodels.stats.multitest import (
    fdrcorrection,  # 用于进行FDR（错误发现率）校正，控制多重检验中的假阳性
)

# 设置Matplotlib的后端。'TkAgg'是一个常用的后端，它允许在Tkinter窗口中显示图形。
# 在某些环境下，可能需要设置此项以确保图形能正常显示。
matplotlib.use("Agg")
# --- 0. 全局样式设置 ---
# 设置图表中所有文本的默认字体为 'Times New Roman'
plt.rcParams["font.family"] = "Times New Roman"
# 设置图表中所有文本的默认字体大小为 10
plt.rcParams["font.size"] = 10
# --- 从本地Excel文件加载数据 ---
# !!! 注意：请将下面的文件路径替换为您自己的Excel文件路径 !!!
# 文件需要包含一个名为 'Region' 的分组列，一个名为 'Target_Y' 的目标变量列，以及所有需要分析的自变量列。
input_filename = str(DATA_DIR / "simulated_correlation_data.xlsx")
# --- 新增代码：指定结果输出文件夹 ---
# !!! 注意：请将下面的文件夹路径替换为您希望保存结果的文件夹路径 !!!
output_folder = str(OUTPUT_DIR)
# --- 创建输出文件夹 ---
# 如果指定的输出文件夹不存在，则自动创建它
os.makedirs(output_folder, exist_ok=True)
# 尝试使用pandas的read_excel函数读取指定路径的Excel文件
full_df = pd.read_excel(input_filename)
# 如果成功，打印成功加载文件的消息
print(f"成功从 '{input_filename}' 加载数据。")
# 从加载的数据（DataFrame）中获取分组和变量信息
# 提取 'Region' 列中的所有唯一值，并将其转换为列表
regions = full_df["Region"].unique().tolist()
# 定义一个列表，包含所有需要进行相关性分析的自变量的列名
variables = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "EVI",
    "EVI2",
    "NDPI",
    "NDVI",
    "OSAVI",
    "RVI",
    "DEM",
    "Slope",
    "Latitude",
    "Longitude",
    "Air Pressure",
    "Precipitation",
    "Temperature",
]
# --- 分析 ---
# 初始化一个空列表，用于存储每个区域和每个变量的分析结果
results = []
# 外层循环：遍历每一个区域
for region in regions:
    # 根据当前区域名称，从总数据中筛选出该区域的数据
    region_df = full_df[full_df["Region"] == region]
    # 内层循环：遍历每一个需要分析的自变量
    for var in variables:
        # 检查以确保当前变量列和目标变量'Target_Y'列都存在于区域数据中，
        # 并且这两列都不是完全由空值（NaN）组成
        if (
            var in region_df
            and "Target_Y" in region_df
            and not region_df[var].isnull().all()
            and not region_df["Target_Y"].isnull().all()
        ):
            # 如果数据有效，使用pearsonr计算自变量和目标变量'Target_Y'之间的皮尔逊相关系数和p值
            corr, p_value = pearsonr(region_df[var], region_df["Target_Y"])
            # 如果计算出的p值是NaN（通常因为数据标准差为0），则将其设为1.0（不显著）
            if np.isnan(p_value):
                p_value = 1.0
            # 将结果（区域、变量、相关系数、p值）作为一个字典添加到results列表中
            results.append(
                {"Region": region, "Variable": var, "Correlation": corr, "p_value": p_value}
            )
        else:
            # 如果数据无效（例如列不存在或全为空），则添加一个默认的结果（相关性为0，p值为1.0）
            results.append({"Region": region, "Variable": var, "Correlation": 0, "p_value": 1.0})
# 将存储结果的列表转换为Pandas DataFrame，方便后续处理和分析
results_df = pd.DataFrame(results)
# 再次检查p值列，并将其中任何可能存在的NaN值填充为1.0
results_df["p_value"] = results_df["p_value"].fillna(1.0)
# 对所有的p值进行FDR校正，以获得q值。这有助于在进行大量相关性检验时控制假阳性率。
# fdrcorrection返回两个值：一个布尔数组表示在给定alpha下是否拒绝原假设，以及校正后的q值。
rejected, q_values = fdrcorrection(results_df["p_value"])
# 将计算出的q值作为一个新列'q_value'添加到结果DataFrame中
results_df["q_value"] = q_values
# --- 打印并保存分析结果 ---
print("\n--- 相关性分析结果 ---")
# 打印整个结果DataFrame。使用to_string()可以确保即使有很多行，也会全部显示出来，而不是被截断。
print(results_df.to_string())
# 将分析结果DataFrame保存到CSV文件
# 使用 os.path.join 构造完整的文件路径，确保跨平台兼容性
output_csv_path = os.path.join(output_folder, str(OUTPUT_DIR / "correlation_analysis_results.csv"))
# to_csv函数将DataFrame写入CSV文件。index=False表示不将DataFrame的索引写入文件。
# encoding='utf-8-sig' 确保在Excel中打开时中文字符不会乱码。
results_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
print(f"\n分析结果已成功保存到: {output_csv_path}")


# ---数据可视化 ---
# 定义一个函数，用于根据每一行数据的相关性(corr)和q值(q_val)来决定气泡的大小和颜色
def get_plot_attributes_final(row):
    # 从输入行中获取'Correlation'和'q_value'
    corr, q_val = row["Correlation"], row["q_value"]
    # 定义气泡大小的缩放因子和基础大小
    size_scaler, base_size = 200, 20
    # 如果q值大于0.05（不显著）或q值为NaN，则设置一个默认的小尺寸和灰色
    if q_val > 0.05 or pd.isna(q_val):
        size, color = base_size, "#f0f0f0"
    else:
        # 如果结果显著，则根据相关系数的绝对值计算气泡大小
        size = abs(corr) * size_scaler + base_size
        # 如果是正相关
        if corr > 0:
            # 根据q值的不同显著性水平，赋予不同深浅的红色
            if q_val < 0.0001:
                color = "#a50f15"  # 极显著正相关
            elif q_val < 0.001:
                color = "#de2d26"
            elif q_val < 0.01:
                color = "#fb6a4a"
            else:  # q_val < 0.05
                color = "#fcae91"  # 显著正相关
        # 如果是负相关
        else:
            # 根据q值的不同显著性水平，赋予不同深浅的蓝色
            if q_val < 0.0001:
                color = "#08519c"  # 极显著负相关
            elif q_val < 0.001:
                color = "#3182bd"
            elif q_val < 0.01:
                color = "#6baed6"
            else:  # q_val < 0.05
                color = "#bdd7e7"  # 显著负相关
    # 返回一个包含计算出的尺寸和颜色的Pandas Series
    return pd.Series([size, color])


# 将上述函数应用到results_df的每一行，并将返回的尺寸和颜色分别存入新的'Size'和'Color'列
results_df[["Size", "Color"]] = results_df.apply(get_plot_attributes_final, axis=1)
# --- 最终布局与绘图 ---
# 创建一个图形窗口（Figure），设置尺寸为11x5英寸，分辨率为150 DPI
fig = plt.figure(figsize=(11, 5), dpi=150)
# 设置图形窗口的背景色为白色
fig.patch.set_facecolor("white")
# 使用绝对坐标手动添加主图表（气泡图）的坐标轴（Axes）
# [left, bottom, width, height]，所有值都是相对于图形窗口尺寸的比例
ax_main = fig.add_axes([0.12, 0.22, 0.65, 0.73])
# ---- 定义图例的复杂布局 ----
# 定义图例区域的总高度、起始y坐标和间隙
total_h, total_y_start = 0.73, 0.22
gap = 0.035
# 将图例内容区域的高度按比例 [0.43, 0.27, 0.30] 分配给三个图例部分
content_h = total_h - (2 * gap)
h_ratio = [0.43, 0.27, 0.30]
h_size, h_neg, h_pos = content_h * h_ratio[0], content_h * h_ratio[1], content_h * h_ratio[2]
# 计算每个图例部分的y坐标
y_pos_bottom = total_y_start
y_neg_bottom = y_pos_bottom + h_pos + gap
y_size_bottom = y_neg_bottom + h_neg + gap
# 根据计算出的位置和大小，添加三个用于图例的坐标轴
ax_leg_size = fig.add_axes([0.77, y_size_bottom, 0.2, h_size])  # 大小图例
ax_leg_neg = fig.add_axes([0.77, y_neg_bottom, 0.2, h_neg])  # 负相关图例
ax_leg_pos = fig.add_axes([0.77, y_pos_bottom, 0.2, h_pos])  # 正相关图例
# ---- 绘制主气泡图 ----
# 遍历结果DataFrame的每一行
for _, row in results_df.iterrows():
    # 在主坐标轴上绘制一个散点（气泡）
    # x轴是变量名，y轴是区域名，s是气泡大小，c是气泡颜色
    # linewidths=0 去掉气泡边缘线，zorder=2 使气泡在网格线之上
    ax_main.scatter(
        x=row["Variable"], y=row["Region"], s=row["Size"], c=row["Color"], linewidths=0, zorder=2
    )
# 为每个区域（y轴的每个刻度）画一条水平的灰色参考线
y_lines_center = range(len(regions))
for y_val in y_lines_center:
    ax_main.axhline(y=y_val, color="lightgrey", linestyle="-", linewidth=1, zorder=1)
# 设置x轴的刻度位置和标签
ax_main.set_xticks(np.arange(len(variables)))
ax_main.set_xticklabels(variables, rotation=45, fontsize=10)  # 标签旋转90度
# 设置y轴的刻度位置和标签
ax_main.set_yticks(np.arange(len(regions)))
ax_main.set_yticklabels(regions, fontsize=10)
# 反转y轴，使第一个区域显示在顶部
ax_main.invert_yaxis()
# 设置主图表边框（spines）的样式
for spine in ax_main.spines.values():
    spine.set_edgecolor("black")  # 边框颜色为黑色
    spine.set_linewidth(1.5)  # 边框宽度为1.5
# 调整y轴的范围，使得顶部和底部的气泡不会被边框切掉
ax_main.set_ylim(len(regions) - 0.5, -0.5)
# ---- 绘制图例 ----
# 定义图例标题的字体样式
title_font = {"weight": "bold", "size": 9, "family": "Times New Roman"}
# 定义图例标签的字体大小
label_font_size = 8
# 遍历三个图例坐标轴，并关闭它们的坐标轴显示（因为我们只需要显示图例内容）
for ax in [ax_leg_size, ax_leg_neg, ax_leg_pos]:
    ax.set_axis_off()
# -- 绘制“相关系数大小”图例 --
# 定义用于图例的几个相关系数值
size_levels = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
# 为每个大小级别创建一个“代理”艺术家（一个带有特定大小标记的不可见线条）
# marker='o' 表示圆形标记，markersize通过公式计算得出，与主图中的气泡大小计算方式相对应
size_proxies = [
    mlines.Line2D(
        [],
        [],
        color="white",
        marker="o",
        markersize=np.sqrt(lvl * 200 + 20),
        markeredgecolor="black",
        markeredgewidth=1,
        linestyle="None",
    )
    for lvl in size_levels
]
# 在ax_leg_size上创建图例
ax_leg_size.legend(
    size_proxies,
    [str(lvl) for lvl in size_levels],
    title="Correlation Coefficient",
    loc="center left",
    bbox_to_anchor=(0, 0.5),  # 图例位置
    frameon=False,
    title_fontproperties=title_font,
    fontsize=label_font_size,
    labelspacing=0.8,
)

# -- 绘制“负相关q值”图例 --
# 定义负相关的q值水平和对应的颜色
neg_q_levels = {"<0.0001": "#08519c", "<0.001": "#3182bd", "<0.01": "#6baed6", "<0.05": "#bdd7e7"}
# 为每个级别创建颜色
neg_proxies = [
    mlines.Line2D(
        [], [], color=color, marker="o", markersize=7, linestyle="None", markeredgewidth=0
    )
    for color in neg_q_levels.values()
]
# 创建图例
neg_legend = ax_leg_neg.legend(
    neg_proxies,
    neg_q_levels.keys(),
    title="Negative\ncorrelation\nq-value",
    loc="center left",
    bbox_to_anchor=(0, 0.5),
    frameon=False,
    title_fontproperties=title_font,
    fontsize=label_font_size,
)
# 设置图例标题的行间距
neg_legend.get_title().set_linespacing(1.2)
# -- 绘制“正相关q值”图例 --
# 定义正相关的q值水平和对应的颜色，包括不显著的情况
pos_q_levels = {
    "<0.0001": "#a50f15",
    "<0.001": "#de2d26",
    "<0.01": "#fb6a4a",
    "<0.05": "#fcae91",
    ">0.05": "#f0f0f0",
}
# 创建颜色列表
pos_proxies = []
# 遍历每个级别
for label, color in pos_q_levels.items():
    # 特别处理'>0.05'（不显著）的情况，为其添加一个灰色边框以便看清
    edge = "grey" if label == ">0.05" else "none"
    pos_proxies.append(
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            markersize=7,
            linestyle="None",
            markeredgecolor=edge,
            markeredgewidth=1,
        )
    )
# 创建图例
pos_legend = ax_leg_pos.legend(
    pos_proxies,
    pos_q_levels.keys(),
    title="Positive\ncorrelation\nq-value",
    loc="center left",
    bbox_to_anchor=(0, 0.5),
    frameon=False,
    title_fontproperties=title_font,
    fontsize=label_font_size,
)
# 设置图例标题的行间距
pos_legend.get_title().set_linespacing(1.2)
# 必须在 plt.show() 之前调用.
# bbox_inches='tight' 会自动裁剪图片周围多余的白边.
# dpi=300 设置图片分辨率为300，以保证图片质量.
output_figure_path = os.path.join(output_folder, str(OUTPUT_DIR / "correlation_bubble_matrix.png"))
plt.savefig(output_figure_path, bbox_inches="tight", dpi=300)
print(f"成果图已成功保存到: {output_figure_path}")
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

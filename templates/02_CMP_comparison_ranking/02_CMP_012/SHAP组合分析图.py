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

# 导入必要的库
import os
import warnings

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
# 忽略一些常见的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# --- 定义文件路径和列名 ---
excel_filename = str(DATA_DIR / "0601练习数据.xlsx")
sheet_name = "SimulatedData"
target_column_name = "Target_y"  # 目标变量所在的列名
ecoregion_column_name = "Ecoregion"  # 生态区的列名
# 使用 pandas 读取 Excel 文件指定工作表的数据到 DataFrame
df_loaded = pd.read_excel(excel_filename, sheet_name=sheet_name, engine="openpyxl")
# 打印成功加载信息和数据的形状（行数和列数）
print(f" Successfully loaded data with shape: {df_loaded.shape}")
# 定义必须存在的列名列表
required_columns = [target_column_name, ecoregion_column_name]
# 检查所有必需的列是否都在加载的 DataFrame 的列名中
if not all(col in df_loaded.columns for col in required_columns):
    # 如果有必需的列缺失，打印错误信息并退出脚本
    print(
        f"ERROR: Required columns '{target_column_name}' or '{ecoregion_column_name}' not found in the Excel sheet."
    )
    exit()
# 从加载的 DataFrame 中提取目标变量列，存储为 Series y
y = df_loaded[target_column_name]
# 创建 DataFrame X，包含除目标变量之外的所有列（即特征和生态区）
X = df_loaded.drop(columns=[target_column_name])
# --- 从数据中动态获取 Ecoregions ---
# 检查生态区列是否在 DataFrame X 中 (理论上应该在，因为前面检查过)
if ecoregion_column_name in X.columns:
    # 获取 'Ecoregion' 列中所有唯一的值
    ecoregions_from_data = X[ecoregion_column_name].unique().tolist()
    # 对获取到的生态区列表进行排序，以保证后续处理和绘图的顺序一致性
    ecoregions = sorted(ecoregions_from_data)
    # 获取生态区的数量
    n_eco = len(ecoregions)
    # 打印信息，显示动态识别出的生态区数量和列表
    print(f"Dynamically identified {n_eco} ecoregions: {ecoregions}")
else:
    # 如果生态区列在之前的检查后仍然缺失（理论上不太可能），打印错误并退出
    print(
        f"ERROR: Ecoregion column '{ecoregion_column_name}' not found after checks. Cannot proceed."
    )
    exit()
# --- 动态获取结束 ---
# 从 DataFrame X 中移除生态区列，得到只包含模型输入特征的 DataFrame X_model
X_model = X.drop(columns=[ecoregion_column_name])
# 获取实际用于模型训练的特征名称列表
features_actual = X_model.columns.tolist()
# 打印识别出的特征数量和列表
print(f"Identified {len(features_actual)} features: {features_actual}")
# 打印目标变量 y 的最小值和最大值范围
print(f"Target variable '{target_column_name}' range: {y.min():.2f} to {y.max():.2f}")
# 更新样本数量（数据行数）
n_samples = len(df_loaded)
# 更新特征数量
n_feat = len(features_actual)
# --- 训练模型 ---
print("Training model using loaded data...")
# 定义随机森林回归模型
model = RandomForestRegressor(
    n_estimators=120, random_state=42, n_jobs=-1, max_depth=18, min_samples_leaf=3, max_features=0.7
)
# 使用特征数据 X_model 和目标变量 y 训练模型
model.fit(X_model, y)
# --- 计算 SHAP 值 ---
# 打印信息，告知用户正在开始计算 SHAP 值
print("Calculating SHAP values...")
# 创建一个 TreeExplainer 对象，用于解释基于树的模型（如随机森林）
explainer = shap.TreeExplainer(model)
# 使用 explainer 计算每个样本每个特征的 SHAP 值
shap_values_raw = explainer.shap_values(X_model)
# --- 聚合 SHAP 值 ---
# 打印信息，告知用户正在聚合 SHAP 值
print("Aggregating SHAP values...")
# 计算 SHAP 值的绝对值，因为我们关心的是贡献的大小而非方向
abs_shap_values = np.abs(shap_values_raw)
# 将绝对 SHAP 值数组转换为 DataFrame，列名为实际的特征名
df_abs_shap = pd.DataFrame(abs_shap_values, columns=features_actual)
# 将原始数据中的生态区列添加到 SHAP 值 DataFrame 中，用于后续分组
df_abs_shap[ecoregion_column_name] = X[ecoregion_column_name].values
# 按生态区列分组，并计算每个生态区内每个特征的平均绝对 SHAP 值
df_shap_aggregated = df_abs_shap.groupby(ecoregion_column_name).mean()
# 按动态获取的、排序后的 ecoregions 列表和实际特征列表重新索引聚合后的 SHAP DataFrame
# 这确保了即使某些生态区在分组聚合后不存在（理论上不会），或者特征顺序不同，最终的 df_shap 也有统一的结构
# fillna(0) 将可能因 reindex 产生的缺失值（NaN）填充为 0
df_shap = df_shap_aggregated.reindex(index=ecoregions, columns=features_actual).fillna(0)
# --- 计算总体贡献并排序特征 ---
# 打印信息，告知用户正在计算特征的总体贡献并排序
print("Sorting features by contribution...")
# 按列（axis=0）求和 df_shap，得到每个特征在所有生态区的总平均绝对 SHAP 值
overall_contributions = df_shap.sum(axis=0)
# 计算所有特征总贡献之和
total_shap_sum = overall_contributions.sum()
# 检查总贡献是否大于 0，以避免除以零错误
if total_shap_sum > 0:
    # 计算每个特征贡献占总贡献的百分比
    overall_contributions_pct = (overall_contributions / total_shap_sum) * 100
else:
    # 如果总贡献为 0，则所有特征的贡献百分比都为 0
    overall_contributions_pct = pd.Series(0, index=features_actual)
# 按贡献百分比降序排列特征
overall_contributions_pct = overall_contributions_pct.sort_values(ascending=False)
# 获取排序后的特征名称列表
sorted_features = overall_contributions_pct.index.tolist()
# 筛选出贡献百分比小于 1.0% 的特征
below_threshold_features = overall_contributions_pct[overall_contributions_pct < 1.0]
# 检查是否存在贡献度低的特征
if not below_threshold_features.empty:
    # 如果存在，打印警告信息和这些特征及其贡献度
    print("WARNING: Features (from loaded data) with contribution < 1%:")
    for feature, pct in below_threshold_features.items():
        print(f" - {feature}: {pct:.4f}%")
else:
    # 如果所有特征贡献度都 >= 1%，打印成功信息
    print("SUCCESS: All features (from loaded data) have contribution >= 1%.")
# --- 准备绘图所需的数据 ---
# 打印信息，告知用户正在准备绘图所需的数据结构
print("Preparing data for plotting...")
# 创建用于绘制热图的 DataFrame，其列顺序按特征贡献度排序
df_shap_sorted_for_heatmap = df_shap[sorted_features]
# 创建一个字典，将每个排序后的特征映射到其类型 ('Var' 或 'TE')
# 通过检查特征名是否包含 '_TE' 来判断类型
feature_types = {feat: ("TE" if "_TE" in feat else "Var") for feat in sorted_features}
# 定义绘图中使用的颜色
var_color = "salmon"  # Var 类型特征的颜色
te_color = "lightblue"  # TE 类型特征的颜色
na_color = "grey"  # 热图中表示缺失或无效数据的颜色
# 计算 'Var' 类型特征的总贡献百分比
var_total_pct = overall_contributions_pct[
    overall_contributions_pct.index.map(lambda f: feature_types[f] == "Var")
].sum()
# 计算 'TE' 类型特征的总贡献百分比
te_total_pct = overall_contributions_pct[
    overall_contributions_pct.index.map(lambda f: feature_types[f] == "TE")
].sum()
# 检查两者之和，用于归一化
total_pct_sum_check = var_total_pct + te_total_pct
# 计算归一化因子，确保两者之和为 100%
norm_factor = 100 / total_pct_sum_check if total_pct_sum_check > 0 else 1
# 应用归一化因子
var_total_pct *= norm_factor
te_total_pct *= norm_factor
# 创建用于绘制饼图的数据字典
pie_data = {"Var": var_total_pct, "TE": te_total_pct}
# 创建一个字典，将所有实际特征（未排序）映射到其类型 ('Var' 或 'TE')
feature_types_orig = {feat: ("TE" if "_TE" in feat else "Var") for feat in features_actual}
# 创建一个空的 DataFrame，用于存储每个生态区 Var 和 TE 贡献的百分比（用于堆叠条形图）
# 索引使用动态获取的 ecoregions 列表
stacked_bar_data = pd.DataFrame(index=ecoregions, columns=["Var", "TE"], dtype=float)
# 遍历动态获取的每个生态区
for eco in ecoregions:
    # 从已排序和填充的 df_shap 中获取当前生态区的数据行
    eco_data = df_shap.loc[eco]
    # 计算当前生态区的总 SHAP 贡献 (所有特征之和)
    eco_total = np.nansum(eco_data)
    # 如果总贡献大于 0
    if eco_total > 0:
        # 找出属于 'Var' 类型的实际特征索引
        var_indices = [feat for feat in features_actual if feature_types_orig[feat] == "Var"]
        # 找出属于 'TE' 类型的实际特征索引
        te_indices = [feat for feat in features_actual if feature_types_orig[feat] == "TE"]
        # 计算当前生态区 'Var' 特征的总 SHAP 贡献
        var_sum = np.nansum(eco_data[var_indices])
        # 计算当前生态区 'TE' 特征的总 SHAP 贡献
        te_sum = np.nansum(eco_data[te_indices])
        # 计算 'Var' 贡献占当前生态区总贡献的百分比
        stacked_bar_data.loc[eco, "Var"] = (var_sum / eco_total) * 100
        # 计算 'TE' 贡献占当前生态区总贡献的百分比
        stacked_bar_data.loc[eco, "TE"] = (te_sum / eco_total) * 100
    else:
        # 如果当前生态区总贡献为 0，则 Var 和 TE 的百分比都设为 0
        stacked_bar_data.loc[eco, "Var"] = 0
        stacked_bar_data.loc[eco, "TE"] = 0
# 将 stacked_bar_data 中可能存在的 NaN 值（理论上不会有）填充为 0
stacked_bar_data = stacked_bar_data.fillna(0)

# --- 设置图形和 GridSpec 布局 ---
# 打印信息，告知用户正在设置图形布局
print("Setting up figure layout...")
# 设置全局字体为 'Times New Roman'
plt.rcParams["font.family"] = "Times New Roman"
# 创建一个图形对象，指定尺寸
fig = plt.figure(figsize=(14, 10))
# 创建主 GridSpec 布局管理器，定义 2 行 3 列的网格
# height_ratios 控制行高比例，width_ratios 控制列宽比例
gs_main = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 4], width_ratios=[4, 1.5, 0.7])
# 在主网格的指定位置添加子图：总体贡献条形图 (左上)
ax_bar_overall = fig.add_subplot(gs_main[0, 0])
# 在主网格的指定位置添加子图：饼图 (右上)
ax_pie = fig.add_subplot(gs_main[0, 1])
# 在主网格的指定位置添加子图：热图 (左下)
ax_heatmap = fig.add_subplot(gs_main[1, 0])
# 在主网格的指定位置添加子图：堆叠条形图 (右下中间)，并使其 Y 轴与热图共享
ax_bar_stacked = fig.add_subplot(gs_main[1, 1], sharey=ax_heatmap)
# 在主网格的右下角单元格 (gs_main[1, 2]) 内创建一个嵌套的 GridSpec 布局，用于放置图例和颜色条
# 2 行 1 列，指定行高比例和垂直间距
gs_nested = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_main[1, 2], height_ratios=[1, 2.5], hspace=0.4
)
# --- 绘制每个组件 ---
# 打印信息，告知用户正在使用加载的数据绘制图形组件
print("Plotting components using loaded data...")
# a) 热图 (左下角)
# 获取 'RdYlBu_r' (红黄蓝反转) 颜色映射，并复制一份以防修改原始映射
cmap_heatmap = plt.get_cmap("RdYlBu_r").copy()
# 设置颜色映射中用于表示 'bad' 值 (通常是 NaN) 的颜色
cmap_heatmap.set_bad(color=na_color)
# 计算 df_shap 中所有值的最大值和最小值，用于设定颜色条范围
vmax_real = np.nanmax(df_shap.values)
vmin_real = 0
# 如果数据为空或最大值接近于 0，设置一个默认的最大值以避免错误
if df_shap.empty or vmax_real < 1e-6:
    vmax_real = 1.0
# 使用 seaborn 绘制热图
# data: 排序后的 SHAP 数据ax: 指定绘图的子图对象cmap: 使用的颜色映射linewidths/linecolor: 单元格边框线宽和颜色cbar=False: 不自动添加颜色条 (我们将手动添加)vmin/vmax: 颜色映射的值范围
sns.heatmap(
    df_shap_sorted_for_heatmap,
    ax=ax_heatmap,
    cmap=cmap_heatmap,
    linewidths=0.5,
    linecolor="white",
    cbar=False,
    vmin=vmin_real,
    vmax=vmax_real,
)
ax_heatmap.set_ylabel("Ecoregion", fontsize=16, labelpad=5, fontweight="bold")  # 设置 Y 轴标签
ax_heatmap.set_xlabel("")  # 清空 X 轴标签 (通常在上方条形图显示)
x_tick_positions = np.arange(len(sorted_features)) + 0.5  # 计算 X 轴刻度的位置 (每个特征的中心)
ax_heatmap.set_xticks(x_tick_positions)  # 设置 X 轴刻度位置
ax_heatmap.set_xticklabels(
    sorted_features, rotation=90, fontsize=14, ha="center", fontweight="bold"
)  # 设置 X 轴刻度标签 (排序后的特征名)，旋转 90 度，设置字体大小和对齐方式
y_tick_positions = (
    np.arange(n_eco) + 0.5
)  # 使用动态获取的 n_eco 和 ecoregions 设置 Y 轴，计算 Y 轴刻度的位置 (每个生态区的中心)
ax_heatmap.set_yticks(y_tick_positions)  # 设置 Y 轴刻度位置
ax_heatmap.set_yticklabels(
    ecoregions, rotation=0, fontsize=16, va="center"
)  # 设置 Y 轴刻度标签 (动态获取的生态区名)，不旋转，设置字体大小和垂直对齐方式
# 调整 X 轴刻度线的参数 (不显示底部和顶部的刻度线，但显示标签)
ax_heatmap.tick_params(axis="x", bottom=False, top=False, labelbottom=True)
# 调整 Y 轴刻度线的参数 (不显示左侧和右侧的刻度线，但显示标签)
ax_heatmap.tick_params(axis="y", left=False, right=False, labelleft=True)
# b) 总体特征贡献条形图 (左上角)
bar_colors = [
    var_color if feature_types[feat] == "Var" else te_color for feat in sorted_features
]  # 根据排序后特征的类型确定每个条形的颜色
x_bar_positions = np.arange(len(sorted_features))  # 计算每个条形的 X 轴位置
bar_width = 0.7  # 设置条形的宽度
bars = ax_bar_overall.bar(
    x_bar_positions, overall_contributions_pct.values, color=bar_colors, width=bar_width
)  # 绘制条形图，x 为位置，height 为贡献百分比，指定颜色和宽度
# 设置 Y 轴标签
ax_bar_overall.set_ylabel("Feature\nContribution (%)", fontsize=14, fontweight="bold")
for tick_label in ax_bar_overall.get_yticklabels():
    # 设置该刻度标签的字体大小
    tick_label.set_fontsize(16)  # 将 14 修改为你需要的大小
    # 设置该刻度标签的字体粗细
    # tick_label.set_fontweight('bold') # 设置为粗体 ('normal' 为普通)
# 隐藏 X 轴刻度标签 (因为与热图 X 轴对应)
ax_bar_overall.set_xticks([])
# 在 Y 轴方向添加网格线
ax_bar_overall.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.7)
# 设置 X 轴的显示范围，使其与热图对齐
ax_bar_overall.set_xlim(-0.5, len(sorted_features) - 0.5)
# 计算 Y 轴的最大值，并增加一些边距
y_max_overall = overall_contributions_pct.max() if not overall_contributions_pct.empty else 10
if y_max_overall <= 0:
    y_max_overall = 1.0  # 处理贡献全为0的情况
ax_bar_overall.set_ylim(0, y_max_overall * 1.25)
# 遍历每个条形，在其顶部添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()  # 获取条形高度
    if height > 0.01:  # 只为有显著高度的条形添加标签
        # 在条形上方添加文本，格式化为两位小数
        ax_bar_overall.text(
            x_bar_positions[i],
            height * 1.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
            clip_on=True,
        )
# c) 总体类型贡献饼图 (右上角)
# 获取饼图的标签、大小和颜色
pie_labels = list(pie_data.keys())
pie_sizes = list(pie_data.values())
pie_colors = [var_color, te_color]
# 仅当总贡献显著时绘制饼图 (避免绘制一个几乎为空的图)
if sum(pie_sizes) > 0.1:
    # 绘制饼图 (设置为圆环图效果)，labels=None: 不在内部显示标签，autopct=None: 不在内部显示百分比，startangle=90: 起始角度，pctdistance: 内部百分比的显示距离 (这里未使用)，wedgeprops: 设置楔形属性，width 控制圆环宽度，edgecolor 设置边框
    wedges, texts = ax_pie.pie(
        pie_sizes,
        labels=None,
        colors=pie_colors,
        autopct=None,
        startangle=90,
        pctdistance=0.8,
        wedgeprops=dict(width=0.6, edgecolor="w"),
    )
    # 设置引线和外部标签的样式
    kw = dict(arrowprops=dict(arrowstyle="-"), va="center")
    # 遍历每个楔形 (pie slice)
    for i, p in enumerate(wedges):
        # 计算楔形中心角度
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        # 计算引线终点坐标 (在单位圆上)
        y_coord = np.sin(np.deg2rad(ang))
        x_coord = np.cos(np.deg2rad(ang))
        # 根据 x 坐标确定文本的水平对齐方式
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_coord))]
        # 设置引线的连接样式
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        # 仅当楔形大小 > 1% 时添加外部标签
        if pie_sizes[i] > 1:
            # 添加注释文本（标签和百分比）
            ax_pie.annotate(
                f"{pie_labels[i]}\n{pie_sizes[i]:.1f}%",
                xy=(x_coord, y_coord),
                xytext=(1.3 * np.sign(x_coord), 1.4 * y_coord),
                horizontalalignment=horizontalalignment,
                **kw,
                fontsize=16,
                fontweight="bold",
            )
    # 设置坐标轴比例相等，确保饼图是圆形的
    ax_pie.axis("equal")
else:
    # 如果总贡献太小，显示文本提示
    ax_pie.text(0.5, 0.5, "No significant\ncontribution", ha="center", va="center", fontsize=14)
    ax_pie.axis("off")
# d) 每个生态区贡献比例堆叠条形图 (右下角中间)
# 使用动态获取的生态区数量 n_eco 计算 Y 轴位置 (与热图 Y 轴刻度一致)
bar_y_positions = np.arange(n_eco) + 0.5
# 设置水平条形的高度
bar_height = 0.8
# 绘制代表 'Var' 贡献的条形
ax_bar_stacked.barh(
    bar_y_positions,
    stacked_bar_data["Var"],
    color=var_color,
    edgecolor="white",
    label="Var",
    height=bar_height,
)
# 绘制代表 'TE' 贡献的条形，起点（left）为 'Var' 的值，实现堆叠效果
ax_bar_stacked.barh(
    bar_y_positions,
    stacked_bar_data["TE"],
    left=stacked_bar_data["Var"],
    color=te_color,
    edgecolor="white",
    label="TE",
    height=bar_height,
)
# 遍历动态获取的每个生态区，为其堆叠条形添加内部数值标签
for i, eco in enumerate(ecoregions):
    # 获取当前生态区的 Var 和 TE 百分比值
    var_val = stacked_bar_data.loc[eco, "Var"]
    te_val = stacked_bar_data.loc[eco, "TE"]
    # 如果 Var 值大于 5%，在其条形中间添加标签
    if var_val > 5:
        ax_bar_stacked.text(
            var_val / 2,
            bar_y_positions[i],
            f"{var_val:.1f}",
            ha="center",
            va="center",
            color="black",
            fontsize=14,
            weight="bold",
            clip_on=True,
        )
    # 如果 TE 值大于 5%，在其条形中间添加标签 (注意 X 坐标是 Var + TE/2)
    if te_val > 5:
        ax_bar_stacked.text(
            var_val + te_val / 2,
            bar_y_positions[i],
            f"{te_val:.1f}",
            ha="center",
            va="center",
            color="black",
            fontsize=14,
            weight="bold",
            clip_on=True,
        )

# 由于此子图 Y 轴与热图共享，隐藏其 Y 轴刻度线和标签
ax_bar_stacked.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
# 设置 X 轴标签
ax_bar_stacked.set_xlabel("Contribution (%)", fontsize=16, fontweight="bold")
# 遍历 ax_bar_stacked 的 X 轴刻度标签对象
for tick_label in ax_bar_stacked.get_xticklabels():
    # 设置该刻度标签的字体大小
    tick_label.set_fontsize(16)  # 将 14 修改为你需要的大小
    # 设置该刻度标签的字体粗细
    # tick_label.set_fontweight('bold') # 设置为粗体 ('normal' 为普通)
# 设置 X 轴范围为 0 到 105 (略大于 100 以防标签溢出)
ax_bar_stacked.set_xlim(0, 105)
# 反转 Y 轴方向，使其与热图从上到下顺序一致
ax_bar_stacked.invert_yaxis()
# --- 添加图例和手动定位的颜色条 ---
# 打印信息，告知用户正在添加图例和颜色条
print("Adding legend and colorbar...")
# 在嵌套网格的第一个位置 (顶部) 添加子图作为图例的容器
ax_legend_container = fig.add_subplot(gs_nested[0, 0])
# 关闭图例容器的坐标轴显示
ax_legend_container.axis("off")
# 创建图例所需的图形元素 (带边框的彩色矩形)
handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [te_color, var_color, na_color]]
# 创建图例标签
labels = ["TE", "Var", "NA"]
# 添加图例到图例容器中
# title: 图例标题
# loc/bbox_to_anchor: 控制图例位置 (这里使用相对定位)
# frameon=False: 不显示图例边框
# ncol: 图例列数
# handlelength/handleheight: 图例色块大小
feature_legend = ax_legend_container.legend(
    handles,
    labels,
    title="Feature type",
    loc="upper left",
    bbox_to_anchor=(-0.5, 1.2),
    frameon=False,
    fontsize=14,
    title_fontsize=14,
    ncol=1,
    handlelength=1.5,
    handleheight=1.5,
)

# 使用 try-except 块处理手动放置颜色条可能出现的错误
try:
    # 获取嵌套网格的基准位置和图例容器的位置，用于计算颜色条位置
    base_pos = gs_main[1, 2].get_position(fig)  # 获取 gs_main[1,2] 单元格的位置
    legend_pos = ax_legend_container.get_position()  # 获取图例容器的位置
    # 定义颜色条相对于基准位置的偏移、宽度比例、高度比例和底部边距
    cbar_left_offset = 0.25
    cbar_width_ratio = 0.15
    cbar_height_ratio = 0.6
    cbar_bottom_margin = 0.1
    # 计算颜色条的绝对左侧位置、宽度
    cbar_left = base_pos.x0 + base_pos.width * cbar_left_offset + 0.08  # 微调增加间距
    cbar_width = base_pos.width * cbar_width_ratio
    # 计算颜色条可用的垂直空间 (图例底部到基准位置底部) 和实际高度
    available_height = legend_pos.y0 - base_pos.y0
    cbar_height = available_height * cbar_height_ratio * 1.85  # 调整高度比例因子
    # 计算颜色条的绝对底部位置
    cbar_bottom = base_pos.y0 + available_height * cbar_bottom_margin
    # 在计算好的绝对位置创建一个新的 Axes 对象用于放置颜色条
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    # 创建一个归一化对象，将数据值映射到 [0, 1] 范围
    norm = plt.Normalize(vmin=vmin_real, vmax=vmax_real)
    # 创建一个 ScalarMappable 对象，它将归一化的数据映射到颜色
    sm = plt.cm.ScalarMappable(cmap=cmap_heatmap, norm=norm)
    # 必须设置一个空数组，否则可能报错
    sm.set_array([])
    # 在指定的 Axes (cax) 中绘制颜色条
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    # 设置颜色条的标签
    cbar.set_label(
        "Mean Absolute SHAP value", fontsize=16, labelpad=10, fontweight="bold", rotation=270
    )
    # 设置颜色条刻度标签的字体大小
    cbar.ax.tick_params(labelsize=16)
    # 设置颜色条刻度的定位器 (自动选择最多 5 个刻度)
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    # 更新刻度
    cbar.update_ticks()
# 捕获放置颜色条时可能发生的任何异常
except Exception as e:
    print(f"  Error during manual colorbar placement: {e}")

# --- 最终调整和显示 ---
# 打印信息，告知用户正在进行最终调整并显示图形
print("Final adjustments and display...")
# 定义包含主要绘图区域的 Axes 列表
main_plot_axes = [ax_bar_overall, ax_pie, ax_heatmap, ax_bar_stacked]
# 遍历主要 Axes
for ax in main_plot_axes:
    # 确保 Axes 对象存在
    if ax is not None:
        # 遍历每个 Axes 的边框 (spines)
        for spine in ax.spines.values():
            # 设置边框可见
            spine.set_visible(True)
            # 设置边框颜色为黑色
            spine.set_edgecolor("black")
            # 设置边框线宽
            spine.set_linewidth(1.0)
# 设置图形的自动约束布局的填充参数 (调整子图间的间距)
fig.set_constrained_layout_pads(w_pad=0.005, h_pad=0.005, hspace=0.005, wspace=0.005)
# 尝试启用自动约束布局来优化元素间距，防止重叠
fig.set_constrained_layout(True)
plt.savefig(str(OUTPUT_DIR / "shap.png"), dpi=1080, bbox_inches="tight")
# 显示最终绘制好的图形窗口
plt.close("all")  # Interactive display removed; assets were exported above.
print("\nProcess complete.")

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

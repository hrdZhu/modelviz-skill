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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import shap
import xgboost
from matplotlib.cm import ScalarMappable

matplotlib.use("Agg")
from sklearn.model_selection import GridSearchCV

# ===================================================================
# 全局字体设置
# ===================================================================
# 设置全局默认字体族为'serif'（衬线字体族）
plt.rcParams["font.family"] = "serif"
# 指定'serif'字体族具体使用'Times New Roman'字体
plt.rcParams["font.serif"] = ["Times New Roman"]
# 解决在使用非Unicode字体时，坐标轴负号显示为方框的问题
plt.rcParams["axes.unicode_minus"] = False
# 设置坐标轴线的宽度为3
plt.rcParams["axes.linewidth"] = 3
# 设置X轴主刻度的宽度为3
plt.rcParams["xtick.major.width"] = 3
# 设置Y轴主刻度的宽度为3
plt.rcParams["ytick.major.width"] = 3
# 设置X轴次刻度的宽度为1.0
plt.rcParams["xtick.minor.width"] = 3.0
# 设置Y轴次刻度的宽度为1.0
plt.rcParams["ytick.minor.width"] = 1.0
# 加载数据
excel_path = str(DATA_DIR / "simulated_data.xlsx")
data_df = pd.read_excel(excel_path)
print(f"成功从 '{excel_path}' 加载数据。")
# 目标变量
target_column_name = "Target_Variable"
# 检查目标列是否存在于数据中
if target_column_name not in data_df.columns:
    print(f"错误：指定的目标列 '{target_column_name}' 在Excel文件中不存在！")
    print(f"文件中包含的列有: {data_df.columns.tolist()}")
# 根据目标列名，分离特征(X)和目标(y)
y = data_df[target_column_name]
X = data_df.drop(columns=[target_column_name])
# 从加载的数据中自动获取特征名称列表
feature_names = X.columns.tolist()
print(f"已识别目标变量: '{target_column_name}'")
print(f"已识别 {len(feature_names)} 个特征变量。")
# 建立和训练XGBoost模型
# ------------------------------------
# 定义要搜索的超参数网格
param_grid = {
    "n_estimators": [100, 200],  # 树的数量
    # 'max_depth': [3, 5, 7],               # 树的最大深度
    # 'learning_rate': [0.05, 0.1, 0.2],    # 学习率
    # 'subsample': [0.8, 1.0]               # 训练每棵树时样本的采样比例
}
# 初始化一个XGBoost回归模型
xgb_reg = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
# 设置GridSearchCV，使用5折交叉验证
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=2,
)
print("\n正在进行超参数搜索和交叉验证...")
# 使用模拟数据进行网格搜索
grid_search.fit(X, y)
# 获取搜索到的最佳模型
best_model = grid_search.best_estimator_
print("超参数搜索完成。")
print(f"找到的最佳参数: {grid_search.best_params_}")
print("已使用最佳参数设置模型，准备进行SHAP分析。")
# 使用最佳模型进行SHAP分析
model = best_model
# 进行SHAP分析
# 创建一个TreeExplainer对象，用于解释基于树的模型
explainer = shap.TreeExplainer(model)
# 计算数据集中每个样本每个特征的SHAP值
shap_values = explainer(X)
# 计算每个特征的平均绝对SHAP值，作为特征重要性的度量
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
# 将平均绝对SHAP值和特征名称组合成一个pandas Series
shap_series = pd.Series(mean_abs_shap, index=X.columns)
# 按照原始feature_names的顺序重新索引Series
shap_series = shap_series.reindex(feature_names)
# 按SHAP值降序排列特征
shap_series.sort_values(ascending=False, inplace=True)
# 打印SHAP分析完成和排序的消息
print("SHAP分析完成，特征重要性已排序。")
# 准备绘图所需的数据和颜色
# 获取排序后的特征名称列表
sorted_features = shap_series.index.tolist()
# 获取排序后的平均绝对SHAP值数组
sorted_shap_values = shap_series.values
# 创建一个自定义的线性分段颜色映射（从蓝色到靛蓝再到红色）
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
# 这里的归一化操作是安全且必要的，它仅用于计算颜色
# 创建一个归一化对象，将SHAP值映射到0-1范围，用于颜色映射
color_norm = mcolors.Normalize(vmin=sorted_shap_values.min(), vmax=sorted_shap_values.max())
# 根据归一化后的SHAP值和颜色映射表，为每个特征条形图生成颜色
colors = cmap(color_norm(sorted_shap_values))
# 打印原始SHAP值的前5行前3列
print(pd.DataFrame(shap_values.values[:5, :3], columns=X.columns[:3]).round(4))
# 打印解释信息
print("\n2. 第一个图条形长度所使用的'平均绝对SHAP值' (sorted_shap_values):")
# 打印用于条形图长度的排序后平均绝对SHAP值，并四舍五入到4位小数
print(np.round(sorted_shap_values, 4))
# 开始绘图
# 创建一个16x15英寸的画布
fig = plt.figure(figsize=(16, 15))
# 定义画布的边距
left_margin, right_margin, bottom_margin, top_margin = 0.08, 0.08, 0.12, 0.12
# 定义图与图之间的间距
space_between_plots = 0.04
# 定义颜色条的宽度
colorbar_width = 0.02
# 定义主图的底部位置
plot_bottom = bottom_margin
# 定义主图的高度
plot_height = 1.0 - bottom_margin - top_margin
# 定义颜色条的左侧位置
cbar_left = left_margin
# 定义主坐标轴的左侧位置
main_ax_left = cbar_left + colorbar_width + space_between_plots
# 定义主坐标轴的宽度
main_ax_width = 1.0 - main_ax_left - right_margin
# 在画布上添加颜色条的坐标轴
ax_cbar = fig.add_axes([cbar_left, plot_bottom, colorbar_width, plot_height])
# 在画布上添加主条形图的坐标轴
ax_bar = fig.add_axes([main_ax_left, plot_bottom, main_ax_width, plot_height])
# 创建一个ScalarMappable对象，用于将数据值映射到颜色
sm = ScalarMappable(cmap=cmap, norm=color_norm)
# 在指定的坐标轴(ax_cbar)上绘制颜色条
cbar = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
# 设置颜色条的标签为空，但保留空间
cbar.set_label("", size=18, labelpad=15)
# 移除颜色条上的刻度
cbar.set_ticks([])
# 将颜色条刻度位置设置在左侧
cbar.ax.yaxis.set_ticks_position("left")
# 在颜色条顶部添加'High'文本
ax_cbar.text(0.5, 1.01, "High", transform=ax_cbar.transAxes, ha="center", va="bottom", fontsize=24)
# 在颜色条底部添加'Low'文本
ax_cbar.text(0.5, -0.01, "Low", transform=ax_cbar.transAxes, ha="center", va="top", fontsize=24)
# 隐藏颜色条的边框
cbar.outline.set_visible(False)
# 在颜色条左侧添加旋转的文本标签'Feature Contribution Value'
ax_cbar.text(
    -1.4,
    0.5,
    "Contribution for CEs ($10^4$ t)",
    transform=ax_cbar.transAxes,
    fontsize=24,
    rotation=90,
    va="center",
)
# 将主图的X轴刻度设置在底部
ax_bar.xaxis.tick_bottom()
# 将主图的X轴标签设置在底部位置
ax_bar.xaxis.set_label_position("bottom")
# 反转主图的X轴方向
ax_bar.invert_xaxis()
# 绘制水平条形图
ax_bar.barh(y=range(len(sorted_features)), width=sorted_shap_values, color=colors, height=0.6)
# 反转主图的Y轴方向，使最重要的特征在顶部
ax_bar.invert_yaxis()
# 设置主图的X轴标签
ax_bar.set_xlabel("Contribution for CEs ($10^4$ t)", size=24, labelpad=20)
# 移除主图的Y轴刻度
ax_bar.set_yticks([])
# 隐藏左边和顶部的坐标轴脊(spines)
ax_bar.spines[["left", "top"]].set_visible(False)
# 将右侧的坐标轴脊移动到数据值为0的位置
ax_bar.spines["right"].set_position(("data", 0))
# 显示右侧的坐标轴脊
ax_bar.spines["right"].set_visible(True)
# 显示底部的坐标轴脊
ax_bar.spines["bottom"].set_visible(True)
# 设置X轴主刻度的样式
ax_bar.tick_params(axis="x", which="major", direction="in", labelsize=24, length=6, pad=8)
# 自动设置X轴的次刻度定位器
ax_bar.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
# 设置X轴次刻度的样式
ax_bar.tick_params(axis="x", which="minor", direction="in", length=4)
# 定义特征标签在X轴方向上的内边距
label_x_padding = 0.005
# 遍历排序后的特征并在图上添加文本标签
for i, feature in enumerate(sorted_features):
    ax_bar.text(label_x_padding, i, feature, ha="right", va="center", color="black", fontsize=24)
# 在主图的左上角添加子图标签'(a)'
ax_bar.text(
    0.02, 0.98, "(a)", transform=ax_bar.transAxes, fontsize=30, weight="bold", ha="left", va="top"
)
# 计算径向图（内嵌图）的左侧位置
inset_left = main_ax_left - 0.15
# 计算径向图的底部位置
inset_bottom = plot_bottom - 0.05
# 计算径向图的大小
inset_size = min(main_ax_width, plot_height) * 0.85
# 定义径向图的位置和大小 [left, bottom, width, height]
inset_ax_rect = [inset_left, inset_bottom, inset_size, inset_size]
# 在画布上添加径向图的坐标轴，并设置为极坐标投影
ax_radial_inset = fig.add_axes(inset_ax_rect, projection="polar")
# 设置径向图背景为透明
ax_radial_inset.patch.set_alpha(0)
# 计算每个特征SHAP值占总和的百分比
percentages = (sorted_shap_values / sorted_shap_values.sum()) * 100
# 根据百分比计算每个扇区的宽度（弧度）
widths = (sorted_shap_values / sorted_shap_values.sum()) * 2 * np.pi
# 获取特征的数量
num_vars = len(sorted_features)
# 定义径向图的初始长度、固定增量和彩色环的宽度
base_length, fixed_increment, colored_ring_width = 3.0, 0.5, 2.0
# 计算每个扇区的总长度
total_lengths = [base_length + i * fixed_increment for i in range(num_vars)]
# 计算内部灰色/白色环的高度
inner_heights = [max(0, tl - colored_ring_width) for tl in total_lengths]
# 定义内部环的交替颜色
inner_colors = ["#EAEAEA", "#FFFFFF"] * (num_vars // 2 + 1)
# 截取所需数量的颜色
inner_colors = inner_colors[:num_vars]
# 设置一个偏移量，让第一个扇区从接近1点钟的位置开始
one_oclock_offset = np.pi / 21
# 计算每个扇区的起始角度（theta）
thetas = np.cumsum([0] + widths[:-1].tolist()) - one_oclock_offset
# 绘制内部的灰色/白色交替环
ax_radial_inset.bar(
    x=thetas,
    height=inner_heights,
    width=widths,
    color=inner_colors,
    align="edge",
    edgecolor="white",
    linewidth=1.5,
)
# 绘制外部的彩色环
ax_radial_inset.bar(
    x=thetas,
    height=[colored_ring_width] * num_vars,
    width=widths,
    bottom=inner_heights,
    color=colors,
    align="edge",
    edgecolor="white",
    linewidth=1.5,
)
# 遍历每个扇区，添加百分比标签
for i in range(num_vars):
    # 计算标签的角度
    label_angle_rad = thetas[i] + widths[i] / 2
    # 计算标签的半径
    label_radius = total_lengths[i] + 0.5
    # 在指定位置添加文本
    ax_radial_inset.text(
        label_angle_rad,
        label_radius,
        f"{percentages[i]:.1f}%",
        ha="center",
        va="center",
        fontsize=18,
    )
# 移除径向图的Y轴（半径轴）标签
ax_radial_inset.set_yticklabels([])
# 移除径向图的X轴（角度轴）标签
ax_radial_inset.set_xticklabels([])
# 隐藏极坐标图的外围圆环
ax_radial_inset.spines["polar"].set_visible(False)
# 隐藏网格线
ax_radial_inset.grid(False)
# 设置0度角（theta=0）在正北方向
ax_radial_inset.set_theta_zero_location("N")
# 设置角度增加的方向为顺时针
ax_radial_inset.set_theta_direction(-1)
# 设置Y轴（半径轴）的范围
ax_radial_inset.set_ylim(0, max(total_lengths) + 2)
# 定义原始组合图的保存路径
original_image_path = str(OUTPUT_DIR / "shap.png")
# 保存图像，设置DPI和裁剪白边
plt.savefig(original_image_path, dpi=208, bbox_inches="tight")
# 显示图像
plt.close("all")  # Interactive display removed; assets were exported above.
print("\n正在使用 `shap.summary_plot` 功能绘制蜂巢摘要图...")
# 为了更好地控制图形属性，可以先创建一个 Figure
# 这不是必需的，但提供了更多灵活性
plt.figure(figsize=(16, 15))
# 直接调用 summary_plot 函数
# shap_values: 之前计算好的SHAP值对象
# X: 原始数据，用于为散点着色
# plot_type="dot": 指定生成蜂巢图 (这是默认值)
# show=False: 阻止图形立即弹出，允许我们进行后续的自定义修改
# cmap=cmap: 使用我们之前定义的自定义颜色映射
shap.summary_plot(shap_values, X, plot_type="dot", show=False, cmap=cmap)
# (可选) 对图形进行微调，例如调整字体大小
# 获取当前坐标轴
ax = plt.gca()
# 设置X轴标签和字体大小
ax.set_xlabel("SHAP Value (impact on model output)", fontsize=18)
# 调整Y轴刻度标签字体大小
ax.tick_params(axis="y", labelsize=16)
# 调整X轴刻度标签字体大小
ax.tick_params(axis="x", labelsize=14)
# 找到并调整颜色条
# plt.gcf() 获取当前图形, .axes 是图形中所有坐标轴的列表
# 颜色条通常是最后一个添加的坐标轴
if len(plt.gcf().axes) > 1:
    # 获取颜色条的坐标轴
    cbar_ax = plt.gcf().axes[-1]
    # 设置颜色条的标签、大小、旋转角度和边距
    cbar_ax.set_ylabel("Feature Value", size=16, rotation=-90, labelpad=20)
    # 调整颜色条刻度标签的字体大小
    cbar_ax.tick_params(labelsize=14)
# 调整整体布局以防标签重叠
plt.tight_layout()
# 保存第二个图（原生蜂巢图）的路径
beeswarm_image_path = str(OUTPUT_DIR / "shap_beeswarm_native.png")
# 保存第二个图
plt.savefig(beeswarm_image_path, dpi=208, bbox_inches="tight")
# 显示第二个图
plt.close("all")  # Interactive display removed; assets were exported above.
plt.figure(figsize=(16, 15))
# 再次调用 summary_plot 函数来生成图形内容
# 设置 show=False 以便我们进行修改
shap.summary_plot(shap_values, X, plot_type="dot", show=False, cmap=cmap)
# 获取第三张图的坐标轴
ax_third_plot = plt.gca()
# 将Y轴的刻度标签设置为空列表，移除所有特征的名称
ax_third_plot.set_yticklabels([])
# 将Y轴的轴标题也设置为空，确保左侧完全干净
ax_third_plot.set_ylabel("")
# 在右上角添加文本标注 (b)
ax_third_plot.text(
    1,
    0.98,
    "(b)",
    transform=ax_third_plot.transAxes,
    fontsize=24,
    fontweight="bold",
    va="top",
    ha="right",
)
# =======================================================
# 设置X轴标签和字体大小
ax_third_plot.set_xlabel("SHAP Value (impact on model output)", fontsize=18)
# 只调整X轴刻度字体
ax_third_plot.tick_params(axis="x", labelsize=14)
# 同样，找到并调整颜色条
if len(plt.gcf().axes) > 1:
    # 获取第三张图的颜色条坐标轴
    cbar_ax_third = plt.gcf().axes[-1]
    # 设置颜色条的标签和样式
    cbar_ax_third.set_ylabel("Feature Value", size=16, rotation=-90, labelpad=20)
    # 调整颜色条刻度标签的大小
    cbar_ax_third.tick_params(labelsize=14)
# 调整整体布局
plt.tight_layout()
# 保存
third_plot_image_path = str(OUTPUT_DIR / "shap_beeswarm_no_labels.png")
plt.savefig(third_plot_image_path, dpi=208, bbox_inches="tight")
# 显示
plt.close("all")  # Interactive display removed; assets were exported above.
# 合并图(a)和图(b)
# ===================================================================
# 在脚本顶部，请确保有这行导入语句
from shap.plots import beeswarm  # 从shap.plots模块中直接导入beeswarm函数

print("\n正在将图(a)和图(b)合并绘制到一张图中 (保持原始样式)...")  # 打印当前操作的提示信息
# 1. 创建一个更宽的新画布，为原始样式提供充足空间
fig_combined = plt.figure(figsize=(34, 25))  # 创建一个宽34英寸、高15英寸的画布对象
# 2. 定义整体布局参数，为两个图分配合理空间
# 全局布局
left_margin = 0.05  # 定义整个画布的左边距为画布宽度的5%
right_margin = 0.05  # 定义整个画布的右边距为画布宽度的5%
bottom_margin = 0.12  # 定义整个画布的底边距为画布高度的12%
top_margin = 0.1  # 定义整个画布的顶边距为画布高度的10%
space_between = 0.01  # 定义两个子图之间的水平间隙为画布宽度的1%
# 计算绘图区域的尺寸
plot_bottom = bottom_margin  # 设置绘图区域的底部位置
plot_height = 1 - bottom_margin - top_margin  # 计算绘图区域的实际高度
total_plot_width = 1 - left_margin - right_margin - space_between  # 计算可用于绘图的总宽度
left_plot_width = total_plot_width * 0.6  # 将总绘图宽度的60%分配给左侧图形
right_plot_width = total_plot_width * 0.4  # 将总绘图宽度的40%分配给右侧图形
# =======================================================
# 绘制左侧图形 (图 a)
# =======================================================
# --- 左图的颜色条 ---
cbar_left = 0.1  # 设置颜色条的左侧起始位置
colorbar_width = 0.01  # 设置颜色条的宽度
ax_cbar_new = fig_combined.add_axes(
    [cbar_left, plot_bottom, colorbar_width, plot_height]
)  # 在画布上添加用于颜色条的坐标轴
sm = ScalarMappable(cmap=cmap, norm=color_norm)  # 创建一个可映射标量对象，用于将数据值映射到颜色
cbar = fig_combined.colorbar(
    sm, cax=ax_cbar_new, orientation="vertical"
)  # 在指定的坐标轴上绘制垂直方向的颜色条
cbar.set_label("", size=18, labelpad=5)  # 设置颜色条的标签为空白，但保留字体大小和边距设置
cbar.set_ticks([])  # 移除颜色条上的所有刻度
cbar.ax.yaxis.set_ticks_position("left")  # 将颜色条的刻度位置（即使不可见）设置在左侧
ax_cbar_new.text(
    0.5, 1.01, "High", transform=ax_cbar_new.transAxes, ha="center", va="bottom", fontsize=30
)  # 在颜色条顶部添加'High'文本
ax_cbar_new.text(
    0.5, -0.01, "Low", transform=ax_cbar_new.transAxes, ha="center", va="top", fontsize=30
)  # 在颜色条底部添加'Low'文本
cbar.outline.set_visible(False)  # 隐藏颜色条的边框线
ax_cbar_new.text(
    -1.4,
    0.5,
    "Contribution for CEs ($10^4$ t)",
    transform=ax_cbar_new.transAxes,
    fontsize=30,
    rotation=90,
    va="center",
)  # 在颜色条左侧添加旋转90度的文本标签
# --- 左图的主条形图 ---
main_ax_left = cbar_left + colorbar_width + 0.05  # 计算主条形图的左侧起始位置
ax_bar_new = fig_combined.add_axes(
    [main_ax_left, plot_bottom, left_plot_width, plot_height]
)  # 在画布上添加用于主条形图的坐标轴
ax_bar_new.xaxis.tick_bottom()  # 将X轴的刻度线设置在坐标轴底部
ax_bar_new.xaxis.set_label_position("bottom")  # 将X轴的标签设置在坐标轴底部
ax_bar_new.invert_xaxis()  # 反转X轴的方向，使条形图从右向左延伸
ax_bar_new.barh(
    y=range(len(sorted_features)), width=sorted_shap_values, color=colors, height=0.6
)  # 绘制水平条形图
ax_bar_new.invert_yaxis()  # 反转Y轴的方向，使最重要的特征显示在顶部
ax_bar_new.set_xlabel(
    "Contribution for CEs ($10^4$ t)", size=30, labelpad=20
)  # 设置X轴的标签及其字体大小和边距
ax_bar_new.set_yticks([])  # 移除Y轴的所有刻度
ax_bar_new.spines[["left", "top"]].set_visible(False)  # 隐藏左边和顶部的坐标轴脊（边框）
ax_bar_new.spines["right"].set_position(("data", 0))  # 将右侧的坐标轴脊移动到X轴数据为0的位置
ax_bar_new.spines["right"].set_visible(True)  # 显示右侧的坐标轴脊
ax_bar_new.spines["bottom"].set_visible(True)  # 显示底部的坐标轴脊
ax_bar_new.tick_params(
    axis="x", which="major", direction="in", labelsize=30, length=6, pad=8
)  # 设置X轴主刻度的样式
ax_bar_new.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))  # 自动为X轴设置次刻度定位器
ax_bar_new.tick_params(axis="x", which="minor", direction="in", length=4)  # 设置X轴次刻度的样式
label_x_padding = 0.005  # 内边距值，用于文本标签的X轴位置
for i, feature in enumerate(sorted_features):  # 遍历每一个排序后的特征
    # 让文本右对齐到坐标轴
    ax_bar_new.text(
        label_x_padding, i, feature, ha="right", va="center", color="black", fontsize=30
    )  # 在图上添加特征名称文本，并右对齐
ax_bar_new.text(
    0.02,
    0.98,
    "(a)",
    transform=ax_bar_new.transAxes,
    fontsize=30,
    weight="bold",
    ha="left",
    va="top",
)  # 在主图的左上角添加子图标签'(a)'
# --- 左图的内嵌径向图 ---
inset_size = min(left_plot_width, plot_height) * 0.85  # 计算内嵌径向图的尺寸
inset_left = main_ax_left - 0.15  # 计算内嵌图的左侧位置
inset_bottom = plot_bottom - 0.05  # 计算内嵌图的底部位置
inset_ax_rect = [inset_left, inset_bottom, inset_size, inset_size]  # 定义内嵌图的位置和大小
ax_radial_inset_new = fig_combined.add_axes(
    inset_ax_rect, projection="polar"
)  # 在画布上添加一个极坐标投影的内嵌图坐标轴
ax_radial_inset_new.patch.set_alpha(0)  # 设置内嵌图背景为完全透明
ax_radial_inset_new.bar(
    x=thetas,
    height=inner_heights,
    width=widths,
    color=inner_colors,
    align="edge",
    edgecolor="white",
    linewidth=1.5,
)  # 绘制内部的灰色/白色交替环
ax_radial_inset_new.bar(
    x=thetas,
    height=[colored_ring_width] * num_vars,
    width=widths,
    bottom=inner_heights,
    color=colors,
    align="edge",
    edgecolor="white",
    linewidth=1.5,
)  # 绘制外部的彩色环
for i in range(num_vars):  # 遍历每个特征扇区
    label_angle_rad = thetas[i] + widths[i] / 2  # 计算百分比标签的角度
    label_radius = total_lengths[i] + 0.5  # 计算百分比标签的半径
    ax_radial_inset_new.text(
        label_angle_rad,
        label_radius,
        f"{percentages[i]:.1f}%",
        ha="center",
        va="center",
        fontsize=26,
    )  # 在扇区外添加百分比文本标签
ax_radial_inset_new.set_yticklabels([])  # 移除径向图的Y轴（半径轴）刻度标签
ax_radial_inset_new.set_xticklabels([])  # 移除径向图的X轴（角度轴）刻度标签
ax_radial_inset_new.spines["polar"].set_visible(False)  # 隐藏极坐标图的外围圆环
ax_radial_inset_new.grid(False)  # 隐藏极坐标网格线
ax_radial_inset_new.set_theta_zero_location("N")  # 设置0度角（theta=0）在正北方向
ax_radial_inset_new.set_theta_direction(-1)  # 设置角度增加的方向为顺时针
ax_radial_inset_new.set_ylim(0, max(total_lengths) + 2)  # 设置Y轴（半径轴）的范围
# 为右侧的SHAP蜂巢图创建一个新的坐标轴
right_plot_left = main_ax_left + left_plot_width + space_between  # 计算右侧图的左侧起始位置
ax_beeswarm = fig_combined.add_axes(
    [right_plot_left, plot_bottom, right_plot_width, plot_height]
)  # 在画布上添加用于蜂巢图的坐标轴
# 使用我们最终确定的、能正确运行的函数调用
beeswarm(
    shap_values,
    max_display=len(sorted_features),
    ax=ax_beeswarm,
    show=False,
    color=cmap,
    plot_size=None,
)  # 调用beeswarm函数绘图，并指定坐标轴、颜色表且禁用其自动尺寸调整
# 对右侧的图进行定制化修改
ax_beeswarm.set_yticklabels([])  # 移除Y轴的所有刻度标签
ax_beeswarm.set_ylabel("")  # 移除Y轴的轴标题
ax_beeswarm.set_xlabel(
    "SHAP Value (impact on model output)", fontsize=30
)  # 设置X轴的标签及其字体大小
ax_beeswarm.tick_params(axis="x", labelsize=30)  # 设置X轴刻度标签的字体大小
# 在右上角添加标签 '(b)'
ax_beeswarm.text(
    0.98,
    0.98,
    "(b)",  # 在坐标轴内的相对位置(0.98, 0.98)处添加文本
    transform=ax_beeswarm.transAxes,  # 指定坐标系为坐标轴相对坐标系
    fontsize=30,  # 设置字体大小
    fontweight="bold",  # 设置字体粗细为粗体
    va="top",  # 设置垂直对齐方式为顶部对齐
    ha="right",
)  # 设置水平对齐方式为右对齐
# 调整右图自动生成的颜色条
if (
    len(fig_combined.axes) > 3
):  # 检查是否存在由beeswarm图自动生成的颜色条（通常是第4个及以上的坐标轴）
    cbar_ax_right = fig_combined.axes[-1]  # 获取最后一个坐标轴，即beeswarm图的颜色条坐标轴
    cbar_ax_right.set_ylabel(
        "Feature Value", size=30, rotation=270, labelpad=5
    )  # 设置颜色条的标签、字体大小、旋转角度和边距
    cbar_ax_right.tick_params(labelsize=30)  # 设置颜色条刻度标签的字体大小
# =======================================================
# 保存并显示最终的组合图
# =======================================================
combined_image_path = str(
    OUTPUT_DIR / "combined_shap_plot_final_style_preserved.png"
)  # 定义最终组合图的保存路径和文件名
plt.savefig(
    combined_image_path, dpi=208, bbox_inches="tight"
)  # 保存图像，设置分辨率(dpi)并裁剪掉周围多余的白边
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

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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
# --- 1. 从Excel文件加载数据 ---
input_excel_data_path = str(DATA_DIR / "simulated_input_data.xlsx")
data_sheet_name = "模拟特征X"
df_loaded_all_data = pd.read_excel(input_excel_data_path, sheet_name=data_sheet_name)
y_loaded = df_loaded_all_data.iloc[:, 0].values
X_loaded = df_loaded_all_data.iloc[:, 1:].values
feature_names = df_loaded_all_data.columns[1:].tolist()
num_features = X_loaded.shape[1]
num_samples = X_loaded.shape[0]
# 打印数据加载成功的相关信息
print(f"从Excel文件 '{input_excel_data_path}' (工作表: '{data_sheet_name}') 加载数据成功。")
print(f"样本数量: {num_samples}, 特征数量: {num_features}")
print(f"因变量名称 (第一列): {df_loaded_all_data.columns[0]}")
print(f"特征名称: {feature_names}")
# --- 2. 训练机器学习模型 ---
X_train, X_test, y_train, y_test = train_test_split(
    X_loaded, y_loaded, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# --- 3. 使用SHAP分析模型 ---
# 创建一个SHAP解释器，这里使用的是针对树模型的TreeExplainer
explainer = shap.TreeExplainer(model)
# 计算训练集样本的SHAP值矩阵
shap_values_matrix = explainer.shap_values(X_train)
# --- 4. 计算特征重要性并排序 ---
# 计算每个特征的平均绝对SHAP值，作为其重要性的度量
mean_abs_shap_values = np.mean(np.abs(shap_values_matrix), axis=0)
# 获取根据平均绝对SHAP值降序排列的特征索引
sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
# 根据排序后的索引获取排序后的特征名称列表
sorted_feature_names_for_bar = [feature_names[i] for i in sorted_indices]
# 根据排序后的索引获取排序后的平均绝对SHAP值列表
sorted_mean_abs_shap_values_for_bar = mean_abs_shap_values[sorted_indices]
# 创建一个Pandas DataFrame存储特征重要性结果
df_feature_importance = pd.DataFrame(
    {
        "特征名称": sorted_feature_names_for_bar,
        "平均绝对SHAP值": sorted_mean_abs_shap_values_for_bar,
    }
)
# --- 5. 准备颜色映射 ---
# 获取实际用于颜色映射的特征数量
num_actual_features_for_colors = len(feature_names)
# 初始化生成的颜色列表
generated_colors = []
# 如果没有特征，打印警告信息
if num_actual_features_for_colors <= 0:
    print("警告: 没有特征用于颜色映射。")
# 如果特征数量小于等于20，使用tab20颜色映射
elif num_actual_features_for_colors <= 20:
    cmap_obj = plt.get_cmap("tab20")
    generated_colors = list(cmap_obj.colors[:num_actual_features_for_colors])
# 如果特征数量小于等于40，组合tab20b和tab20c颜色映射
elif num_actual_features_for_colors <= 40:
    cmap1_obj = plt.get_cmap("tab20b")
    cmap2_obj = plt.get_cmap("tab20c")
    combined_colors = list(cmap1_obj.colors) + list(cmap2_obj.colors)
    generated_colors = combined_colors[:num_actual_features_for_colors]
# 如果特征数量大于40，使用hsv颜色映射生成颜色
else:
    hsv_cmap = plt.get_cmap("hsv")
    generated_colors = [
        hsv_cmap(i / num_actual_features_for_colors) for i in range(num_actual_features_for_colors)
    ]
# 将生成的颜色列表赋值给colors变量
colors = generated_colors
# --- 6. 创建图形和子图布局 ---
# 用户指定希望绘制的顶部特征数量 (用于散点图)
num_top_features_to_plot_scatter = 6  # 您可以修改这里，例如 6, 8, 10等
# 实际将要创建的散点图数量 (不超过总特征数)
num_scatter_plots_actual = min(num_top_features_to_plot_scatter, num_features)
# 定义散点图排列的行数
scatter_plot_rows = 2
# 计算需要的散点图列数，确保所有散点图都能放下
scatter_plot_cols = (num_scatter_plots_actual + scatter_plot_rows - 1) // scatter_plot_rows
# 如果没有特征可供散点图使用，则不创建散点图列
if num_scatter_plots_actual == 0:
    scatter_plot_cols = 0
# 计算图形的总列数（1列给条形图，其余给散点图）
total_fig_cols = 1 + scatter_plot_cols
# 动态调整图形宽度，保持每个子图大致宽度
base_col_width = 4  # 假设每个散点图子图基础宽度为4英寸
bar_plot_width_ratio = 1.5  # 条形图相对于散点图的宽度比例
# 计算总图形宽度
fig_width = (bar_plot_width_ratio * base_col_width) + (scatter_plot_cols * base_col_width)
# 计算总图形高度
fig_height = scatter_plot_rows * (base_col_width * 0.9)  # 假设高度略小于宽度以接近方形
# 如果只有条形图，重新计算图形尺寸
if num_scatter_plots_actual == 0:
    fig_width = bar_plot_width_ratio * base_col_width * 1.5
    fig_height = 8  # 仅条形图时的默认高度
# 创建一个图形对象，并设置其大小
fig = plt.figure(figsize=(fig_width, fig_height))
# 设置GridSpec的宽度比例
if scatter_plot_cols > 0:
    gs_width_ratios = [bar_plot_width_ratio] + [1] * scatter_plot_cols
else:  # 只有条形图
    gs_width_ratios = [1]
# 创建一个GridSpec对象，用于定义子图的网格布局
gs = gridspec.GridSpec(
    scatter_plot_rows if scatter_plot_cols > 0 else 1,  # 行数
    total_fig_cols if total_fig_cols > 0 else 1,  # 列数
    figure=fig,  # 关联的图形对象
    width_ratios=gs_width_ratios,  # 各列的宽度比例
    height_ratios=[1] * (scatter_plot_rows if scatter_plot_cols > 0 else 1),  # 各行的高度比例
    wspace=0.4,  # 子图之间的水平间距
    hspace=0.55,
)  # 子图之间的垂直间距
# 添加条形图子图；如果无散点图，则条形图占整个网格
ax_bar = fig.add_subplot(gs[:, 0] if scatter_plot_cols > 0 else gs[0, 0])
# 初始化存储散点图子图对象的列表
axes_scatter_flat = []
# 如果有散点图列，则创建并添加散点图子图
if scatter_plot_cols > 0:
    for i in range(num_scatter_plots_actual):
        # 计算当前散点图在网格中的行索引
        r_idx = i // scatter_plot_cols
        # 计算当前散点图在GridSpec中的列索引（条形图占第0列，散点图从第1列开始）
        c_idx_gs = 1 + (i % scatter_plot_cols)
        # 添加散点图子图
        axes_scatter_flat.append(fig.add_subplot(gs[r_idx, c_idx_gs]))
# --- 准备子图注释标签 ---
custom_annotation_labels = []
# 计算可见子图的总数（1个条形图 + N个散点图）
num_total_visible_subplots = 1 + len(axes_scatter_flat)
# 将自定义标签列表赋值给annotation_labels
annotation_labels = custom_annotation_labels
# 如果自定义标签为空或数量不足，则使用默认的 (a), (b), ... 标签
if len(annotation_labels) < num_total_visible_subplots:
    annotation_labels = [f"({chr(ord('a') + k)})" for k in range(num_total_visible_subplots)]
# 设置子图注释的字体大小
annotation_fontsize = 20
# --- 7. 绘制左侧的条形图 ---
# 初始化图例句柄列表
legend_handles = []
# 定义图例中方形标记的大小
legend_marker_size = 10  # 可以调整这个值 (例如 10, 12, 14)
# 遍历排序后的特征名称和SHAP值，绘制条形图
for i in range(len(sorted_feature_names_for_bar)):
    try:
        # 获取当前特征在原始特征列表中的索引，以便获取对应的颜色
        original_index = feature_names.index(sorted_feature_names_for_bar[i])
        # 获取条形图的颜色
        bar_color = colors[original_index] if original_index < len(colors) else "gray"
    # 如果特征名称在原始列表中找不到（理论上不应发生），则使用黑色
    except ValueError:
        bar_color = "black"
    # 绘制水平条形图
    ax_bar.barh(
        sorted_feature_names_for_bar[i],  # y轴：特征名称
        sorted_mean_abs_shap_values_for_bar[i],  # x轴：平均绝对SHAP值
        color=bar_color,
    )  # 条形颜色
    # 只为最重要的前N（这里是8）个特征创建图例项
    if i < 6:
        # 创建一个Line2D对象作为图例标记（方形）
        handle = plt.Line2D(
            [0],
            [0],  # 虚拟数据点
            marker="s",  # 标记形状：'s'代表方形
            linestyle="None",  # 不显示线条
            markersize=legend_marker_size,  # 标记大小
            markerfacecolor=bar_color,  # 方形填充颜色
            markeredgecolor=bar_color,  # 方形边框颜色（与填充色相同，实现实心方块）
            label=sorted_feature_names_for_bar[i],
        )  # 图例标签
        # 将创建的图例句柄添加到列表中
        legend_handles.append(handle)
# 反转y轴，使得最重要的特征显示在图表顶部
ax_bar.invert_yaxis()
# 设置x轴标签及其字体大小
ax_bar.set_xlabel("Mean(|SHAP| value)", fontsize=20)
# 设置y轴刻度标签的字体大小
ax_bar.tick_params(axis="y", labelsize=20)
# 设置x轴刻度标签的字体大小
ax_bar.tick_params(axis="x", labelsize=20)
# 添加图例
ax_bar.legend(
    handles=legend_handles,  # 图例句柄
    title="",  # 图例标题（空）
    fontsize=16,  # 图例文字字体大小
    title_fontsize=11,  # 图例标题字体大小（此处无标题）
    frameon=False,  # 不显示图例边框
    handletextpad=0.02,  # 图例标记与文字之间的水平间隔
    labelspacing=0.02,
)  # 不同图例条目之间的垂直间隔
# 移除条形图的网格线
ax_bar.grid(False)
# 加粗条形图的图框（坐标轴线）
for spine in ax_bar.spines.values():
    spine.set_linewidth(1.5)
# 添加条形图的左上角注释
if annotation_labels:
    ax_bar.text(
        -0.45,
        1,
        annotation_labels[0],
        transform=ax_bar.transAxes,  # 文本内容及相对坐标
        fontsize=annotation_fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )  # 字体样式


# 定义二次多项式函数，用于后续的曲线拟合
def polynomial_func(x, a, b, c):
    return a * x**2 + b * x + c


# --- 9. 绘制右侧的散点图 ---
# 循环遍历已创建的散点图子图对象列表
for i in range(len(axes_scatter_flat)):
    # 获取当前散点图子图对象
    current_ax = axes_scatter_flat[i]
    # 获取当前要绘制的特征在排序后索引列表中的原始索引
    original_feature_index_to_plot = sorted_indices[i]
    # 获取当前要绘制的特征的名称
    feature_name_to_plot = feature_names[original_feature_index_to_plot]
    # 获取当前特征对应的颜色
    feature_color_for_plot = (
        colors[original_feature_index_to_plot]
        if original_feature_index_to_plot < len(colors)
        else "gray"
    )
    # 获取当前特征的原始值（来自训练集X_train）
    x_data_for_plot = X_train[:, original_feature_index_to_plot]
    # 获取当前特征对应的SHAP值（来自shap_values_matrix）
    y_data_shap_for_plot = shap_values_matrix[:, original_feature_index_to_plot]
    # 绘制散点图：x轴为特征值，y轴为SHAP值
    current_ax.scatter(
        x_data_for_plot,
        y_data_shap_for_plot,
        color=feature_color_for_plot,
        alpha=0.5,
        s=30,
        label="SHAP 值",
    )
    # 在SHAP值为0的位置绘制一条水平虚线
    current_ax.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    # 尝试进行非线性拟合并绘制拟合曲线及置信区间
    try:
        # 确保用于拟合的x数据是排序的，以便fill_between正确工作
        sort_indices_fit = np.argsort(x_data_for_plot)
        x_data_for_fit_sorted = x_data_for_plot[sort_indices_fit]
        y_data_shap_for_fit_sorted = y_data_shap_for_plot[sort_indices_fit]
        # 使用curve_fit进行二次多项式拟合
        popt, pcov = curve_fit(
            polynomial_func, x_data_for_fit_sorted, y_data_shap_for_fit_sorted, maxfev=10000
        )  # 增加最大迭代次数
        # 生成用于绘制拟合曲线的x值序列
        x_fit = np.linspace(x_data_for_fit_sorted.min(), x_data_for_fit_sorted.max(), 100)
        # 计算拟合曲线对应的y值
        y_fit = polynomial_func(x_fit, *popt)
        # 绘制拟合曲线
        current_ax.plot(
            x_fit,
            y_fit,
            color=feature_color_for_plot,
            linewidth=2,
            linestyle="-",
            label="非线性拟合",
        )
        # 计算拟合参数的标准误差
        perr = np.sqrt(np.diag(pcov))
        # 尝试计算并绘制95%置信区间（近似）
        try:
            # 计算置信区间的上界和下界对应的y值
            y_fit_upper_approx = polynomial_func(x_fit, *(popt + 1.96 * perr))
            y_fit_lower_approx = polynomial_func(x_fit, *(popt - 1.96 * perr))
            # 确保下界总是小于上界
            lower_bound = np.minimum(y_fit_upper_approx, y_fit_lower_approx)
            upper_bound = np.maximum(y_fit_upper_approx, y_fit_lower_approx)
            # 填充置信区间区域
            current_ax.fill_between(
                x_fit,
                lower_bound,
                upper_bound,
                color=feature_color_for_plot,
                alpha=0.15,
                label="95% 置信区间 (近似)",
            )
        # 如果计算或绘制置信区间时发生异常，打印错误信息
        except Exception as e_ci:
            print(f"无法为特征 '{feature_name_to_plot}' 计算或绘制置信区间: {e_ci}")
    # 如果拟合过程中发生RuntimeError（通常是拟合失败），打印错误信息并在图中标注
    except RuntimeError as e_fit:
        print(f"无法为特征 '{feature_name_to_plot}' 进行拟合: {e_fit}")
        current_ax.text(
            0.5,
            0.5,
            "拟合失败",
            horizontalalignment="center",
            verticalalignment="center",
            transform=current_ax.transAxes,
            fontsize=20,
            color="red",
            fontweight="bold",
        )
    # 设置x轴标签为当前特征名称，并设置字体大小
    current_ax.set_xlabel(f"{feature_name_to_plot}", fontsize=20)
    # 设置y轴标签为'SHAP 值'，并设置字体大小
    current_ax.set_ylabel("SHAP value", fontsize=20)
    # 移除散点图的网格线
    current_ax.grid(False)
    # 设置坐标轴刻度标签的字体大小
    current_ax.tick_params(axis="both", which="major", labelsize=20)
    # 将y轴刻度标签格式化为保留一位小数
    current_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # 加粗散点图的图框（坐标轴线）
    for spine in current_ax.spines.values():
        spine.set_linewidth(1.5)
    # 尝试使子图的绘图区域（box）为正方形
    current_ax.set_box_aspect(1)
    # 添加散点图的左上角注释
    if 1 + i < len(annotation_labels):  # 检查注释标签是否存在
        current_ax.text(
            -0.45,
            1.0,
            annotation_labels[1 + i],
            transform=current_ax.transAxes,  # 文本内容及相对坐标
            fontsize=annotation_fontsize,
            fontweight="bold",
            va="top",
            ha="left",
        )  # 字体样式
plt.savefig(str(OUTPUT_DIR / "1.png"), dpi=1080, bbox_inches="tight")
# 显示图形
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

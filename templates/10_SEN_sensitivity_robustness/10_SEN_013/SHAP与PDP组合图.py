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

import os  # 导入 os 模块，用于处理文件路径等操作系统相关功能

import matplotlib  # 导入 matplotlib 库
import matplotlib.gridspec as gridspec  # 从 matplotlib 导入 gridspec，用于自定义子图布局
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算
import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import shap  # 导入 shap 库，用于计算 SHAP 值解释模型
from sklearn.ensemble import (
    RandomForestRegressor,  # 从 scikit-learn 导入 RandomForestRegressor 模型
)
from sklearn.inspection import (
    partial_dependence,  # 从 scikit-learn 导入 partial_dependence 用于计算部分依赖图
)
from sklearn.model_selection import (
    train_test_split,  # 从 scikit-learn 导入 train_test_split 用于划分数据集
)
from sklearn.utils import resample  # 从 scikit-learn 导入 resample 用于自助采样

matplotlib.use("Agg")  # 设置 matplotlib 使用 TkAgg 后端
# --- 全局字体设置 ---
plt.rcParams["font.family"] = "serif"  # 设置全局字体族为衬线字体
plt.rcParams["font.serif"] = "Times New Roman"  # 设置衬线字体为 Times New Roman
plt.rcParams["axes.unicode_minus"] = False  # 设置正常显示负号
# --- 设置数据文件路径 ---
# !!! 用户注意：这里的路径是您指定要加载的CSV文件的路径 !!!
data_filepath = str(DATA_DIR / "simulated_data.csv")  # 定义要加载的模拟数据文件的完整路径
target_column_name = "target_variable"  # 在CSV文件中，目标变量的列名
# --- 从指定的CSV文件加载数据 ---
loaded_data_df = pd.read_csv(data_filepath)  # 读取CSV文件到DataFrame
print(f"数据加载成功。形状: {loaded_data_df.shape}")  # 打印加载数据的维度信息
print(f"找到的列: {loaded_data_df.columns.tolist()}")  # 打印所有列名
# --- 从加载的数据中定义特征 (X) 和目标 (y) ---
y = loaded_data_df[target_column_name]  # 提取目标变量y
X = loaded_data_df.drop(columns=[target_column_name])  # 提取特征X (除去目标变量列的所有其他列)
print(f"特征 (X) 形状: {X.shape}, 目标 (y) 形状: {y.shape}")  # 打印X和y的形状
# --- 定义/确认特征元数据 (应与CSV文件中的特征列匹配) ---
real_feature_names = X.columns.tolist()  # 直接从加载的X获取特征名称列表
# --- 新增：为每个特征生成唯一的颜色 ---
num_total_features = len(real_feature_names)
# 使用 nipy_spectral 调色板为每个特征生成不同颜色
# 如果特征数量较少（例如 < 20），也可以考虑 tab20 等其他调色板
try:
    cmap = plt.colormaps.get_cmap("nipy_spectral")
    colors_list = [
        cmap(i / float(num_total_features - 1)) if num_total_features > 1 else cmap(0.5)
        for i in range(num_total_features)
    ]
except AttributeError:
    import matplotlib.cm as cm

    colors_list = [
        cm.nipy_spectral(i / float(num_total_features - 1))
        if num_total_features > 1
        else cm.nipy_spectral(0.5)
        for i in range(num_total_features)
    ]
feature_color_palette = dict(zip(real_feature_names, colors_list))
print(f"已为 {num_total_features} 个特征生成了唯一的颜色。")
# 确保 categories_for_names 长度与 real_feature_names 匹配，如果CSV列与预期不符
units_for_names = {  # 定义每个特征对应的单位字典
    "Avg Temperature": "°C",
    "Max Temperature": "°C",
    "Min Temperature": "°C",
    "Precipitation": "mm",
    "Solar Radiation": "W/m²",
    "Wind Speed": "m/s",
    "Humidity": "%",
    "Evapotranspiration": "mm",
    "Forest Cover": "%",
    "Urban Area": "%",
    "Cropland": "%",
    "Grassland": "%",
    "Water Body": "%",
    "Soil Organic Carbon": "g/kg",
    "Soil pH": "",
    "Clay Content": "%",
    "Silt Content": "%",
    "Sand Content": "%",
    "Soil Depth": "cm",
    "Elevation": "m",
    "Slope": "°",
    "Aspect": "°",
    "Curvature": "",
    "TWI": "",
    "Roughness": "",
}
if len(real_feature_names) != X.shape[1]:  # 再次确认特征数量是否匹配
    print(
        f"警告: 加载的特征数量 ({X.shape[1]}) 与 'real_feature_names' 列表长度 ({len(real_feature_names)}) 不匹配。"
    )
feature_to_unit_map = units_for_names  # 特征名到单位的映射字典
# --- 划分数据并训练模型 ---
print("划分数据并训练 RandomForestRegressor 模型...")  # 打印信息：开始划分数据和训练模型
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # 将数据集划分为训练集和测试集
model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5
)  # 初始化随机森林回归模型
model.fit(X_train, y_train)  # 使用训练数据拟合模型
print("模型训练完成。")  # 打印模型训练完成的消息
# --- 计算 SHAP 值 (Panel a) ---
print("使用 TreeExplainer 计算 SHAP 值...")  # 打印信息：开始计算 SHAP 值
explainer = shap.TreeExplainer(model)  # 创建 TreeExplainer 对象
shap_values = explainer.shap_values(X_test)  # 计算测试集的 SHAP 值
print(f"SHAP 值计算完成。形状: {shap_values.shape}")  # 打印 SHAP 值的形状
mean_abs_shap = np.abs(shap_values).mean(axis=0)  # 计算每个特征的平均绝对 SHAP 值
shap_std = np.abs(shap_values).std(axis=0)  # 计算每个特征绝对 SHAP 值的标准差
shap_summary = pd.DataFrame(
    {  # 创建 SHAP 值摘要的 DataFrame
        "feature": X_test.columns,  # 特征名称（来自测试集列名）
        "mean_abs_shap": mean_abs_shap,  # 平均绝对 SHAP 值
        "shap_std_dev": shap_std,  # SHAP 值的标准差
    }
)
shap_summary = shap_summary.sort_values(by="mean_abs_shap", ascending=False).reset_index(
    drop=True
)  # 按重要性排序
print("SHAP 值摘要 (前5个):")  # 打印摘要信息
print(shap_summary.head())  # 显示前5个最重要的特征
features_plot_a_names_only = shap_summary["feature"].tolist()  # 获取用于图a的特征名称列表（已排序）
mean_shap_plot_a = shap_summary["mean_abs_shap"].values  # 获取对应的平均绝对 SHAP 值
ci_half_width_plot_a = shap_summary["shap_std_dev"].values  # 获取对应的标准差
features_plot_a_display = [  # 创建用于图a显示的特征名称列表（包含单位）
    f"{name} ({feature_to_unit_map.get(name, '')})" if feature_to_unit_map.get(name) else name
    for name in features_plot_a_names_only
]
print("图a的数据已准备好。")  # 打印图a数据准备完成的消息
# --- 计算部分依赖图 (PDP) 数据 (Panel b) ---
print("计算主要特征的部分依赖数据...")  # 打印信息：开始计算 PDP 数据
N_top_features = 6  # 定义计算 PDP 的顶部特征数量
top_features_for_pdp = (
    shap_summary["feature"].head(N_top_features).tolist()
)  # 获取 SHAP 排序后的顶部特征名称 (这些是原始特征名)
top_features_data_real = []  # 初始化列表以存储 PDP 数据
n_bootstrap_pdp = 25  # PDP 的自助采样次数 (可适当减小以加速测试)
n_pdp_points = 50  # PDP 曲线的点数
for i, feature_name_actual in enumerate(top_features_for_pdp):  # 遍历顶部特征
    # feature_category = feature_to_category_map.get(feature_name_actual, 'general') # 特征类别 (颜色不再基于此)
    feature_unit = feature_to_unit_map.get(feature_name_actual, "")  # 获取特征单位
    feature_name_with_unit_display = (
        f"{feature_name_actual} ({feature_unit})" if feature_unit else feature_name_actual
    )  # 带单位的特征显示名
    print(
        f"处理PDP特征: {feature_name_with_unit_display} ({i + 1}/{N_top_features})..."
    )  # 打印当前处理的PDP特征
    bootstrap_lines = []  # 存储自助采样的PDP线
    pdp_values = None  # PDP的x轴值
    current_X_for_pdp = X_test  # PDP基于测试集计算
    for j in range(n_bootstrap_pdp):  # 执行自助采样
        X_sample_df = resample(
            current_X_for_pdp, n_samples=len(current_X_for_pdp), random_state=42 + j
        )  # 自助采样，保持DataFrame结构
        try:  # 尝试计算PDP
            pdp_result = partial_dependence(  # 计算部分依赖
                model,  # 模型
                X_sample_df,  # 采样数据
                features=[feature_name_actual],  # 当前特征名 (字符串)
                kind="average",  # 计算平均效果
                grid_resolution=n_pdp_points,  # 网格点数
                response_method="auto",  # 自动响应方法
            )
            current_pdp_line = pdp_result["average"][0]  # 获取PDP y值
            bootstrap_lines.append(current_pdp_line)  # 添加到列表
            if pdp_values is None:  # 仅从第一次成功计算中获取x值
                pdp_values = pdp_result["values"][0]  # 获取PDP x值
        except Exception as e:  # 如果计算失败
            print(
                f"警告: 特征 {feature_name_actual} 的第 {j} 次自助采样PDP计算失败。错误: {e}"
            )  # 打印警告
            continue  # 继续下一次采样
    if not bootstrap_lines:  # 如果没有成功的自助采样线
        print(
            f"错误: 无法为 {feature_name_actual} 计算任何 PDP 自助采样线。跳过此特征。"
        )  # 打印错误
        continue  # 跳过此特征
    if pdp_values is None:  # 如果没有获取到x轴值
        print(f"错误: 无法为 {feature_name_actual} 获取 PDP x轴值。跳过此特征。")  # 打印错误
        continue  # 跳过此特征

    average_line = np.mean(bootstrap_lines, axis=0)  # 计算平均PDP线
    top_features_data_real.append(
        {  # 存储PDP结果
            "name": feature_name_with_unit_display,  # 显示名称
            "original_name": feature_name_actual,  # 存储原始特征名，用于获取颜色
            # 'category': feature_category, # 类别 (颜色不再基于此)
            "x": pdp_values,  # x值
            "bootstrap": bootstrap_lines,  # 自助采样线
            "average": average_line,  # 平均线
        }
    )

print("图b的部分依赖数据已准备好。")  # 打印图b数据准备完成的消息

# --- 绘制图表 ---
print("生成图表...")  # 打印信息：开始生成图表
plt.style.use("seaborn-v0_8-ticks")  # 使用seaborn风格
fig = plt.figure(figsize=(14, 9))  # 创建图形
gs = gridspec.GridSpec(
    3,
    3,
    figure=fig,  # 定义GridSpec布局
    width_ratios=[1.5, 1, 1],  # 列宽比例
    wspace=0.25,  # 列间距
    hspace=0.6,
)  # 行间距
# --- Panel a: 特征贡献 (SHAP) ---
ax_a = fig.add_subplot(gs[:, 0])  # SHAP图占据第一列
n_features_a = len(features_plot_a_display)  # 获取SHAP图中特征数量
y_pos_a = np.arange(n_features_a)  # SHAP图中y轴位置
for i in range(n_features_a):  # 遍历每个特征绘制误差棒
    idx_a = n_features_a - 1 - i  # 反向索引，使最重要的在顶部
    feature_name_for_color_a = features_plot_a_names_only[idx_a]  # 获取原始特征名以查找颜色
    color_a = feature_color_palette.get(
        feature_name_for_color_a, "black"
    )  # 从调色板获取特征特定颜色
    ax_a.errorbar(
        mean_shap_plot_a[idx_a],
        y_pos_a[i],
        xerr=ci_half_width_plot_a[idx_a],  # 绘制误差棒
        fmt="o",
        color=color_a,
        ecolor=color_a,
        capsize=3,
        lw=1,
        markersize=5,
        elinewidth=1,
    )

ax_a.set_yticks(y_pos_a)  # 设置y轴刻度
ax_a.set_yticklabels(features_plot_a_display[::-1], fontsize=9)  # 设置y轴标签（反转以匹配绘图）
ax_a.set_xlabel("mean |SHAP value| (impact on model output)", fontsize=11)  # 设置x轴标签

valid_means_a = mean_shap_plot_a[np.isfinite(mean_shap_plot_a)]  # 有效的平均SHAP值
if len(valid_means_a) > 0:  # 如果存在有效值
    upper_bounds_a = mean_shap_plot_a + ci_half_width_plot_a  # 计算误差棒上界
    valid_upper_bounds_a = upper_bounds_a[np.isfinite(upper_bounds_a)]  # 有效的上界值
    max_err_bar_end_a = (
        np.max(valid_upper_bounds_a) if len(valid_upper_bounds_a) > 0 else np.max(valid_means_a)
    )  # 计算x轴最大值
    xlim_max_val_a = max_err_bar_end_a  # x轴最大值的基准
    xlim_max_a = xlim_max_val_a * 1.1 if np.isfinite(xlim_max_val_a) else 1.0  # 增加10%边距
    if xlim_max_a <= 0:
        xlim_max_a = 1.0  # 处理特殊情况
else:  # 如果没有有效值
    xlim_max_a = 1.0  # 默认x轴上限

ax_a.set_xlim(left=-0.05 * xlim_max_a, right=xlim_max_a)  # 设置x轴范围
ax_a.grid(True, axis="x", linestyle=":", color="lightgray", alpha=0.7)  # x轴网格线
ax_a.grid(False, axis="y")  # 不显示y轴网格线

ax_a.spines["top"].set_visible(True)
ax_a.spines["right"].set_visible(True)
ax_a.spines["bottom"].set_visible(True)
ax_a.spines["left"].set_visible(True)

ax_a.tick_params(axis="x", which="major", labelsize=9)  # x轴刻度字体
ax_a.text(
    -0.20, 1.01, "a", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="top", ha="right"
)  # Panel 'a' 标签

# --- Panel b: 部分依赖图 ---
axes_b = []  # PDP子图的Axes对象列表
if not top_features_data_real:  # 如果没有PDP数据
    print("警告: 未能成功计算任何 PDP 数据。图 B 将为空。")
else:  # 如果有PDP数据
    for i_row_b in range(3):  # 遍历PDP子图的行
        for j_col_b in range(2):  # 遍历PDP子图的列
            plot_idx_b = i_row_b * 2 + j_col_b  # 当前PDP子图索引
            if (
                plot_idx_b < len(top_features_data_real) and plot_idx_b < N_top_features
            ):  # 确保不超过数据量
                ax_b = fig.add_subplot(
                    gs[i_row_b, j_col_b + 1]
                )  # 创建PDP子图（从GridSpec的第二列开始）
                axes_b.append(ax_b)  # 添加到列表

all_pdp_y_values_b = []  # 收集所有PDP的y值以统一y轴

for i_b, pdp_data_item_b in enumerate(top_features_data_real):  # 遍历每个特征的PDP数据
    if i_b >= len(axes_b):
        break  # 如果超出已创建的子图，则停止
    ax_b_current = axes_b[i_b]  # 获取当前子图Axes

    if (
        "x" not in pdp_data_item_b or pdp_data_item_b["x"] is None or len(pdp_data_item_b["x"]) == 0
    ):  # 检查x值
        ax_b_current.set_title(
            f"{pdp_data_item_b.get('name', '错误')}\n(X 数据缺失)", fontsize=9, color="red"
        )  # 显示错误
        continue  # 跳过

    x_coords_pdp_b = pdp_data_item_b["x"]  # PDP的x坐标
    valid_bootstrap_lines_for_avg_b = []  # 有效的自助采样线

    if "bootstrap" in pdp_data_item_b and pdp_data_item_b["bootstrap"]:  # 如果有自助采样数据
        for line_data_b in pdp_data_item_b["bootstrap"]:  # 遍历每条线
            if line_data_b is not None and len(x_coords_pdp_b) == len(line_data_b):  # 检查有效性
                ax_b_current.plot(
                    x_coords_pdp_b, line_data_b, color="silver", linewidth=0.8, alpha=0.5
                )  # 绘制自助采样线
                finite_line_data_b = line_data_b[np.isfinite(line_data_b)]  # 有效数据点
                if finite_line_data_b.size > 0:  # 如果存在
                    all_pdp_y_values_b.extend(finite_line_data_b)  # 添加到列表
                    valid_bootstrap_lines_for_avg_b.append(line_data_b)  # 添加到列表

    avg_line_to_plot_b = pdp_data_item_b.get("average")  # 获取平均线
    if (
        avg_line_to_plot_b is None and valid_bootstrap_lines_for_avg_b
    ):  # 如果平均线不存在但有自助采样线
        avg_line_to_plot_b = np.mean(valid_bootstrap_lines_for_avg_b, axis=0)  # 重新计算平均线

    if avg_line_to_plot_b is not None and len(x_coords_pdp_b) == len(
        avg_line_to_plot_b
    ):  # 如果平均线有效
        original_feature_name_for_color_b = pdp_data_item_b["original_name"]  # 获取原始特征名
        avg_line_color_b = feature_color_palette.get(
            original_feature_name_for_color_b, "black"
        )  # 从调色板获取特征特定颜色
        ax_b_current.plot(
            x_coords_pdp_b, avg_line_to_plot_b, color=avg_line_color_b, linewidth=2.0
        )  # 绘制平均线
        finite_avg_line_b = avg_line_to_plot_b[np.isfinite(avg_line_to_plot_b)]  # 有效数据点
        if finite_avg_line_b.size > 0:  # 如果存在
            all_pdp_y_values_b.extend(finite_avg_line_b)  # 添加到列表
    else:  # 如果无法绘制平均线
        ax_b_current.text(
            0.5,
            0.5,
            "平均线缺失",
            horizontalalignment="center",  # 显示提示
            verticalalignment="center",
            transform=ax_b_current.transAxes,
            color="red",
            fontsize=8,
        )

    ax_b_current.set_title(pdp_data_item_b["name"], fontsize=10)  # 设置子图标题
    ax_b_current.grid(True, linestyle=":", color="lightgray", alpha=0.7)  # 网格线
    ax_b_current.spines["top"].set_visible(False)  # 隐藏上边框
    ax_b_current.spines["right"].set_visible(False)  # 隐藏右边框
    ax_b_current.tick_params(axis="both", which="major", labelsize=8)  # 刻度字体

if all_pdp_y_values_b and axes_b:  # 如果有y值和子图
    y_min_pdp_b, y_max_pdp_b = np.min(all_pdp_y_values_b), np.max(all_pdp_y_values_b)  # 计算y轴范围
    pdp_buffer_b = (
        (y_max_pdp_b - y_min_pdp_b) * 0.1 if (y_max_pdp_b - y_min_pdp_b) > 1e-6 else 0.1
    )  # 计算缓冲区
    final_y_min_pdp_b = y_min_pdp_b - pdp_buffer_b  # 最终y轴最小值
    final_y_max_pdp_b = y_max_pdp_b + pdp_buffer_b  # 最终y轴最大值
    for ax_b_item_b in axes_b:  # 遍历PDP子图
        ax_b_item_b.set_ylim(final_y_min_pdp_b, final_y_max_pdp_b)  # 设置统一y轴范围
elif axes_b:  # 如果有子图但无y值
    print("警告: 未在所有 PDP 图中找到有效的 Y 值，无法自动设置统一的 Y 轴范围。")

y_label_pdp_text = "Predicted Output"  # PDP y轴标签
x_label_pdp_text = "Feature Value"  # PDP x轴标签
if axes_b:  # 如果创建了PDP子图
    num_pdp_cols_b = 2  # PDP列数
    num_pdp_axes_b = len(axes_b)  # PDP子图总数
    for idx_b, ax_b_item_b in enumerate(axes_b):  # 遍历PDP子图
        if idx_b % num_pdp_cols_b == 0:  # 如果是第一列的子图
            ax_b_item_b.set_ylabel(y_label_pdp_text, fontsize=10)  # 设置y轴标签
        if idx_b >= num_pdp_axes_b - num_pdp_cols_b:  # 如果是最后一行的子图
            ax_b_item_b.set_xlabel(x_label_pdp_text, fontsize=10)  # 设置x轴标签
    axes_b[0].text(
        -0.35,
        1.18,
        "b",
        transform=axes_b[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="right",
    )  # Panel 'b' 标签

plt.subplots_adjust(left=0.18, right=0.96, top=0.92, bottom=0.08)  # 调整整体布局

print("显示图表...")  # 打印信息：开始显示图表
plt.savefig(str(OUTPUT_DIR / "shap+PDP图.png"), dpi=1080)
plt.close("all")  # Interactive display removed; assets were exported above.
print("完成。")

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

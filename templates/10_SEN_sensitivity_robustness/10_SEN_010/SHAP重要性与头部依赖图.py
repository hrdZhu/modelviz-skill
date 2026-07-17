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

# 导入所有必需的库
import matplotlib.gridspec as gridspec  # 导入gridspec，用于在matplotlib中创建复杂的子图布局。
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于创建图表和进行数据可视化。
import numpy as np  # 导入numpy库，用于进行高效的数值计算，特别是数组操作。
import pandas as pd  # 导入pandas库，用于数据处理，特别是创建和操作DataFrame。
import shap  # 导入shap库，用于计算和可视化模型的SHAP（SHapley Additive explanations）值。
import xgboost  # 导入xgboost库，用于训练梯度提升树模型。
from sklearn.model_selection import (  # 从scikit-learn库中导入train_test_split和GridSearchCV函数。
    GridSearchCV,
    train_test_split,
)

# 打印出关键库的版本号，这对于调试和确保环境一致性非常有用。
print(f"SHAP version: {shap.__version__}")
print(f"XGBoost version: {xgboost.__version__}")
import os  # 导入os库，用于处理文件路径

import matplotlib  # 再次导入matplotlib库（可选）。

matplotlib.use(
    "Agg"
)  # 设置matplotlib的图形后端。'TkAgg'是一个常用的交互式后端，允许在独立的窗口中显示图形。如果遇到问题，可以尝试更换为 'Agg' (非交互式) 或 'Qt5Agg'。
# --- 用于寻找“拐点”的辅助函数 ---


from scipy.signal import savgol_filter


def find_knee_point(x_data, y_data, window_length=5, polyorder=2):
    """
    通过基于曲率的方法寻找曲线上趋势变化最显著的点（即“拐点”或“膝点”）。
    该方法对y数据进行平滑，然后计算其二阶导数，二阶导数绝对值最大的点被认为是拐点。

    :param x_data: x坐标数组 (Pandas Series or NumPy array)
    :param y_data: y坐标数组 (NumPy array)
    :param window_length: Savitzky-Golay 滤波器的窗口长度，必须是奇数。
    :param polyorder: Savitzky-Golay 滤波器的多项式阶数，必须小于 window_length。
    :return: 拐点的x坐标值
    """
    if len(x_data) < window_length:
        print(
            f"警告：数据点数 ({len(x_data)}) 小于 Savitzky-Golay 滤波器窗口长度 ({window_length})。正在返回中位数。"
        )
        return np.median(x_data)

    # 确保窗口长度是奇数
    if window_length % 2 == 0:
        window_length += 1

    # 确保多项式阶数小于窗口长度
    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:  # 确保 polyorder 至少为 1
            polyorder = 1

    # 对 y_data 进行平滑处理并计算其二阶导数
    # deriv=2 表示计算二阶导数
    y_second_deriv = savgol_filter(y_data, window_length, polyorder, deriv=2)

    # 寻找二阶导数绝对值最大的点，该点是曲率变化最剧烈的地方
    knee_index = np.argmax(np.abs(y_second_deriv))

    # 返回该“拐点”的x坐标作为阈值
    # 需要将x_data和y_data重新对齐，因为savgol_filter返回的是与y_data等长的数组
    sorted_x = np.array(x_data)[np.argsort(x_data)]
    return sorted_x[knee_index]


# --------------------------------------------------------------------------
# 步骤 1: 加载本地 Excel 数据
# --------------------------------------------------------------------------
print("--> 正在加载本地 Excel 数据...")
# 定义 Excel 文件路径
excel_file_path = str(DATA_DIR / "simulated_shap_data.xlsx")
# 检查文件是否存在
if not os.path.exists(excel_file_path):
    raise FileNotFoundError(f"未找到文件：{excel_file_path}。请检查文件路径是否正确。")
# 读取 Excel 文件
df = pd.read_excel(excel_file_path)
print(f"--> 已成功加载文件: {excel_file_path}")
print("数据预览:")
print(df.head())
# 定义特征列和目标列
target_column = "Target_Variable"  # 您的目标变量列名
feature_columns = [col for col in df.columns if col != target_column]
# 分离特征 (X) 和目标变量 (y)
X = df[feature_columns]
y = df[target_column]
# 使用 sklearn 的工具函数，将数据按80/20的比例划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
## 步骤 2: 训练 XGBoost 模型并进行超参数搜索
print("--> 正在对 XGBoost 模型进行超参数搜索...")
# 定义要搜索的超参数网格
param_grid = {
    "n_estimators": [100, 200, 300],  # 树的数量
    "max_depth": [3, 5, 7],  # 每棵树的最大深度
    "learning_rate": [0.05, 0.1, 0.2],  # 学习率（步长）
    "subsample": [0.7, 0.9],  # 训练每棵树时，用于采样的数据比例
    "colsample_bytree": [0.7, 0.9],  # 训练每棵树时，用于采样特征的比例
}
# 实例化XGBoost回归器模型
xgb_model = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
# 设置 GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
)
# 在训练数据上执行网格搜索
grid_search.fit(X_train, y_train)
# 获取最佳模型和最佳参数
print("\n--> 超参数搜索完成！")
print(f"最佳均方误差 (MSE): {-grid_search.best_score_:.4f}")
print("最佳参数组合:")
print(grid_search.best_params_)
# 使用最佳模型进行后续分析
model = grid_search.best_estimator_
print("\n--> 已使用最佳参数训练的模型进行后续分析。")
# --------------------------------------------------------------------------
# 步骤 3: 计算 SHAP 值
# --------------------------------------------------------------------------
print("--> 正在计算 SHAP 值...")
# TreeExplainer是为XGBoost、LightGBM等树模型优化的SHAP解释器，计算速度快
explainer = shap.TreeExplainer(model)
# 计算测试集中每个样本每个特征的SHAP值
shap_values = explainer(X_test)
# --------------------------------------------------------------------------
# 步骤 4: 绘制组合图
# --------------------------------------------------------------------------
print("--> 正在绘制 SHAP 组合图...")
# ==================== 美学参数配置区 ====================
# 在这里集中修改所有关于字体大小、颜色条、间距的参数，方便统一调整。
aesthetic_params = {
    # 字体大小设置
    "suptitle_size": 22,  # 总标题字号
    "ax_label_size": 16,  # 坐标轴标签字号
    "tick_label_size": 16,  # 坐标轴刻度字号
    "legend_size": 14,  # 图例字号
    "cbar_label_size": 12,  # 颜色条标签/标题字号
    # 摘要图颜色条的独立控制参数
    "summary_cbar_width": 0.015,  # 颜色条的宽度（占整个图表宽度的百分比）
    "summary_cbar_height_shrink": 1.0,  # 颜色条的高度缩放（1.0代表与主图等高）
    "summary_cbar_pad": 0.01,  # 颜色条与主图的水平间距
    # 依赖图颜色条的独立、精确控制参数
    "dep_cbar_width": 0.005,  # 依赖图颜色条的宽度
    "dep_cbar_height_shrink": 1.0,  # 依赖图颜色条的高度缩放（1.0代表与子图等高）
    "dep_cbar_pad": 0.002,  # 依赖图颜色条与子图的间距
    "dep_cbar_tick_length": 1,  # 依赖图颜色条上刻度的长度
    # 子图网格间距控制
    "grid_wspace": 0.45,  # 控制子图的水平间距
    "grid_hspace": 0.4,  # 控制子图的垂直间距
}
# ========================================================
# --- 设置全局字体为 "Times New Roman" ---
plt.rcParams["font.family"] = "Times New Roman"
# 创建画布和网格布局
fig = plt.figure(figsize=(20, 15))  # 创建一个大的画布(Figure)
gs = gridspec.GridSpec(  # 使用GridSpec创建一个3行4列的网格布局系统
    3,
    4,
    figure=fig,
    wspace=aesthetic_params["grid_wspace"],  # 并设置好子图的水平间距
    hspace=aesthetic_params["grid_hspace"],  # 和垂直间距
)
# --- 摘要图 (左侧) ---
# 将左侧所有行和前两列合并成一个大的子图区域
ax_main = fig.add_subplot(gs[:, :2])
# 计算每个特征的SHAP绝对值的平均值
mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
# 创建DataFrame以排序特征
feature_importance_df = pd.DataFrame(
    {"feature": X_test.columns, "importance": mean_abs_shaps}
).sort_values("importance", ascending=True)
# 设置Y轴刻度和标签
ax_main.set_yticks(range(len(feature_importance_df)))
ax_main.set_yticklabels(
    feature_importance_df["feature"], fontsize=aesthetic_params["tick_label_size"]
)
# 创建共享Y轴的顶部X轴
ax_top = ax_main.twiny()
# 在顶部轴绘制条形图
ax_top.barh(
    range(len(feature_importance_df)),
    feature_importance_df["importance"],
    color="lightgray",
    alpha=0.6,
    height=0.7,
)
# 设置顶部轴的标签和刻度字体
ax_top.set_xlabel(
    "Mean Absolute SHAP Value (Global Importance)", fontsize=aesthetic_params["ax_label_size"]
)
ax_top.tick_params(axis="x", labelsize=aesthetic_params["tick_label_size"])
ax_top.grid(False)  # 关闭顶部轴网格
# 定义颜色映射
cmap = plt.get_cmap("viridis")
# 循环绘制蜂巢图的散点
for i, feature_name in enumerate(feature_importance_df["feature"]):
    original_idx = X_test.columns.get_loc(feature_name)
    shap_vals_for_feature = shap_values.values[:, original_idx]
    feature_vals_for_color = X_test.iloc[:, original_idx]
    y_jitter = np.random.normal(0, 0.08, shap_vals_for_feature.shape[0])
    ax_main.scatter(
        shap_vals_for_feature,
        i + y_jitter,
        c=feature_vals_for_color,
        cmap=cmap,
        s=15,
        alpha=0.8,
        zorder=2,
    )
# 设置底部轴的标签和刻度字体
ax_main.set_xlabel(
    "SHAP value (impact on model output)", fontsize=aesthetic_params["ax_label_size"]
)
ax_main.tick_params(axis="x", labelsize=aesthetic_params["tick_label_size"])
ax_main.grid(True, axis="x", linestyle="--", alpha=0.6)  # 显示底部轴网格
# --- 摘要图颜色条 ---
fig.canvas.draw()  # 强制更新画布以获取正确位置
ax_main_pos = ax_main.get_position()  # 获取主图位置
# 计算颜色条的精确位置和大小
cax_left = ax_main_pos.x1 + aesthetic_params["summary_cbar_pad"]
cax_bottom = ax_main_pos.y0 + (
    ax_main_pos.height * (1 - aesthetic_params["summary_cbar_height_shrink"]) / 2
)
cax_width = aesthetic_params["summary_cbar_width"]
cax_height = ax_main_pos.height * aesthetic_params["summary_cbar_height_shrink"]
# 手动创建颜色条的坐标轴
cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
# 创建颜色映射标量
norm = plt.Normalize(vmin=X_test.values.min(), vmax=X_test.values.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
# 在指定坐标轴上绘制颜色条
cbar = fig.colorbar(sm, cax=cax)
# 设置颜色条标签
cbar.set_label(
    "Feature value", rotation=90, labelpad=-15, fontsize=aesthetic_params["cbar_label_size"]
)
cbar.outline.set_visible(False)  # 隐藏边框
cbar.set_ticks([])  # 移除数字刻度
# 手动添加"High"和"Low"文本
cbar.ax.text(
    0.6,
    1.02,
    "High",
    ha="center",
    va="top",
    transform=cbar.ax.transAxes,
    fontsize=aesthetic_params["tick_label_size"],
)
cbar.ax.text(
    0.6,
    -0.02,
    "Low",
    ha="center",
    va="bottom",
    transform=cbar.ax.transAxes,
    fontsize=aesthetic_params["tick_label_size"],
)
# --- 依赖图 (右侧) ---
# 获取最重要的6个特征
top_6_features = feature_importance_df["feature"].tail(6).iloc[::-1].tolist()
# 创建依赖图的子图坐标轴列表
axes_scatter = []
for i in range(3):
    for j in range(2):
        axes_scatter.append(fig.add_subplot(gs[i, j + 2]))
# 循环绘制每个依赖图
for i, feature in enumerate(top_6_features):
    ax = axes_scatter[i]  # 获取当前子图坐标轴
    feature_idx = X_test.columns.get_loc(feature)  # 获取特征索引
    x_data = X_test[feature]  # 准备x轴数据
    y_data = shap_values.values[:, feature_idx]  # 准备y轴数据
    color_data = y_test  # 准备颜色数据（使用目标变量）
    scatter = ax.scatter(x_data, y_data, c=color_data, cmap=cmap, s=25, alpha=0.8)  # 绘制散点图
    # --- 依赖图颜色条 ---
    fig.canvas.draw()  # 强制更新画布
    ax_pos = ax.get_position()  # 获取当前子图位置
    # 计算颜色条的精确位置
    cax_dep_left = ax_pos.x1 + aesthetic_params["dep_cbar_pad"]
    cax_dep_bottom = ax_pos.y0 + (
        ax_pos.height * (1 - aesthetic_params["dep_cbar_height_shrink"]) / 2
    )
    cax_dep_width = aesthetic_params["dep_cbar_width"]
    cax_dep_height = ax_pos.height * aesthetic_params["dep_cbar_height_shrink"]
    # 手动创建颜色条坐标轴
    cax_dep = fig.add_axes([cax_dep_left, cax_dep_bottom, cax_dep_width, cax_dep_height])
    # 在指定坐标轴上绘制颜色条
    cbar = fig.colorbar(scatter, cax=cax_dep)
    # 将标签设置为标题，放在颜色条顶部
    cbar.ax.set_title(target_column, fontsize=10)  # 将颜色条标题改为目标变量的列名
    cbar.outline.set_visible(False)  # 隐藏边框
    # 控制颜色条刻度
    cbar.ax.tick_params(
        axis="y",
        length=aesthetic_params["dep_cbar_tick_length"],  # 控制刻度线长短
        labelsize=aesthetic_params["tick_label_size"],  # 控制刻度数字大小
    )
    # --- 子图的其他元素 ---
    # 设置坐标轴标签
    ax.set_xlabel(feature, fontsize=aesthetic_params["ax_label_size"])
    ax.set_ylabel(f"SHAP", fontsize=12, labelpad=-8)
    # --- 阈值线计算 ---
    median_val = X_test[feature].median()  # 计算中位数
    threshold_val = find_knee_point(x_data, y_data)  # 动态计算趋势“拐点”
    # 绘制垂直线
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1)
    ax.axvline(threshold_val, color="red", linestyle=":", linewidth=1.2)
    # 创建图例来解释垂直线
    from matplotlib.lines import Line2D

    # 确保这里的标签是您原始代码中的样子
    line_handles = [
        Line2D([0], [0], color="black", lw=1, linestyle="--", label=f"Medain: {median_val:.2f}"),
        Line2D(
            [0], [0], color="red", lw=1, linestyle=":", label=f"Thresholds: {threshold_val:.2f}"
        ),
    ]
    ax.legend(
        handles=line_handles, loc="best", fontsize=aesthetic_params["legend_size"]
    )  # 显示图例
    ax.tick_params(
        axis="both", which="major", labelsize=aesthetic_params["tick_label_size"]
    )  # 设置刻度字体
# --- 最终布局与保存 ---
# 定义输出图片路径和文件名
output_image_path = str(OUTPUT_DIR / "shap_analysis_plot.png")
# 保存图像到指定路径，设置高分辨率
plt.savefig(
    output_image_path, dpi=300, bbox_inches="tight"
)  # bbox_inches='tight' 确保图表内容不被裁剪
print(f"--> SHAP 组合图已成功保存到文件: {output_image_path}")
# 显示最终图表
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

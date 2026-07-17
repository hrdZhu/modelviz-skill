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

# =============================================================================
# ===== 0. 用户配置 (请在这里修改您的文件路径和列名) =====
# =============================================================================
excel_file_path = str(DATA_DIR / "data.xlsx")  # 定义Excel数据文件的完整路径
target_column_name = "植被覆盖度"  # 定义您数据中作为目标变量（Y值）的列的名称
sheet_name = None  # 如果Excel文件有多个工作表，请在这里指定要读取的表名；如果只有一个或读取第一个，则保持为None
# =============================================================================
# ===== 1. 准备工作：导入库、加载并准备数据 =====
# =============================================================================
import os  # 导入os库，用于与操作系统交互，如创建文件夹、拼接路径
import re  # 导入re库，用于正则表达式操作，如此处用于文件名清理
import warnings  # 导入warnings库，用于控制警告信息的显示

import matplotlib  # 导入matplotlib主库，用于配置
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘制图表
import numpy as np  # 导入numpy库，用于进行高效的数值计算
import pandas as pd  # 导入pandas库，用于数据处理和分析，核心是DataFrame
import shap  # 导入shap库，用于模型解释，计算SHAP值
import xgboost as xgb  # 导入xgboost库，一个高效的梯度提升框架
from sklearn.metrics import (  # 从sklearn导入评估指标计算函数
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (  # 从sklearn导入数据划分和网格搜索交叉验证工具
    GridSearchCV,
    train_test_split,
)
from statsmodels.nonparametric.smoothers_lowess import (
    lowess,  # 从statsmodels库导入lowess平滑函数，用于绘制平滑拟合曲线
)

matplotlib.use("Agg")  # 设置matplotlib的后端，'TkAgg'是一个常用的图形界面后端
plt.rcParams["font.serif"] = [
    "Times New Roman"
]  # 设置matplotlib绘图时使用的衬线字体为'Times New Roman'
plt.rcParams["font.sans-serif"] = [
    "SimHei"
]  # 设置matplotlib绘图时使用的无衬线字体为'SimHei'（黑体），以正确显示中文
plt.rcParams["axes.unicode_minus"] = False  # 设置matplotlib正常显示负号
warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # 忽略特定类型的运行时警告，避免不必要的输出


def apply_plot_styles(ax, tick_width=1.5, tick_length=6, spine_width=1.5):
    """
    为一个Matplotlib Axes对象应用统一的绘图样式。
    参数说明:
    ax (matplotlib.axes.Axes): 要应用样式的子图对象。
    tick_width (float): 坐标轴主刻度线的粗细。
    tick_length (float): 坐标轴主刻度线的长度。
    spine_width (float): 图框（坐标系边框）的粗细。
    """
    # 设置所有四个图框的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    # 设置主刻度线的样式
    # axis='both' -> 同时作用于x和y轴
    # which='major' -> 只修改主刻度线
    # direction='in' -> 刻度线朝内
    ax.tick_params(axis="both", which="major", direction="in", width=tick_width, length=tick_length)


def sanitize_filename(name):  # 定义一个函数，用于清理文件名中的非法字符
    return re.sub(
        r'[\\/*?:"<>|]', "_", name
    )  # 使用正则表达式将Windows文件名中的非法字符替换为下划线


print("--- 开始执行任务 ---")  # 打印任务开始的提示信息
print(f"正在从 '{excel_file_path}' 加载数据...")  # 打印正在加载数据文件的提示
if sheet_name:  # 检查用户是否指定了工作表名称
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)  # 如果指定了，则读取特定的工作表
else:  # 如果没有指定工作表名称
    df = pd.read_excel(excel_file_path)  # 则默认读取Excel文件的第一个工作表
print("数据加载成功！")  # 如果数据成功加载，打印成功信息
if target_column_name not in df.columns:  # 检查目标列是否存在于DataFrame的列名中
    print(f"错误：目标列 '{target_column_name}' 不存在于数据中！")  # 如果不存在，打印错误信息
    print(f"可用的列为: {df.columns.tolist()}")  # 并列出所有可用的列名，方便用户检查
    exit()  # 退出程序
y = df[target_column_name]  # 将目标列的数据赋值给变量y
X = df.drop(columns=[target_column_name])  # 从DataFrame中删除目标列，剩下的作为特征集赋值给X
feature_names = X.columns.tolist()  # 获取所有特征的名称，并存为一个列表
print(f"目标 (y) 已设置为 '{target_column_name}' 列。")  # 打印目标列设置成功的提示
print(
    f"共找到 {len(feature_names)} 个特征用于模型训练: {feature_names}"
)  # 打印找到的特征数量和名称列表
print("正在检查并处理非数值特征...")  # 打印数据预处理开始的提示
for col in X.columns:  # 遍历所有特征列
    if X[col].dtype == "object":  # 检查该列的数据类型是否为'object'（通常是字符串）
        print(f"  -> 特征 '{col}' 是非数值类型，将尝试进行处理。")  # 如果是，打印提示信息
        X_converted = pd.to_numeric(
            X[col], errors="coerce"
        )  # 尝试将该列转换为数值类型，'coerce'会将无法转换的值变为NaN
        if X_converted.isnull().all():  # 如果转换后所有值都变成了NaN，说明该列是纯文本
            print(
                f"  -> 特征 '{col}' 是纯文本分类特征，将使用因子化（factorize）进行编码。"
            )  # 打印将要进行因子化编码的提示
            X[col], _ = pd.factorize(X[col])  # 使用pandas的factorize方法将文本转换为数字编码
        else:  # 如果部分值可以被转换
            X[col] = X_converted  # 将转换后的列（包含NaN）赋回原处
            if X[col].isnull().sum() > 0:  # 检查是否存在因转换失败而产生的NaN值
                median_val = X[col].median()  # 计算该列的中位数
                print(
                    f"  -> 特征 '{col}' 中存在部分无法转换的值，将使用中位数 ({median_val:.2f}) 进行填充。"
                )  # 打印将要用中位数填充的提示
                X[col].fillna(median_val, inplace=True)  # 使用中位数填充NaN值
print("数据预处理完成。")  # 打印数据预处理完成的提示
# =============================================================================
# ===== 2. 数据集划分、超参数搜索与模型训练 (模型为XGBoost) =====
# =============================================================================
print("\n正在划分训练集和验证集...")  # 打印数据集划分开始的提示
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)  # 使用train_test_split划分数据，30%作为测试集
print(
    f"数据集划分完成。训练集: {X_train.shape[0]}样本, 验证集: {X_test.shape[0]}样本。"
)  # 打印划分结果的样本数量
print("\n正在为XGBoost模型进行超参数搜索...")  # 打印超参数搜索开始的提示
FULL_ANALYSIS = _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1"
param_grid = {  # 定义一个字典，包含了希望搜索的超参数及其候选值
    "n_estimators": [100, 200, 500],  # 树的数量
    "max_depth": [5, 10, 15],  # 树的最大深度
    "learning_rate": [0.05, 0.1, 0.2],  # 学习率
}
if not FULL_ANALYSIS:
    param_grid = {parameter: [values[0]] for parameter, values in param_grid.items()}
xgb_model = xgb.XGBRegressor(
    random_state=0, eval_metric="rmse", n_jobs=-1 if FULL_ANALYSIS else 1
)  # 初始化一个XGBoost回归模型，设置随机种子和评估指标
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1 if FULL_ANALYSIS else 1,
    verbose=1,
)  # 配置网格搜索，使用3折交叉验证，并行计算，并打印过程信息
grid_search.fit(X_train, y_train)  # 在训练集上执行网格搜索
model = grid_search.best_estimator_  # 获取网格搜索找到的最佳模型
print("\n超参数搜索完成！")  # 打印搜索完成的提示
print(f"找到的最佳参数为: {grid_search.best_params_}")  # 打印找到的最佳超参数组合
print("已使用最佳参数构建最终模型。")  # 打印已使用最佳模型
print("\n正在绘制训练集与验证集的回归拟合图...")  # 打印开始绘制拟合图的提示
y_train_pred = model.predict(X_train)  # 使用模型对训练集进行预测
y_test_pred = model.predict(X_test)  # 使用模型对测试集进行预测
# 计算训练集的指标
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)
# 计算验证集的指标
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)
print(f"\n训练集评估指标: R2={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
print(f"验证集评估指标: R2={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
# ====================================


plt.figure(figsize=(8, 8), dpi=150)  # 创建一个新的图形窗口，并设置大小和分辨率
plt.scatter(
    y_train, y_train_pred, alpha=0.5, label="Train", color="blue"
)  # 绘制训练集的真实值与预测值的散点图
plt.scatter(
    y_test, y_test_pred, alpha=0.7, label="Validation", color="red", marker="^"
)  # 绘制测试集的真实值与预测值的散点图
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
    label="1：1拟合线 (y=x)",
)  # 绘制一条y=x的参考线
plt.xlabel("Actual Values", fontsize=12)  # 设置X轴标签
plt.ylabel("Predicted Values", fontsize=12)  # 设置Y轴标签
plt.title("XGBoost", fontsize=14)  # 设置图表标题
plt.legend(loc="upper left")  # 显示图例，并放在左上角
plt.grid(True)  # 显示网格线
# 创建一个将在图上显示的文本框
metrics_text = (
    f"验证集:\n"
    f"$R^2$ = {r2_test:.4f}\n"
    f"RMSE = {rmse_test:.4f}\n"
    f"MAE = {mae_test:.4f}\n\n"
    f"训练集:\n"
    f"$R^2$ = {r2_train:.4f}\n"
    f"RMSE = {rmse_train:.4f}\n"
    f"MAE = {mae_train:.4f}"
)
# 将文本框放置在图的右下角
plt.text(
    0.95,
    0.05,
    metrics_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
)
# 获取当前的Axes对象
ax = plt.gca()
# 应用我们定义的样式
apply_plot_styles(ax)
# =======================================
output_main_folder = str(OUTPUT_DIR)  # 定义主输出文件夹的路径
os.makedirs(output_main_folder, exist_ok=True)  # 创建主输出文件夹，如果已存在则不报错
fit_plot_path = os.path.join(
    output_main_folder, str(OUTPUT_DIR / "regression_fit_plot.png")
)  # 拼接拟合图的完整保存路径
plt.savefig(fit_plot_path, bbox_inches="tight")  # 保存图形，bbox_inches='tight'可以裁剪掉空白边缘
print(f"回归拟合图已保存至: '{fit_plot_path}'")  # 打印保存路径
plt.close("all")  # Interactive display removed; assets were exported above.
# =============================================================================
# ===== 3. SHAP分析 (关键修正：所有计算基于X_test) =====
# =============================================================================
print("\n正在使用最佳模型在 *测试集* 上计算SHAP值...")  # 打印开始在测试集上计算SHAP值的提示
explainer = shap.Explainer(model)  # 使用训练好的最佳模型创建一个SHAP解释器
print("正在计算主效应SHAP值 (基于X_test)...")  # 打印开始计算主效应SHAP值的提示
shap_values_obj = explainer(X_test)  # 将测试集数据传入解释器，计算每个样本每个特征的SHAP值
shap_values = shap_values_obj.values  # 从SHAP解释对象中提取SHAP值矩阵
print("主效应SHAP值计算完成。")  # 打印计算完成的提示
print("\n正在计算SHAP交互效应值 (基于X_test)...")  # 打印开始计算交互效应SHAP值的提示
shap_interaction_values = explainer.shap_interaction_values(
    X_test
)  # 计算SHAP交互效应值，这是一个三维数组
print("SHAP交互效应值计算完成。")  # 打印计算完成的提示
print("\n正在根据 SHAP 值对特征进行排序...")  # 打印开始排序特征的提示
mean_shap = np.abs(shap_values).mean(axis=0)  # 计算每个特征的SHAP绝对值的平均值，作为其重要性度量
shap_df = pd.DataFrame(
    {  # 创建一个DataFrame来存储特征名和其重要性
        "feature": feature_names,  # 特征名称列
        "mean_shap": mean_shap,  # 平均绝对SHAP值列
    }
).sort_values("mean_shap", ascending=False)  # 按重要性降序排序
sorted_features = shap_df["feature"].values  # 获取排序后的特征名称列表
print("特征排序完成。")  # 打印排序完成的提示
X_test_sorted = X_test[sorted_features]  # 按照排序后的特征顺序，重新排列测试集的列
orig_index = [
    feature_names.index(f) for f in sorted_features
]  # 获取排序后特征在原始特征列表中的索引
shap_values_sorted = shap_values[:, orig_index]  # 按照新的特征顺序，重新排列SHAP值矩阵的列
shap_interaction_values_sorted = shap_interaction_values[:, orig_index][
    :, :, orig_index
]  # 同样，重新排列交互效应值矩阵的行和列
# =============================================================================
# ===== 通用辅助函数 =====
# =============================================================================


def bootstrap_lowess_ci(
    x, y, n_boot=200, frac=0.5, ci_level=0.95
):  # 定义一个函数，用bootstrap方法计算lowess平滑的置信区间
    """
    使用bootstrap方法计算lowess平滑的置信区间。
    参数说明:
    x (pd.Series): 模型的输入特征（自变量）。
    y (pd.Series): 模型的输出或真实值（因变量）。
    n_boot (int): bootstrap抽样的次数。次数越多，置信区间的估计越稳定，但计算成本也越高。默认为200次。
    frac (float): lowess平滑器中使用的样本比例。这个值控制平滑的程度，介于0和1之间。
                  值越小，曲线越贴近数据点；值越大，曲线越平滑。默认为0.5。
    ci_level (float): 置信区间的水平。例如，0.95表示计算95%的置信区间。默认为0.95。
    返回:
    tuple: (主平滑曲线, (x轴范围, 置信下界, 置信上界)) 或 (None, None)
    """
    if len(x) < 10:
        return None, None  # 如果样本点太少，则不进行计算
    boot_lines = []  # 初始化一个列表，用于存储每次bootstrap抽样得到的平滑曲线
    x_range = np.linspace(x.min(), x.max(), 100)  # 在x的范围内生成100个等间距点，用于插值
    for _ in range(n_boot):  # 循环进行n_boot次bootstrap抽样
        sample_indices = np.random.choice(len(x), len(x), replace=True)  # 有放回地抽取样本索引
        x_sample, y_sample = x.iloc[sample_indices], y[sample_indices]  # 根据索引获取抽样数据
        sorted_indices = np.argsort(x_sample)  # 对抽样的x值进行排序
        x_sorted, y_sorted = (
            x_sample.iloc[sorted_indices].values,
            y_sample[sorted_indices],
        )  # 获取排序后的x和y
        if len(np.unique(x_sorted)) < 2:
            continue  # 如果抽样后x的唯一值少于2个，则跳过此次循环
        smoothed = lowess(y_sorted, x_sorted, frac=frac)  # 对抽样数据进行lowess平滑
        interp_func = np.interp(
            x_range, smoothed[:, 0], smoothed[:, 1]
        )  # 将平滑结果插值到x_range上
        boot_lines.append(interp_func)  # 将插值后的曲线添加到列表中
    if not boot_lines:
        return None, None  # 如果未能生成任何bootstrap曲线，则返回None
    sorted_indices_orig = np.argsort(x)  # 对原始x数据进行排序
    x_sorted_orig, y_sorted_orig = (
        x.iloc[sorted_indices_orig].values,
        y[sorted_indices_orig],
    )  # 获取排序后的原始x和y
    main_smoothed = lowess(
        y_sorted_orig, x_sorted_orig, frac=frac
    )  # 对完整的原始数据进行lowess平滑，作为主曲线
    boot_lines_arr = np.array(boot_lines)  # 将bootstrap曲线列表转换为numpy数组
    alpha = (1 - ci_level) / 2  # 计算置信水平对应的alpha值
    lower_bound, upper_bound = (
        np.quantile(boot_lines_arr, alpha, axis=0),
        np.quantile(boot_lines_arr, 1 - alpha, axis=0),
    )  # 计算每个点的上下置信边界
    return main_smoothed, (x_range, lower_bound, upper_bound)  # 返回主平滑曲线和置信区间数据


def find_and_plot_crossings(
    ax, x_curve, y_curve, color, base_y_offset=0.9
):  # 定义一个函数，用于寻找并绘制曲线与y=0的交点（阈值）
    """
    在给定的Matplotlib Axes上寻找并绘制一条曲线与y=0轴的交点（阈值）。
    该函数通过线性插值精确计算交点位置，并用垂直虚线和文本标签在图上标记出来。
    文本标签会自动上下交错排列以避免重叠。
    参数说明:
    ax (matplotlib.axes.Axes): 要在其上绘图的Matplotlib子图对象。
    x_curve (np.array): 曲线的x坐标数组。
    y_curve (np.array): 曲线的y坐标数组。函数将寻找此曲线与y=0的交点。
    color (str): 用于绘制垂直线和文本背景的颜色。应与对应曲线的颜色匹配。
    base_y_offset (float): 控制文本标签垂直位置的基准偏移量，相对于y轴的高度。
                           默认为0.9，即从顶部向下10%的位置开始，然后交替向下排列。
    """
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0]  # 找到y值符号发生变化的位置
    for i, k in enumerate(sign_changes):  # 遍历所有符号变化点
        x1, y1, x2, y2 = (
            x_curve[k],
            y_curve[k],
            x_curve[k + 1],
            y_curve[k + 1],
        )  # 获取变化点前后的坐标
        if (y2 - y1) == 0:
            continue  # 避免除以零
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1)  # 使用线性插值计算交点的x坐标（根）
        ax.axvline(x=x_root, color=color, linestyle="--", linewidth=1)  # 在交点位置绘制一条垂直虚线
        y_text_position = ax.get_ylim()[1] * (
            base_y_offset - (i % 2) * 0.1
        )  # 计算文本标签的y坐标，使其上下交错防止重叠
        ax.text(
            x_root,
            y_text_position,
            f" {x_root:.2f} ",
            color="white",
            backgroundcolor=color,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor=color, edgecolor="none", pad=1),
        )  # 在垂直线上方添加文本标签显示交点值


def find_roots(x_curve, y_curve):  # 定义一个函数，只计算曲线与y=0的交点，不绘图
    """
    计算一条曲线与y=0轴的所有交点（即方程的根），但不进行绘图。
    该函数通过线性插值法来精确估算交点的x坐标值。
    参数说明:
    x_curve (np.array): 曲线的x坐标数组。
    y_curve (np.array): 曲线的y坐标数组。函数将寻找此曲线与y=0的交点。
    返回:
    list: 一个包含所有计算出的交点（根）的x坐标值的列表。如果不存在交点，则返回空列表。
    """
    roots = []  # 初始化一个列表用于存储交点
    sign_changes = np.where(np.diff(np.sign(y_curve)))[0]  # 找到y值符号发生变化的位置
    for k in sign_changes:  # 遍历所有符号变化点
        x1, y1, x2, y2 = (
            x_curve[k],
            y_curve[k],
            x_curve[k + 1],
            y_curve[k + 1],
        )  # 获取变化点前后的坐标
        if (y2 - y1) == 0:
            continue  # 避免除以零
        x_root = x1 - y1 * (x2 - x1) / (y2 - y1)  # 使用线性插值计算交点的x坐标
        roots.append(x_root)  # 将计算出的交点添加到列表中
    return roots  # 返回所有交点的列表


# =============================================================================
# ===== 4. SHAP总览图与依赖图绘制=====
# =============================================================================
print("\n正在绘制 SHAP 特征重要性总览图...")  # 打印开始绘制总览图的提示
fig = plt.figure(figsize=(10, 10), dpi=300)  # 创建一个新的图形窗口
ax_sw = fig.add_axes([0.32, 0.11, 0.59, 0.77])  # 在图中添加一个子图(axes)用于绘制蜂群图(swarm plot)
ax_bar = ax_sw.twiny()  # 创建一个共享y轴的第二个x轴，用于绘制条形图
ax_bar.set_zorder(0)  # 将条形图坐标轴置于底层
ax_sw.set_zorder(1)  # 将蜂群图坐标轴置于顶层
ax_sw.patch.set_alpha(0)  # 将顶层坐标轴的背景设置为透明
y_pos = np.arange(len(sorted_features))[::-1]  # 计算每个特征在y轴上的位置
ax_bar.barh(
    y=y_pos,
    width=shap_df["mean_shap"].values,
    height=0.6,
    color="blue",
    alpha=0.5,
    edgecolor="none",
    zorder=0,
)  # 绘制水平条形图，表示每个特征的平均重要性
xlim_bar = shap_df["mean_shap"].values.max() * 1.05  # 计算条形图x轴的上限
ax_bar.set_xlim(0, xlim_bar)  # 设置条形图x轴的范围
xticks_bar = np.linspace(0, xlim_bar, 5)  # 计算条形图x轴的刻度
ax_bar.set_xticks(xticks_bar)  # 设置条形图x轴的刻度
ax_bar.set_xticklabels([f"{x:.2f}" for x in xticks_bar])  # 设置条形图x轴的刻度标签格式
ax_bar.set_xlabel("Mean (|SHAP| value)", fontsize=10)  # 设置条形图x轴的标签
ax_bar.set_yticks(y_pos)  # 设置y轴刻度位置（与蜂群图共享）
max_abs_shap = np.abs(shap_values_sorted).max()  # 计算所有SHAP值的最大绝对值
xlim_sw = max_abs_shap * 1.1  # 计算蜂群图x轴的范围
ax_sw.set_xlim(-xlim_sw, xlim_sw)  # 设置蜂群图x轴的范围
sw_xticks = np.linspace(-xlim_sw, xlim_sw, 5)  # 计算蜂群图x轴的刻度
ax_sw.set_xticks(sw_xticks)  # 设置蜂群图x轴的刻度
ax_sw.set_xticklabels([f"{x:.2f}" for x in sw_xticks])  # 设置蜂群图x轴的刻度标签格式
ax_sw.set_xlabel("SHAP value (impact on model output)", fontsize=10)  # 设置蜂群图x轴的标签
expl_main = shap.Explanation(  # 创建一个SHAP的Explanation对象，这是新版shap库绘图所需的标准格式
    values=shap_values_sorted,  # 排序后的SHAP值
    data=X_test_sorted.values,  # 排序后的特征值
    feature_names=list(sorted_features),  # 排序后的特征名
    base_values=shap_values_obj.base_values[0],  # 模型的基准值
)
shap.plots.beeswarm(
    expl_main, max_display=len(sorted_features), ax=ax_sw, show=False, plot_size=None
)  # 在指定的子图上绘制蜂群图
ax_sw.set_yticks(y_pos)  # 设置y轴刻度位置
ax_sw.set_yticklabels(sorted_features, fontsize=12)  # 设置y轴刻度标签（特征名称）
ax_sw.tick_params(axis="y", length=4)  # 调整y轴刻度线的长度
# ===== 移除顶部和右侧的图框 =====
ax_sw.spines["top"].set_visible(False)
ax_sw.spines["right"].set_visible(False)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
# =======================================
# 为两个Axes对象分别应用样式
apply_plot_styles(ax_sw)
apply_plot_styles(ax_bar)
# 由于ax_bar的刻度线我们可能不关心，可以特别设置一下
ax_bar.tick_params(axis="y", length=0)  # 隐藏y轴刻度线，因为它和ax_sw共享

combined_image_path = os.path.join(
    output_main_folder, str(OUTPUT_DIR / "combined_shap_summary_plot.png")
)  # 拼接组合图的保存路径
plt.savefig(combined_image_path, dpi=208, bbox_inches="tight")  # 保存图形
print(f"SHAP 特征重要性总览图已保存至: '{combined_image_path}'")  # 打印保存成功的提示
plt.close("all")  # Interactive display removed; assets were exported above.
print("\n正在绘制 SHAP 交互作用总览图...")  # 打印开始绘制交互作用图的提示
plt.figure()  # 创建一个新的图形窗口
shap.summary_plot(
    shap_interaction_values_sorted, X_test_sorted, max_display=10, show=False
)  # 使用shap库绘制交互作用的摘要图
# 获取刚刚由shap创建的Axes对象
ax = plt.gca()
apply_plot_styles(ax)
interaction_summary_plot_path = os.path.join(
    output_main_folder, str(OUTPUT_DIR / "shap_interaction_summary_plot.png")
)  # 拼接交互作用图的保存路径
plt.savefig(interaction_summary_plot_path, dpi=300, bbox_inches="tight")  # 保存图形
print(f"SHAP 交互作用总览图已保存至: '{interaction_summary_plot_path}'")  # 打印保存成功的提示


def plot_shap_dependence(
    feature_name, x_values, shap_values_for_feature, save_folder, custom_annotation=None
):  # 定义一个函数，用于绘制单个特征的SHAP依赖图
    """
    绘制并保存单个特征的SHAP依赖图。
    **SHAP值散点图**: 显示每个样本的特征值与其对应的SHAP值的关系（蓝色散点）。
    **特征值分布直方图**: 以背景条形图的形式展示该特征在数据集中的分布情况（红色条形）。
    **LOWESS平滑拟合曲线**: 揭示SHAP值随特征值变化的平均趋势（深蓝色实线）。
    **置信区间**: 为LOWESS曲线提供统计可靠性范围，通常是95%置信区间（浅蓝色填充区域）。
    **阈值（交点）标定**: 自动寻找并标记出拟合曲线与y=0的交点，这些点是特征影响方向（正/负）发生改变的关键阈值（绿色虚线和标签）。
    **自定义注释**: 允许用户在图上添加自定义文本。
    参数说明:
    feature_name (str): 要绘制的特征的名称。将用作图表X轴标签和输出文件名的一部分。
    x_values (pd.Series or np.array): 该特征在数据集中的所有样本值。
    shap_values_for_feature (np.array): 与x_values一一对应的SHAP值。
    save_folder (str): 用于保存生成图像文件的文件夹路径。
    custom_annotation (dict, optional): 一个可选字典，用于在图上添加自定义注释。
                                     例如: {'text': '关键区域', 'x': 0.8, 'y': 0.8}
    """
    print(f"  -> 正在绘制特征: {feature_name} ...")  # 打印正在绘制哪个特征的提示
    fig_dep, ax1 = plt.subplots(figsize=(8, 6), dpi=150)  # 创建一个新的图形和子图
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
    ax2.patch.set_alpha(0)  # 将第二个y轴的背景设置为透明
    counts, bin_edges = np.histogram(x_values, bins=30)  # 计算特征值的分布直方图数据
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 计算每个柱子的中心位置
    bin_width = bin_edges[1] - bin_edges[0]  # 计算每个柱子的宽度
    ax1.bar(
        bin_centers,
        counts,
        width=bin_width * 0.6,
        align="center",
        color="#4B0082",
        alpha=0.3,
        label="Distribution",
    )  # 在ax1上绘制分布直方图
    ax1.set_ylabel("Distribution", fontsize=12)  # 设置ax1的y轴标签
    ax1.set_ylim(0, counts.max() * 1.1)  # 设置ax1的y轴范围
    ax2.scatter(
        x_values,
        shap_values_for_feature,
        alpha=0.3,
        s=25,
        color="#00008B",
        label="Sample",
        zorder=2,
    )  # 在ax2上绘制SHAP值的散点图
    if len(x_values) > 1:  # 检查样本数量是否足够
        main_fit, ci_data = bootstrap_lowess_ci(
            x_values, shap_values_for_feature, frac=0.3
        )  # 计算lowess平滑曲线和置信区间
        if main_fit is not None and ci_data is not None:  # 如果成功计算
            ax2.plot(
                main_fit[:, 0], main_fit[:, 1], color="#9400D3", lw=2, label="LOWESS Fit", zorder=4
            )  # 绘制主平滑曲线
            ax2.fill_between(
                ci_data[0], ci_data[1], ci_data[2], color="#D3D3D3", alpha=0.15, label="95%CI"
            )  # 填充置信区间
            ax2.axhline(0, color="black", linestyle="--", lw=1, zorder=1)  # 绘制y=0的参考线
            find_and_plot_crossings(
                ax2, main_fit[:, 0], main_fit[:, 1], "black"
            )  # 寻找并绘制阈值线
    ax2.set_ylabel("SHAP value", fontsize=12)  # 设置ax2的y轴标签
    y_max = np.abs(shap_values_for_feature).max() * 1.15  # 计算ax2的y轴范围
    if y_max < 1e-6:
        y_max = 1  # 避免范围过小
    ax2.set_ylim(-y_max, y_max)  # 设置ax2的y轴范围
    ax1.set_xlabel(f"{feature_name}", fontsize=12)  # 设置共享的x轴标签
    if custom_annotation and isinstance(custom_annotation, dict):  # 检查是否有自定义注释
        text = custom_annotation.get("text", "")  # 获取注释文本
        x_pos = custom_annotation.get("x", 0.95)  # 获取注释x坐标
        y_pos = custom_annotation.get("y", 0.95)  # 获取注释y坐标
        props = {
            "ha": custom_annotation.get("ha", "right"),
            "va": custom_annotation.get("va", "top"),
            "fontsize": custom_annotation.get("fontsize", 12),
            "color": custom_annotation.get("color", "darkred"),
            "fontweight": custom_annotation.get("fontweight", "bold"),
        }  # 定义注释的样式
        ax1.text(x_pos, y_pos, text, transform=ax1.transAxes, **props)  # 在图上添加自定义注释
    h1, l1 = ax1.get_legend_handles_labels()  # 获取ax1的图例句柄和标签
    h2, l2 = ax2.get_legend_handles_labels()  # 获取ax2的图例句柄和标签
    ax2.legend(h2 + h1, l2 + l1, loc="upper right", fontsize=10)  # 合并两个y轴的图例并显示
    apply_plot_styles(ax1)
    apply_plot_styles(ax2)
    sanitized_feature_name = sanitize_filename(feature_name)  # 清理特征名以用作文件名
    output_filename = f"dependence_plot_{sanitized_feature_name}.png"  # 构建输出文件名
    full_path = os.path.join(save_folder, output_filename)  # 拼接完整保存路径
    fig_dep.savefig(full_path, dpi=200, bbox_inches="tight")  # 保存图形
    plt.close(fig_dep)  # 关闭图形，释放内存


print("\n开始为所有特征批量生成依赖图...")  # 打印批量生成依赖图的开始提示
output_folder_dep = os.path.join(
    output_main_folder, "dependence_plots"
)  # 定义依赖图的输出文件夹路径
os.makedirs(output_folder_dep, exist_ok=True)  # 创建该文件夹
print(f"所有依赖图将被保存到 '{output_folder_dep}' 文件夹中。")  # 打印保存路径提示
for i, feature_name in enumerate(sorted_features):  # 遍历所有排序后的特征
    x_data_loop = X_test_sorted[feature_name]  # 获取当前特征的数值
    if not np.isfinite(x_data_loop).all():  # 检查特征值中是否包含NaN或无穷大等非有限值
        print(
            f"  -> 跳过特征: '{feature_name}'，因为它包含非有限值 (例如 NaN)。"
        )  # 如果有，则跳过该特征
        continue  # 继续下一个循环
    y_data_shap_loop = shap_values_sorted[:, i]  # 获取当前特征对应的SHAP值
    annotation_for_this_plot = None  # 初始化注释为空
    # if i == 0: # 如果是第一个（最重要的）特征
    #     annotation_for_this_plot = {'text': 'This is the most important feature!', 'x': 0.98, 'y': 0.98,
    #                                 'color': 'purple'} # 为其添加一个特殊的注释
    # elif i == 1: # 如果是第二个特征
    #     annotation_for_this_plot = {'text': 'Second most important', 'x': 0.05, 'y': 0.2, 'ha': 'left',
    #                                 'color': 'green'} # 添加另一个不同的注释
    plot_shap_dependence(
        feature_name=feature_name,
        x_values=x_data_loop,
        shap_values_for_feature=y_data_shap_loop,
        save_folder=output_folder_dep,
        custom_annotation=annotation_for_this_plot,
    )  # 调用绘图函数绘制并保存依赖图
print(f"\n任务完成！所有依赖图已成功生成并保存。")  # 打印任务完成的提示
# =============================================================================
# ===== 5. 提取并保存每个特征的详细SHAP交互值 =====
# =============================================================================
print("\n正在为每个特征提取详细的、每个样本的SHAP交互值...")  # 打印开始提取交互值的提示
output_folder_interactions = os.path.join(
    output_main_folder, "interaction_values_per_feature"
)  # 定义交互值CSV文件的输出文件夹
os.makedirs(output_folder_interactions, exist_ok=True)  # 创建该文件夹
print(
    f"每个特征的详细交互值将被保存到 '{output_folder_interactions}' 文件夹中。"
)  # 打印保存路径提示
for i, primary_feature_name in enumerate(sorted_features):  # 遍历每个特征，作为“主特征”
    interaction_data = {}  # 初始化一个字典，用于存储该主特征与其他所有特征的交互值
    for j, secondary_feature_name in enumerate(sorted_features):  # 再次遍历所有特征，作为“次要特征”
        if i == j:
            continue  # 如果主特征和次要特征是同一个，则跳过
        column_name = f"interaction_{sanitize_filename(primary_feature_name)}_vs_{sanitize_filename(secondary_feature_name)}"  # 构建列名
        interaction_values_for_pair = shap_interaction_values_sorted[
            :, i, j
        ]  # 从交互效应矩阵中提取这对特征的交互值
        interaction_data[column_name] = interaction_values_for_pair  # 将交互值存入字典
    feature_interaction_df = pd.DataFrame(interaction_data)  # 将字典转换为DataFrame
    print(
        f"  -> 正在处理主特征: '{primary_feature_name}'，并保存其交互值..."
    )  # 打印正在处理哪个主特征的提示
    csv_filename = (
        f"interactions_for_{sanitize_filename(primary_feature_name)}.csv"  # 构建CSV文件名
    )
    full_path = os.path.join(output_folder_interactions, csv_filename)  # 拼接完整保存路径
    feature_interaction_df.to_csv(
        full_path, index=False, encoding="utf-8-sig"
    )  # 将DataFrame保存为CSV文件，使用'utf-8-sig'编码以兼容Excel打开中文
print(
    f"\n任务完成！所有 {len(sorted_features)} 个特征的详细交互值已作为独立的CSV文件保存。"
)  # 打印任务完成的提示
if len(sorted_features) > 0:  # 检查是否有特征
    first_feature_name = sorted_features[0]  # 获取最重要的特征名称
    print(
        f"\n预览：最重要的特征 '{first_feature_name}' 与其他特征的交互值 (前5行):"
    )  # 打印预览提示
    first_feature_csv_path = os.path.join(
        output_folder_interactions, f"interactions_for_{sanitize_filename(first_feature_name)}.csv"
    )  # 获取其对应的CSV文件路径
    if os.path.exists(first_feature_csv_path):  # 检查文件是否存在
        preview_df = pd.read_csv(first_feature_csv_path)  # 读取该CSV文件
        print(preview_df.head())  # 打印文件的前5行作为预览
# =============================================================================
# ===== 6. 绘制高级SHAP交互图 =====
# =============================================================================
print("\n--- 开始执行高级交互图绘制任务 (最终修正版) ---")  # 打印高级绘图任务开始的提示


# 定义一个函数，用于绘制两个特征之间的高级交互图
def plot_advanced_interaction(
    primary_feature_name,
    interacting_feature_name,
    x_values,
    interaction_feature_values,
    shap_interaction_slice,
    save_folder,
):
    """
    绘制并保存一个高级的、信息丰富的特征交互SHAP图。
    该函数旨在可视化一个主特征（primary_feature）的SHAP值如何受到一个交互特征（interacting_feature）的影响。
    图表主要包含以下几个部分：
    1.  **交互散点图**: 以主特征值为X轴，SHAP交互值为Y轴。散点的颜色由交互特征的值决定，
        使用'seismic'（蓝-白-红）色谱，直观展示交互特征值高低对主特征效应的影响。
    2.  **分组拟合曲线**: 将交互特征按其中位数分为“高值组”和“低值组”，并为这两组数据
        分别绘制LOWESS平滑拟合曲线（红色和蓝色实线）及其置信区间。这清晰地揭示了
        主特征的效应趋势是否因交互特征的水平不同而改变。
    3.  **共同阈值标定**: 自动计算并寻找两组拟合曲线共同穿过y=0的“稳定”阈值点。
        如果找到，则用紫色虚线和标签在图上标出。这个阈值点可能代表一个不受交互特征影响的、
        稳健的效应转变点。
    4.  **背景分布图**: 以灰色条形图在背景中展示主特征的数据分布，为趋势分析提供数据密度参考。
    参数说明:
    primary_feature_name (str): 主特征的名称，将显示在X轴。
    interacting_feature_name (str): 交互特征的名称，其值将决定散点的颜色和分组。
    x_values (pd.Series or np.array): 主特征的实际值数组。
    interaction_feature_values (pd.Series or np.array): 交互特征的实际值数组。
    shap_interaction_slice (np.array): 对应的主特征与交互特征之间的SHAP交互值数组。
    save_folder (str): 用于保存生成图像文件的文件夹路径。
    """
    sanitized_primary, sanitized_interacting = (
        sanitize_filename(primary_feature_name),
        sanitize_filename(interacting_feature_name),
    )  # 清理主特征和交互特征的名称
    print(
        f"  -> 正在绘制: '{primary_feature_name}' (交互特征: '{interacting_feature_name}')"
    )  # 打印正在绘制哪对特征的提示
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)  # 创建一个新的图形和子图
    ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["blue", "#4B0082", "red"])
    points = ax2.scatter(
        x_values,
        shap_interaction_slice,
        c=interaction_feature_values,
        cmap=cmap,
        alpha=1,
        s=25,
        zorder=2,
        label="sample",
    )  # 在ax2上绘制散点图，点的颜色由交互特征的值决定
    median_val = interaction_feature_values.median()  # 计算交互特征值的中位数
    low_mask, high_mask = (
        interaction_feature_values <= median_val,
        interaction_feature_values > median_val,
    )  # 根据中位数创建两个布尔掩码，用于分组
    # --- 修改1：修正图例(Legend)的标签文本 ---
    groups = {  # 定义两个组的配置信息
        "low": {
            "mask": low_mask,
            "color": "blue",
            "offset": 0.9,
            "label": f" {interacting_feature_name} <= {median_val:.2f}",
        },  # 低值组的配置，包含动态生成的图例标签
        "high": {
            "mask": high_mask,
            "color": "red",
            "offset": 0.8,
            "label": f" {interacting_feature_name} > {median_val:.2f}",
        },  # 高值组的配置
    }
    counts, bin_edges = np.histogram(x_values, bins=30)  # 计算主特征的分布数据
    bin_centers, bin_width = (
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        bin_edges[1] - bin_edges[0],
    )  # 计算柱子中心和宽度
    ax1.bar(
        bin_centers, counts, width=bin_width * 0.7, color="gray", alpha=0.2, label="Distribution"
    )  # 在ax1上绘制分布直方图
    ax1.set_ylabel("Distribution", fontsize=12)  # 设置ax1的y轴标签
    ax1.set_ylim(0, counts.max() * 1.1)  # 设置ax1的y轴范围
    fits, roots = {}, {}  # 初始化字典用于存储拟合曲线和根
    for name, info in groups.items():  # 遍历高低值两个组
        x_group, shap_group = (
            x_values[info["mask"]],
            shap_interaction_slice[info["mask"]],
        )  # 根据掩码获取分组数据
        if len(x_group) < 10:
            continue  # 如果组内样本太少，则跳过
        main_fit, ci_data = bootstrap_lowess_ci(
            x_group, shap_group
        )  # 对该组数据进行lowess平滑和置信区间计算
        if main_fit is not None and ci_data is not None:  # 如果计算成功
            ax2.plot(
                main_fit[:, 0],
                main_fit[:, 1],
                color=f"dark{info['color']}",
                lw=2.5,
                label=info["label"],
            )  # <-- 使用修正后的标签来绘制平滑曲线
            ax2.fill_between(
                ci_data[0], ci_data[1], ci_data[2], color=info["color"], alpha=0.15
            )  # 填充该曲线的置信区间
            fits[name] = main_fit  # 存储拟合曲线
            roots[name] = find_roots(main_fit[:, 0], main_fit[:, 1])  # 计算并存储该曲线的根
    if "low" in roots and "high" in roots:  # 如果高低两组都找到了根
        tolerance = (
            x_values.max() - x_values.min()
        ) * 0.05  # 定义一个容差，用于判断两个根是否“接近”
        for r_low in roots["low"]:  # 遍历低值组的根
            for r_high in roots["high"]:  # 遍历高值组的根
                if abs(r_low - r_high) < tolerance:  # 如果两个根非常接近
                    avg_root = (r_low + r_high) / 2  # 计算它们的平均值
                    ax2.axvline(
                        x=avg_root, color="black", linestyle="--", linewidth=1
                    )  # 在该位置绘制一条紫色的垂直虚线，表示共同的阈值
                    ax2.text(
                        avg_root,
                        ax2.get_ylim()[1] * 0.9,
                        f" {avg_root:.2f} ",
                        color="white",
                        backgroundcolor="purple",
                        ha="center",
                        va="center",
                        fontsize=9,
                        bbox=dict(facecolor="purple", edgecolor="none", pad=1),
                    )  # 在线上方添加文本标签
    # ---调整颜色条(Color Bar)的间距 ---
    # 在图窗(Figure)的指定位置手动创建一个新的坐标系(Axes)，专门用于放置颜色条。
    # 这种方法提供了最精确的布局控制。
    # 坐标 [left, bottom, width, height] 是相对于整个图窗的比例（0到1）。
    #   left=0.92: 颜色条的左边界从图窗左侧92%处开始，即放在主图的右侧并留出一些间隙。
    #   bottom=0.11, height=0.77: 定义了颜色条的垂直位置和长度，使其与主图的坐标系在垂直方向上对齐，解决了颜色条自动变短的问题。
    #   width=0.03: 定义了颜色条的宽度，这个值现在是唯一独立控制颜色条“粗细”的参数。
    cbar_ax = fig.add_axes([0.975, 0.11, 0.02, 0.77])
    # 将颜色条(colorbar)绘制到我们刚刚创建的专用坐标系 cbar_ax 中。
    #   points: 这是散点图对象(ax.scatter的返回值)，颜色条将根据它的颜色映射(cmap)和数据范围来绘制。
    #   cax=cbar_ax: 这里的 cax 参数是关键，它告诉matplotlib“请把颜色条画在这个指定的cbar_ax里”，而不是让它自动寻找位置。
    cbar = fig.colorbar(points, cax=cbar_ax)
    cbar.set_label(f"Value of {interacting_feature_name}", size=12)  # 设置颜色条的标签
    ax1.set_xlabel(f"{primary_feature_name}", fontsize=12)  # 设置x轴标签
    ax2.set_ylabel(f"SHAP Interaction Value", fontsize=12)  # 设置ax2的y轴标签
    # fig.suptitle(f"Interaction: {primary_feature_name} vs {interacting_feature_name}", fontsize=14) # 设置整个图的标题
    ax2.axhline(0, color="black", linestyle="--", lw=1, zorder=0)  # 绘制y=0的参考线
    y_max_abs = np.abs(shap_interaction_slice).max() * 1.1  # 计算ax2的y轴范围
    ax2.set_ylim(
        -y_max_abs if y_max_abs > 1e-6 else -1, y_max_abs if y_max_abs > 1e-6 else 1
    )  # 设置ax2的y轴范围
    ax2.legend(loc="best", fontsize=10)  # 显示图例
    ax1.set_zorder(0)  # 将ax1置于底层
    ax2.set_zorder(1)  # 将ax2置于顶层
    ax2.patch.set_alpha(0)  # 将ax2的背景设置为透明
    apply_plot_styles(ax1)
    apply_plot_styles(ax2)
    output_filename = (
        f"interaction_{sanitized_primary}_vs_{sanitized_interacting}.png"  # 构建输出文件名
    )
    full_path = os.path.join(save_folder, output_filename)  # 拼接完整保存路径
    fig.savefig(full_path, dpi=200, bbox_inches="tight")  # 保存图形
    plt.close(fig)  # 关闭图形，释放内存


output_folder_advanced_interactions = os.path.join(
    output_main_folder, "advanced_interaction_plots_final"
)  # 定义高级交互图的输出文件夹
os.makedirs(output_folder_advanced_interactions, exist_ok=True)  # 创建该文件夹
print(
    f"\n所有高级交互图将按指定条件被保存到 '{output_folder_advanced_interactions}' 文件夹中。"
)  # 打印保存路径提示
n_features = len(sorted_features) if FULL_ANALYSIS else min(4, len(sorted_features))
for i in range(n_features):  # 遍历所有特征，作为主特征
    primary_feature_name = sorted_features[i]  # 获取主特征名称
    for j in range(n_features):  # 再次遍历所有特征，作为交互特征
        if i == j:
            continue  # 如果是同一个特征，则跳过
        interacting_feature_name = sorted_features[j]  # 获取交互特征名称
        x_values = X_test_sorted[primary_feature_name]  # 获取主特征的数值
        interaction_feature_values = X_test_sorted[
            interacting_feature_name
        ]  # 获取交互特征的数值（用于着色）
        shap_interaction_slice = (
            shap_interaction_values_sorted[:, i, j] * 2
        )  # 获取这对特征的交互SHAP值（乘以2是为了与shap库的默认行为保持一致）
        plot_advanced_interaction(  # 调用高级交互图的绘图函数
            primary_feature_name=primary_feature_name,
            interacting_feature_name=interacting_feature_name,
            x_values=x_values,
            interaction_feature_values=interaction_feature_values,
            shap_interaction_slice=shap_interaction_slice,
            save_folder=output_folder_advanced_interactions,
        )
print(f"\n--- 所有绘图任务完成！ ---")  # 打印所有任务完成的最终提示

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

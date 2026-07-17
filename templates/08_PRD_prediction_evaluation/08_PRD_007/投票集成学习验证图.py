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

# -*- 指定文件编码为 UTF-8，以支持中文字符 -*-
# -------------------------------------------------------------------
# 步骤 0: 导入所有必需的 Python 库
# -------------------------------------------------------------------
import os  # 导入 os 库，用于处理文件和目录路径
import time  # 导入 time 库，用于计算代码执行时间

import matplotlib  # 导入 matplotlib 主库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于数据可视化和绘图，通常简写为 plt
import numpy as np  # 导入 numpy 库，用于进行高效的数值计算，通常简写为 np
import pandas as pd  # 导入 pandas 库，用于数据处理和分析，通常简写为 pd
import shap  # 导入 shap 库，用于解释机器学习模型的预测
from scipy.stats import linregress  # 从 scipy.stats 库中导入 linregress 函数，用于计算线性回归
from sklearn.base import clone  # 从 scikit-learn 库中导入 clone 函数，用于创建模型的副本
from sklearn.ensemble import (  # 从 scikit-learn 库中导入梯度提升、随机森林和投票集成回归模型
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.kernel_ridge import KernelRidge  # 从 scikit-learn 库中导入 KernelRidge 模型 (核岭回归)
from sklearn.metrics import (  # 从 scikit-learn 库中导入用于模型评估的指标函数
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (  # 从 scikit-learn 库中导入用于划分数据集、超参数搜索和绘制学习曲线的工具
    GridSearchCV,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # 从 scikit-learn 库中导入 Pipeline 类，用于串联多个处理步骤
from sklearn.preprocessing import (
    StandardScaler,  # 从 scikit-learn 库中导入 StandardScaler 类，用于数据标准化
)
from sklearn.svm import SVR  # 从 scikit-learn 库中导入 SVR 模型 (支持向量回归)
from xgboost import XGBRegressor  # 从 xgboost 库中导入 XGBRegressor 模型 (XGBoost 回归)

# 设置 matplotlib 的后端为 'Agg'，这使得图形可以保存到文件而不会在屏幕上显示
matplotlib.use("Agg")
# -------------------------------------------------------------------
# 步骤 1: 配置 Matplotlib 以正确显示中文字符
# -------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = [
    "SimHei"
]  # 设置 Matplotlib 的默认字体为 'SimHei' (黑体)，以支持中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决在使用中文主题时，坐标轴负号显示为方框的问题
# -------------------------------------------------------------------
# 步骤 2: 加载数据并进行预处理
# -------------------------------------------------------------------
print("正在加载数据...")  # 在控制台输出提示信息，表示开始加载数据
file_path = str(DATA_DIR / "data.xlsx")  # 定义存储原始数据的 Excel 文件路径
output_folder = str(OUTPUT_DIR)  # 定义用于存放所有分析结果和图表的输出文件夹路径
if not os.path.exists(output_folder):  # 检查输出文件夹是否已经存在
    os.makedirs(output_folder)  # 如果文件夹不存在，则创建该文件夹
    print(f"已创建输出文件夹: {output_folder}")  # 打印确认信息，告知用户输出文件夹已创建
df = pd.read_excel(file_path)  # 使用 pandas 读取指定路径下的 Excel 文件到 DataFrame 中
print("数据加载成功！")  # 打印确认信息，告知用户数据已成功加载
df = pd.get_dummies(
    df, columns=["所属区域"], prefix="区域"
)  # 对 '所属区域' 这个类别特征进行独热编码 (One-Hot Encoding)，生成新的二进制列
target_column = "植被覆盖度"  # 定义目标变量（需要预测的列）的名称
feature_columns = [
    col for col in df.columns if col != target_column and not col.startswith("区域_")
]  # 定义特征列的列表，即所有除了目标变量和独热编码区域之外的列
X = df[feature_columns]  # 从 DataFrame 中提取所有特征列，创建特征矩阵 X
y = df[target_column]  # 从 DataFrame 中提取目标变量列，创建目标向量 y
groups = (
    df.filter(like="区域_").idxmax(axis=1).str.replace("区域_", "")
)  # 提取每个样本所属的区域分组信息，用于后续的分组可视化
# 将整个数据集划分为训练验证集 (80%) 和测试集 (20%)
X_train_val, X_test, y_train_val, y_test, groups_train_val, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)
# 将训练验证集进一步划分为训练集 (占总体的60%) 和验证集 (占总体的20%)
X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
    X_train_val, y_train_val, groups_train_val, test_size=0.25, random_state=42
)

print(
    f"数据集划分完毕: 训练集({len(X_train)}), 验证集({len(X_val)}), 测试集({len(X_test)})"
)  # 打印划分后各数据集的大小信息

# -------------------------------------------------------------------
# 步骤 3: 构建模型流水线 (Pipeline) 和集成模型
# -------------------------------------------------------------------
FULL_SEARCH = _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1"
SEARCH_CV = 5 if FULL_SEARCH else 3
WORKERS = -1 if FULL_SEARCH else 1
CURVE_SIZES = np.linspace(0.1, 1.0, 5 if FULL_SEARCH else 3)

pipe_gbdt = Pipeline(
    [("model", GradientBoostingRegressor(random_state=42))]
)  # 创建梯度提升回归 (GBDT) 模型的流水线
pipe_xgb = Pipeline(
    [("model", XGBRegressor(random_state=42, n_jobs=WORKERS))]
)  # 创建 XGBoost 回归模型的流水线，并设置 n_jobs=-1 以使用所有 CPU核心
pipe_rf = Pipeline(
    [("model", RandomForestRegressor(random_state=42, n_jobs=WORKERS))]
)  # 创建随机森林回归 (RF) 模型的流水线，并设置 n_jobs=-1 以使用所有 CPU核心
pipe_krr = Pipeline(
    [("scaler", StandardScaler()), ("model", KernelRidge())]
)  # 创建核岭回归 (KRR) 模型的流水线，包含一个标准化步骤
pipe_svr = Pipeline(
    [("scaler", StandardScaler()), ("model", SVR())]
)  # 创建支持向量回归 (SVR) 模型的流水线，包含一个标准化步骤

# 定义投票回归器 (VotingRegressor)，将上面定义的五个基模型集成在一起
ensemble_model = VotingRegressor(
    estimators=[  # 指定构成集成模型的基模型列表，每个模型都是一个元组 (名称, 模型对象)
        ("gbdt", pipe_gbdt),
        ("xgb", pipe_xgb),
        ("rf", pipe_rf),
        ("krr", pipe_krr),
        ("svr", pipe_svr),
    ],
    n_jobs=WORKERS,
)

# 定义用于网格搜索 (GridSearchCV) 的超参数空间
param_grid = {
    "gbdt__model__n_estimators": [
        100,
        200,
        300,
    ],  # 为 GBDT 模型的 'n_estimators' (树的数量) 参数设置候选值
    # 'gbdt__model__learning_rate': [0.01, 0.05, 0.1],  # 为 GBDT 模型的学习率设置候选值 (已注释)
    # 'gbdt__model__max_depth': [3, 5, 7],  # 为 GBDT 模型的最大深度设置候选值 (已注释)
    "xgb__model__n_estimators": [
        100,
        200,
        300,
    ],  # 为 XGBoost 模型的 'n_estimators' (树的数量) 参数设置候选值
    # 'xgb__model__learning_rate': [0.01, 0.05, 0.1],  # 为 XGBoost 模型的学习率设置候选值 (已注释)
    # 'xgb__model__max_depth': [3, 5, 7],  # 为 XGBoost 模型的最大深度设置候选值 (已注释)
    "rf__model__n_estimators": [
        100,
        200,
        300,
    ],  # 为随机森林模型的 'n_estimators' (树的数量) 参数设置候选值
    # 'rf__model__max_features': ['sqrt', 'log2', 0.8],  # 为随机森林模型的最大特征数设置候选值 (已注释)
    # 'rf__model__max_depth': [10, 20, None],  # 为随机森林模型的最大深度设置候选值 (已注释)
    "krr__model__alpha": [0.1, 1.0, 10.0],  # 为核岭回归模型的正则化强度 'alpha' 设置候选值
    # 'krr__model__kernel': ['linear', 'rbf', 'polynomial'],  # 为核岭回归模型的核函数设置候选值 (已注释)
    # 'krr__model__gamma': [0.01, 0.1, 1],  # 为核岭回归模型的 'gamma' 参数设置候选值 (已注释)
    "svr__model__C": [1, 10, 100],  # 为支持向量回归模型的正则化参数 'C' 设置候选值
    # 'svr__model__gamma': ['scale', 'auto', 0.1],  # 为支持向量回归模型的 'gamma' 参数设置候选值 (已注释)
    # 'svr__model__epsilon': [0.01, 0.1, 0.2]  # 为支持向量回归模型的 'epsilon' 参数设置候选值 (已注释)
}

if not FULL_SEARCH:
    param_grid = {parameter: [values[0]] for parameter, values in param_grid.items()}


# -------------------------------------------------------------------
# 步骤 4: 定义用于模型评估和可视化的函数
# -------------------------------------------------------------------
def plot_prediction_results(
    y_true, y_pred, groups, title, save_path
):  # 定义一个函数，用于绘制真实值与预测值的散点图
    fig, ax = plt.subplots(figsize=(10, 8))  # 创建一个图形和一个子图
    r2 = r2_score(y_true, y_pred)  # 计算 R² 分数
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算均方根误差 (RMSE)
    mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差 (MAE)
    n = len(y_true)  # 获取样本数量
    slope, intercept, _, _, _ = linregress(
        y_true, y_pred
    )  # 对真实值和预测值进行线性回归，得到斜率和截距
    y_pred_series = pd.Series(
        y_pred, index=y_true.index
    )  # 将预测值数组转换为 Pandas Series，以便于按索引进行分组
    color_map = {
        "华北地区": "purple",
        "华南地区": "orange",
        "西北地区": "darkgreen",
    }  # 定义一个颜色映射，为不同区域的散点指定不同颜色
    for group_name, color in color_map.items():  # 遍历颜色映射，为每个区域绘制散点
        mask = groups == group_name  # 创建一个布尔掩码，用于筛选出当前区域的样本
        if mask.sum() > 0:  # 检查当前区域是否有样本
            ax.scatter(
                y_true[mask],
                y_pred_series[mask],
                c=color,
                label=group_name,
                alpha=0.7,
                s=100,
                edgecolors="w",
            )  # 绘制当前区域的散点图
    overall_min = min(y_true.min(), y_pred.min())  # 计算所有真实值和预测值中的最小值
    overall_max = max(y_true.max(), y_pred.max())  # 计算所有真实值和预测值中的最大值
    buffer = (overall_max - overall_min) * 0.05  # 计算坐标轴的缓冲范围
    plot_lim_min = overall_min - buffer  # 设置坐标轴的最小值
    plot_lim_max = overall_max + buffer  # 设置坐标轴的最大值
    ax.set_xlim(plot_lim_min, plot_lim_max)  # 应用计算出的 X 轴范围
    ax.set_ylim(plot_lim_min, plot_lim_max)  # 应用计算出的 Y 轴范围
    ax.plot(
        [plot_lim_min, plot_lim_max],
        [plot_lim_min, plot_lim_max],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="1:1 Line",
    )  # 绘制 1:1 对角线
    x_fit = np.array([y_true.min(), y_true.max()])  # 创建用于绘制拟合线的数据点
    ax.plot(
        x_fit, slope * x_fit + intercept, color="#D32F2F", linewidth=2.5, label="Fitted Line"
    )  # 绘制线性回归拟合线
    stats_text = (
        f"y = {intercept:.2f} + {slope:.2f}x\n"
        f"$R^2$ = {r2:.2f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"MAE = {mae:.2f}\n"
        f"N = {n}"
    )  # 准备一个包含所有性能指标的文本字符串
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FAF0E6", alpha=0.85, edgecolor="none"),
    )  # 在图表的左上角添加文本框
    ax.set_xlabel("真实值 (Measured)", fontsize=16, fontweight="bold")  # 设置 X 轴的标签
    ax.set_ylabel("预测值 (Estimated)", fontsize=16, fontweight="bold")  # 设置 Y 轴的标签
    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)  # 设置图表标题
    legend = ax.legend(loc="lower right", frameon=False, fontsize=14)  # 显示图例
    ax.tick_params(
        axis="both", which="major", length=4, width=1.5, labelsize=16
    )  # 自定义坐标轴刻度的样式
    for label in ax.get_xticklabels() + ax.get_yticklabels():  # 遍历所有坐标轴刻度标签
        label.set_fontweight("bold")  # 将坐标轴刻度标签设置为粗体
    for spine in ax.spines.values():  # 遍历坐标轴的四个边框
        spine.set_linewidth(1.5)  # 设置坐标轴边框线的宽度
    ax.grid(True)  # 显示网格线
    # ax.set_aspect('equal', 'box')  # 设置坐标轴的宽高比为相等
    plt.tight_layout()  # 自动调整子图参数，使其填充整个图像区域
    plt.savefig(save_path, dpi=300)  # 将生成的图表保存到指定路径，分辨率为 300 DPI
    print(f"图表已保存到: {save_path}")  # 打印保存路径的确认信息
    plt.close(fig)  # 关闭图形，释放内存


def plot_residuals(y_true, y_pred, model_name, save_path):  # 定义一个函数，用于绘制模型的残差图
    residuals = y_true - y_pred  # 计算残差（真实值减去预测值）
    std_dev = np.std(residuals)  # 计算残差的标准差
    outlier_threshold = 1.96 * std_dev  # 定义离群点的阈值（约 95% 置信区间以外）
    is_outlier = np.abs(residuals) > outlier_threshold  # 判断每个点是否为离群点
    fig, ax = plt.subplots(figsize=(10, 6))  # 创建一个图形和一个子图
    ax.scatter(
        y_pred[~is_outlier], residuals[~is_outlier], c="blue", alpha=0.6, label="普通样本"
    )  # 绘制非离群点的残差散点图
    ax.scatter(
        y_pred[is_outlier],
        residuals[is_outlier],
        c="red",
        alpha=0.8,
        edgecolors="k",
        label="离群样本",
    )  # 绘制离群点的残差散点图（用不同颜色高亮）
    ax.axhline(y=0, color="red", linestyle="--")  # 在 y=0 的位置绘制一条水平虚线
    ax.fill_between(
        [min(y_pred), max(y_pred)],
        -std_dev,
        std_dev,
        color="yellow",
        alpha=0.3,
        label=f"1倍标准差范围 (σ={std_dev:.2f})",
    )  # 绘制表示一倍标准差范围的阴影区域
    ax.set_xlabel("预测值", fontsize=14)  # 设置 X 轴标签
    ax.set_ylabel("残差 (真实值 - 预测值)", fontsize=14)  # 设置 Y 轴标签
    ax.set_title(f"{model_name} 残差分布图", fontsize=16)  # 设置图表标题
    ax.legend()  # 显示图例
    ax.grid(True)  # 显示网格线
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=300)  # 保存图表
    print(f"残差图已保存到: {save_path}")  # 打印保存确认信息
    plt.close(fig)  # 关闭图形


def plot_shap_summary(
    model, X, feature_names, model_name, save_path
):  # 定义一个函数，用于计算并绘制 SHAP 摘要图以解释特征重要性
    print(f"正在为 {model_name} 进行 SHAP 分析... (此过程可能较慢)")  # 打印 SHAP 分析开始的提示信息

    def predict_fn_wrapper(
        numpy_array,
    ):  # 定义一个包装函数，因为 KernelExplainer 需要一个只接受 NumPy 数组的预测函数
        df_to_predict = pd.DataFrame(
            numpy_array, columns=feature_names
        )  # 将 NumPy 数组转换回带有特征名的 DataFrame
        return model.predict(df_to_predict)  # 使用模型的 predict 方法进行预测

    background_size = min(len(X), 100 if FULL_SEARCH else 20)
    explain_size = min(len(X), len(X) if FULL_SEARCH else 30)
    X_summary = shap.sample(X, background_size, random_state=42)
    X_explain = shap.sample(X, explain_size, random_state=43)
    explainer = shap.KernelExplainer(
        predict_fn_wrapper, X_summary
    )  # 创建一个 KernelExplainer 解释器实例
    shap_values = explainer.shap_values(
        X_explain, nsamples="auto" if FULL_SEARCH else 128
    )  # 默认示例使用受控样本量；完整分析由 MODELVIZ_FULL_SEARCH 开启
    shap.summary_plot(
        shap_values, X_explain, feature_names=feature_names, show=False
    )  # 生成 SHAP 摘要图，但暂时不显示
    ax = plt.gca()  # 获取当前的坐标轴对象
    ax.axvspan(
        ax.get_xlim()[0], 0, color="lightblue", alpha=0.2, zorder=0
    )  # 在图表左侧（负向影响）添加淡蓝色背景
    ax.axvspan(
        0, ax.get_xlim()[1], color="lightpink", alpha=0.2, zorder=0
    )  # 在图表右侧（正向影响）添加淡粉色背景
    plt.title(f"{model_name} SHAP 特征重要性分析", fontsize=16)  # 设置图表标题
    fig = plt.gcf()  # 获取当前的图形对象
    fig.tight_layout()  # 自动调整布局
    fig.savefig(save_path, dpi=300)  # 保存图表
    print(f"SHAP 图表已保存到: {save_path}")  # 打印保存确认信息
    plt.close(fig)  # 关闭图形


def add_noise(X, noise_level=0.05):  # 定义一个函数，用于向数据集中添加指定水平的高斯噪声
    X_noisy = X.copy()  # 创建数据集的副本，以避免修改原始数据
    for col in X_noisy.columns:  # 遍历数据集的每一列（每个特征）
        std = X_noisy[col].std()  # 计算该列的标准差
        if std > 0:  # 检查标准差是否大于0，避免对常数列添加噪声
            noise = np.random.normal(
                0, noise_level * std, size=X_noisy[col].shape
            )  # 生成与该列标准差成比例的正态分布（高斯）噪声
            X_noisy[col] += noise  # 将生成的噪声添加到当前列
    return X_noisy  # 返回添加了噪声的数据集


def print_noisy_evaluation_metrics(
    model, X_train, y_train, X_test, y_test, groups_test
):  # 定义一个函数，用于在添加噪声后的数据上评估模型，以测试其稳健性
    print("\n--- 在 5% 噪声数据上的稳健性评估 ---")  # 打印评估部分的标题
    X_train_noisy = add_noise(X_train)  # 向训练集特征添加噪声
    X_test_noisy = add_noise(X_test)  # 向测试集特征添加噪声
    model_clone_for_noise = clone(model)  # 克隆一个新模型，以避免影响原始训练好的模型
    model_clone_for_noise.fit(X_train_noisy, y_train)  # 使用带噪声的训练数据重新训练模型
    y_pred_noisy = model_clone_for_noise.predict(X_test_noisy)  # 在带噪声的测试集上进行预测
    r2 = r2_score(y_test, y_pred_noisy)  # 计算在噪声数据上的 R²
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_noisy))  # 计算在噪声数据上的 RMSE
    mae = mean_absolute_error(y_test, y_pred_noisy)  # 计算在噪声数据上的 MAE
    residuals_df = pd.DataFrame(
        {"residuals": y_test - y_pred_noisy, "group": groups_test}
    )  # 创建一个包含残差和分组信息的 DataFrame
    group_mean_residuals = residuals_df.groupby("group")[
        "residuals"
    ].mean()  # 按区域分组计算残差的均值
    between_group_variance = (
        group_mean_residuals.var()
    )  # 计算各组残差均值的方差，用于衡量模型的公平性
    print(f"  R²: {r2:.4f}")  # 打印 R²
    print(f"  RMSE: {rmse:.4f}")  # 打印 RMSE
    print(f"  MAE: {mae:.4f}")  # 打印 MAE
    print(
        f"  组间残差均值方差: {between_group_variance:.4f} (越小表示模型对各区域越公平)"
    )  # 打印组间方差


def plot_learning_curve_custom(
    model, model_name, X, y, save_path
):  # 定义一个函数，用于计算并绘制模型的学习曲线
    print(f"\n正在为 {model_name} 计算学习曲线数据...")  # 打印计算开始的提示信息
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=SEARCH_CV,
        n_jobs=WORKERS,
        train_sizes=CURVE_SIZES,
        scoring="r2",
    )  # 调用 scikit-learn 的 learning_curve 函数计算得分
    print(f"✅ {model_name} 学习曲线数据计算完成！")  # 打印计算完成的确认信息
    plt.figure(figsize=(10, 7))  # 创建一个新的图形
    train_scores_mean = np.mean(train_scores, axis=1)  # 计算训练得分的均值
    train_scores_std = np.std(train_scores, axis=1)  # 计算训练得分的标准差
    test_scores_mean = np.mean(test_scores, axis=1)  # 计算交叉验证得分的均值
    test_scores_std = np.std(test_scores, axis=1)  # 计算交叉验证得分的标准差
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="red", label="训练集得分"
    )  # 绘制训练得分的均值曲线
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="red",
    )  # 绘制训练得分的标准差范围
    plt.plot(
        train_sizes, test_scores_mean, "s-", color="green", label="交叉验证得分"
    )  # 绘制交叉验证得分的均值曲线
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="green",
    )  # 绘制交叉验证得分的标准差范围
    plt.title(f"{model_name} 学习曲线", fontsize=18)  # 设置图表标题
    plt.xlabel("训练样本数量", fontsize=14)  # 设置 X 轴标签
    plt.ylabel("$R^2$ 分数", fontsize=14)  # 设置 Y 轴标签
    padding = (test_scores_mean.max() - test_scores_mean.min()) * 0.1  # 计算 Y 轴的填充范围
    plt.ylim(test_scores_mean.min() - padding, 1.05)  # 调整 Y 轴范围以获得更好的可视化效果
    plt.legend(loc="best")  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=300)  # 保存图表
    print(f"学习曲线图已保存到: {save_path}")  # 打印保存确认信息
    plt.close("all")  # 关闭所有打开的图形，释放内存


# -------------------------------------------------------------------
# 步骤 5: 执行模型训练、评估与结果分析
# -------------------------------------------------------------------
print("\n" + "=" * 80)  # 打印一个分隔行以增强可读性
print(
    "🚀 开始为集成模型进行超参数搜索... (由于搜索空间已扩大，此过程可能非常耗时，请耐心等待)"
)  # 打印开始进行网格搜索的提示信息
print("=" * 80)  # 打印一个分隔行

start_time = time.time()  # 记录网格搜索开始的时间
grid_search = GridSearchCV(
    estimator=ensemble_model,
    param_grid=param_grid,
    cv=SEARCH_CV,
    scoring="r2",
    n_jobs=WORKERS,
    verbose=2,
)  # 初始化 GridSearchCV 对象
grid_search.fit(X_train, y_train)  # 在训练集上执行网格搜索和模型训练
end_time = time.time()  # 记录网格搜索结束的时间

print("\n" + "=" * 80)  # 打印报告标题前的分隔行
print("🎉🎉🎉 模型训练与评估分析报告 🎉🎉🎉")  # 打印主报告标题
print("=" * 80)  # 打印报告标题后的分隔行

print(f"\n--- 1. 超参数搜索结果 ---")  # 打印第一部分：超参数搜索结果
print(f"集成模型超参数搜索完成！总耗时: {end_time - start_time:.2f} 秒")  # 打印网格搜索的总耗时
print(f"找到的最佳参数组合: \n{grid_search.best_params_}")  # 打印搜索到的最佳超参数组合

print(f"\n--- 2. 集成模型交叉验证精度 ---")  # 打印第二部分：交叉验证精度
print(
    f"在训练集上进行5折交叉验证得到的最佳 R² 分数: {grid_search.best_score_:.4f}"
)  # 打印在交叉验证中获得的最佳 R² 分数

best_ensemble_model = grid_search.best_estimator_  # 从网格搜索结果中获取性能最佳的模型实例
model_name = "Ensemble_Model"  # 为最佳集成模型指定一个名称，用于后续的函数调用

print(f"\n--- 3. 集成模型在验证集上的精度 ---")  # 打印第三部分：在独立验证集上的性能
y_val_pred_ensemble = best_ensemble_model.predict(X_val)  # 使用最佳模型对验证集进行预测
val_r2_ensemble = r2_score(y_val, y_val_pred_ensemble)  # 计算验证集上的 R²
val_rmse_ensemble = np.sqrt(mean_squared_error(y_val, y_val_pred_ensemble))  # 计算验证集上的 RMSE
val_mae_ensemble = mean_absolute_error(y_val, y_val_pred_ensemble)  # 计算验证集上的 MAE
print(f"集成模型在独立验证集上的表现: ")  # 打印一个子标题
print(f"  R²: {val_r2_ensemble:.4f}")  # 打印 R²
print(f"  RMSE: {val_rmse_ensemble:.4f}")  # 打印 RMSE
print(f"  MAE: {val_mae_ensemble:.4f}")  # 打印 MAE

# ###################################################################
# ### 评估并打印每个基模型的性能 ###
# ###################################################################
print("\n" + "-" * 80)  # 打印分隔线
print("--- 4. 各个基模型的独立性能评估 ---")  # 打印第四部分：评估各个独立的基模型
print("（使用集成模型中已训练好的、带有最佳超参数的基模型进行评估）")  # 打印说明文字
print("-" * 80)  # 打印分隔线

best_params = grid_search.best_params_  # 从网格搜索结果中再次获取最佳参数字典

# 遍历最佳集成模型中每一个已经训练好的基模型（及其流水线）
for model_name_key, trained_model_pipeline in best_ensemble_model.named_estimators_.items():
    print(
        f"\n========== 模型: {model_name_key.upper()} =========="
    )  # 打印当前正在评估的基模型的名称

    print("  [最佳超参数]:")  # 打印子标题，用于显示该模型的最佳超参数
    # 从总的最佳参数字典中，筛选出属于当前基模型的参数
    model_specific_params = {
        k.split("__")[-1]: v for k, v in best_params.items() if k.startswith(model_name_key + "__")
    }
    if model_specific_params:  # 检查当前模型是否有在网格搜索中调整的超参数
        for (
            param,
            value,
        ) in model_specific_params.items():  # 如果有，则遍历并打印这些超参数及其最佳值
            print(f"    - {param}: {value}")
    else:  # 如果没有对应的超参数
        print("    - (该模型未在GridSearch中设置可调参数)")  # 打印提示信息

    y_val_pred_base = trained_model_pipeline.predict(X_val)  # 使用当前基模型对验证集进行预测
    val_r2_base = r2_score(y_val, y_val_pred_base)  # 计算验证集上的 R²
    val_rmse_base = np.sqrt(mean_squared_error(y_val, y_val_pred_base))  # 计算验证集上的 RMSE
    val_mae_base = mean_absolute_error(y_val, y_val_pred_base)  # 计算验证集上的 MAE
    print("  [验证集性能]:")  # 打印验证集性能的子标题
    print(f"    - R²: {val_r2_base:.4f}")  # 打印 R²
    print(f"    - RMSE: {val_rmse_base:.4f}")  # 打印 RMSE
    print(f"    - MAE: {val_mae_base:.4f}")  # 打印 MAE

    y_test_pred_base = trained_model_pipeline.predict(X_test)  # 使用当前基模型对测试集进行预测
    test_r2_base = r2_score(y_test, y_test_pred_base)  # 计算测试集上的 R²
    test_rmse_base = np.sqrt(mean_squared_error(y_test, y_test_pred_base))  # 计算测试集上的 RMSE
    test_mae_base = mean_absolute_error(y_test, y_test_pred_base)  # 计算测试集上的 MAE
    print("  [测试集性能]:")  # 打印测试集性能的子标题
    print(f"    - R²: {test_r2_base:.4f}")  # 打印 R²
    print(f"    - RMSE: {test_rmse_base:.4f}")  # 打印 RMSE
    print(f"    - MAE: {test_mae_base:.4f}")  # 打印 MAE

print("\n" + "-" * 80)  # 打印分隔线
print(
    "通过以上对比，可以清晰地看到每个基模型对最终集成结果的贡献，以及集成模型是否带来了性能上的提升。"
)  # 打印一段总结性文字
print("-" * 80)  # 打印分隔线
# ###################################################################
# ### 结束 ###
# ###################################################################


print(f"\n--- 5. 集成模型在各数据集上进行标准绘图与分析 ---")  # 打印第五部分：集成模型的可视化分析
y_train_pred_ensemble = best_ensemble_model.predict(X_train)  # 使用最佳集成模型对训练集进行预测
y_test_pred_ensemble = best_ensemble_model.predict(X_test)  # 使用最佳集成模型对测试集进行预测

# 调用绘图函数，生成训练集的预测结果图
plot_prediction_results(
    y_train,
    y_train_pred_ensemble,
    groups_train,
    f"Ensemble_Model 在训练集上的表现",
    os.path.join(output_folder, f"Ensemble_Model_performance_TRAIN.png"),
)
# 调用绘图函数，生成验证集的预测结果图
plot_prediction_results(
    y_val,
    y_val_pred_ensemble,
    groups_val,
    f"Ensemble_Model 在验证集上的表现",
    os.path.join(output_folder, f"Ensemble_Model_performance_VALIDATION.png"),
)
# 调用绘图函数，生成测试集的预测结果图
plot_prediction_results(
    y_test,
    y_test_pred_ensemble,
    groups_test,
    f"Ensemble_Model 在测试集上的表现",
    os.path.join(output_folder, f"Ensemble_Model_performance_TEST.png"),
)

# 调用绘图函数，生成测试集的残差图
plot_residuals(
    y_test,
    y_test_pred_ensemble,
    "Ensemble_Model",
    os.path.join(output_folder, f"Ensemble_Model_residuals.png"),
)

# 调用绘图函数，生成集成模型的 SHAP 特征重要性图
plot_shap_summary(
    best_ensemble_model,
    X_test,
    feature_names=X_test.columns,
    model_name="Ensemble_Model",
    save_path=os.path.join(output_folder, f"Ensemble_Model_shap_summary.png"),
)

# 调用评估函数，测试模型在含噪声数据上的稳健性
print_noisy_evaluation_metrics(
    best_ensemble_model, X_train_val, y_train_val, X_test, y_test, groups_test
)

# 调用绘图函数，生成集成模型的学习曲线
plot_learning_curve_custom(
    best_ensemble_model,
    "Ensemble_Model",
    X_train_val,
    y_train_val,
    os.path.join(output_folder, f"Ensemble_Model_learning_curve.png"),
)

print("\n" + "=" * 80)  # 打印结束前的分隔行
print("✅ 所有模型处理及分析完毕！")  # 打印任务全部完成的提示信息
print(f"所有结果图表已保存至: {output_folder}")  # 告知用户所有结果图表的保存位置
print("=" * 80)  # 打印最后的结束分隔行

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

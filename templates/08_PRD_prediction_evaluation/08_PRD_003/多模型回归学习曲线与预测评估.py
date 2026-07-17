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

# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# 导入所需的库
# -------------------------------------------------------------------
import os  # 导入os库，用于与操作系统交互，如文件路径操作
import time  # 导入time库，用于计算代码执行时间

import matplotlib  # 导入matplotlib主库
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于数据可视化
import numpy as np  # 导入numpy库，用于科学计算，特别是数组操作
import pandas as pd  # 导入pandas库，用于数据处理和分析
import shap  # 导入shap库，用于模型解释性分析
from scipy.stats import linregress  # 从scipy导入线性回归函数，用于绘图
from sklearn.ensemble import (  # 从sklearn导入GBDT和随机森林回归模型
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge  # 从sklearn导入核脊回归模型
from sklearn.metrics import (  # 从sklearn导入模型评估指标
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (  # 从sklearn导入数据划分、网格搜索和学习曲线工具
    GridSearchCV,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor  # 从sklearn导入K近邻回归模型
from sklearn.neural_network import MLPRegressor  # 从sklearn导入多层感知机回归模型
from sklearn.preprocessing import StandardScaler  # 从sklearn导入标准化工具
from sklearn.svm import SVR  # 从sklearn导入支持向量回归模型
from xgboost import XGBRegressor  # 从xgboost库导入XGBoost回归模型

matplotlib.use("Agg")  # 设置matplotlib的后端，用于在某些环境中正确显示图形
# -------------------------------------------------------------------
# 第 1 步: 设置 Matplotlib 中文显示
# -------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置matplotlib使用"黑体"字体来显示中文
plt.rcParams["axes.unicode_minus"] = False  # 设置matplotlib正常显示负号
# -------------------------------------------------------------------
# 第 2 步: 加载和准备数据
# -------------------------------------------------------------------
print("正在加载数据...")  # 打印提示信息
file_path = str(DATA_DIR / "模拟单目标分组回归数据.xlsx")  # 定义输入数据文件的路径
output_folder = str(OUTPUT_DIR)  # 定义输出结果的文件夹路径
if not os.path.exists(output_folder):  # 检查输出文件夹是否存在
    os.makedirs(output_folder)  # 如果不存在，则创建该文件夹
    print(f"已创建输出文件夹: {output_folder}")  # 打印提示信息
df = pd.read_excel(file_path)  # 使用pandas读取Excel文件到DataFrame
print("数据加载成功！")  # 打印提示信息
df = pd.get_dummies(
    df, columns=["所属区域"], prefix="区域"
)  # 对'所属区域'列进行独热编码，生成新的'区域_*'列

target_column = "植被覆盖度"  # 定义目标变量（Y值）的列名
feature_columns = [
    col for col in df.columns if col != target_column and not col.startswith("区域_")
]  # 定义特征变量（X值）的列名列表，不包括目标列和独热编码前的区域列
X = df[feature_columns]  # 从DataFrame中提取所有特征列，创建特征集X
print("特征", X)
y = df[target_column]  # 从DataFrame中提取目标列，创建目标集y
print("标签", y)
groups = (
    df.filter(like="区域_").idxmax(axis=1).str.replace("区域_", "")
)  # 提取分组信息，用于后续绘图时的颜色区分

# 将数据划分为训练验证集和测试集，同时划分分组信息以保持数据对齐
X_train_val, X_test, y_train_val, y_test, groups_train_val, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)
# 将训练验证集进一步划分为训练集和验证集
X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
    X_train_val,
    y_train_val,
    groups_train_val,
    test_size=0.25,
    random_state=42,  # test_size=0.25意味着(1-0.2)*0.25=0.2，即验证集占20%
)

scaler = StandardScaler().fit(X_train)  # 创建一个标准化器，并仅在训练数据上进行拟合
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train), columns=X.columns, index=X_train.index
)  # 标准化训练集
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val), columns=X.columns, index=X_val.index
)  # 标准化验证集
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)  # 标准化测试集
X_train_val_scaled = pd.DataFrame(
    scaler.fit(X_train_val).transform(X_train_val), columns=X.columns, index=X_train_val.index
)  # 为学习曲线准备标准化的完整训练验证集

print(
    f"数据集划分完毕: 训练集({len(X_train)}), 验证集({len(X_val)}), 测试集({len(X_test)})"
)  # 打印数据集划分结果

# -------------------------------------------------------------------
# 第 3 步: 定义模型超参数网格
# -------------------------------------------------------------------
FULL_SEARCH = _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1"
SEARCH_CV = 5 if FULL_SEARCH else 3
WORKERS = -1 if FULL_SEARCH else 1
CURVE_SIZES = np.linspace(0.1, 1.0, 5 if FULL_SEARCH else 3)

models_to_run = [  # 创建一个列表，其中每个元素是一个字典，定义了一个要运行的模型及其配置
    {
        "name": "GBDT",
        "estimator": GradientBoostingRegressor(random_state=42),
        "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "needs_scaling": False,
    },
    {
        "name": "XGB",
        "estimator": XGBRegressor(random_state=42, n_jobs=WORKERS),
        "param_grid": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "needs_scaling": False,
    },
    {
        "name": "RF",
        "estimator": RandomForestRegressor(random_state=42),
        "param_grid": {"n_estimators": [100, 200], "min_samples_leaf": [1, 5]},
        "needs_scaling": False,
    },
    {
        "name": "KRR",
        "estimator": KernelRidge(),
        "param_grid": [
            {"kernel": ["rbf"], "alpha": [0.1, 1]},
            {"kernel": ["linear"], "alpha": [0.1, 1]},
        ],
        "needs_scaling": True,
    },
    {
        "name": "SVR",
        "estimator": SVR(),
        "param_grid": {"C": [10, 100], "kernel": ["rbf"]},
        "needs_scaling": True,
    },
    {
        "name": "KNN",
        "estimator": KNeighborsRegressor(),
        "param_grid": {"n_neighbors": [5, 7], "metric": ["minkowski"]},
        "needs_scaling": True,
    },
    {
        "name": "MLP",
        "estimator": MLPRegressor(random_state=42, max_iter=1000, early_stopping=True),
        "param_grid": {
            "hidden_layer_sizes": [(50,), (100,)],
            "solver": ["adam"],
            "alpha": [0.001, 0.01],
        },
        "needs_scaling": True,
    },
]

if not FULL_SEARCH:
    for model_config in models_to_run:
        grid = model_config["param_grid"]
        grids = grid if isinstance(grid, list) else [grid]
        model_config["param_grid"] = {
            parameter: [values[0]] for parameter, values in grids[0].items()
        }
    models_to_run = models_to_run[:3]


# -------------------------------------------------------------------
# 第 4 步: 定义所有绘图和分析函数 (完整版)
# -------------------------------------------------------------------


def plot_prediction_results(
    y_true, y_pred, groups, title, save_path
):  # 定义函数，用于绘制预测结果散点图
    """绘制真实值与预测值的散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))  # 创建一个图形和坐标轴对象
    r2 = r2_score(y_true, y_pred)  # 计算R²分数
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算均方根误差
    mae = mean_absolute_error(y_true, y_pred)  # 计算平均绝对误差
    n = len(y_true)  # 获取样本数量
    slope, intercept, _, _, _ = linregress(y_true, y_pred)  # 计算线性回归的斜率和截距
    y_pred_series = pd.Series(y_pred, index=y_true.index)  # 将预测结果转换为pandas Series以对齐索引
    color_map = {
        "华北地区": "#1f77b4",
        "华南地区": "#ff7f0e",
        "西北地区": "#2ca02c",
    }  # 定义不同分组的颜色
    for group_name, color in color_map.items():  # 遍历每个分组
        mask = groups == group_name  # 创建一个布尔掩码，选择当前分组的数据
        ax.scatter(
            y_true[mask],
            y_pred_series[mask],
            c=color,
            label=group_name,
            alpha=0.7,
            s=50,
            edgecolors="w",
        )  # 绘制当前分组的散点图
    overall_min = min(y_true.min(), y_pred.min())  # 计算所有数据的最小值
    overall_max = max(y_true.max(), y_pred.max())  # 计算所有数据的最大值
    buffer = (overall_max - overall_min) * 0.05  # 计算坐标轴的缓冲范围
    plot_lim_min = overall_min - buffer  # 定义坐标轴的最小值
    plot_lim_max = overall_max + buffer  # 定义坐标轴的最大值
    ax.set_xlim(plot_lim_min, plot_lim_max)  # 设置x轴范围
    ax.set_ylim(plot_lim_min, plot_lim_max)  # 设置y轴范围
    ax.plot(
        [plot_lim_min, plot_lim_max],
        [plot_lim_min, plot_lim_max],  # 绘制1:1对角线
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="1:1 Line",
    )
    x_fit = np.array([y_true.min(), y_true.max()])  # 创建用于绘制拟合线的数据点
    ax.plot(
        x_fit,
        slope * x_fit + intercept,  # 绘制回归拟合线
        color="#D32F2F",
        linewidth=2.5,
        label="Fitted Line",
    )
    stats_text = (
        f"y = {intercept:.2f} + {slope:.2f}x\n"  # 准备要显示的统计文本
        f"$R^2$ = {r2:.2f}\n"
        f"RMSE = {rmse:.2f}\n"
        f"MAE = {mae:.2f}\n"
        f"N = {n}"
    )
    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=14,  # 在图上添加统计文本框
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FAF0E6", alpha=0.85, edgecolor="none"),
    )
    ax.set_xlabel("真实值 (Measured)", fontsize=16, fontweight="bold")  # 设置x轴标签
    ax.set_ylabel("预测值 (Estimated)", fontsize=16, fontweight="bold")  # 设置y轴标签
    ax.set_title(title, fontsize=16, fontweight="bold", pad=16)  # 设置图表标题
    legend = ax.legend(loc="lower right", frameon=False, fontsize=14)  # 显示图例
    ax.tick_params(
        axis="both", which="major", length=4, width=1.5, labelsize=16
    )  # 自定义坐标轴刻度样式
    for label in ax.get_xticklabels() + ax.get_yticklabels():  # 遍历所有刻度标签
        label.set_fontweight("bold")  # 将刻度标签设置为粗体
    for spine in ax.spines.values():  # 遍历图表的四个边框
        spine.set_linewidth(1.5)  # 设置边框粗细
    ax.grid(True)  # 显示网格线
    ax.set_aspect("equal", "box")  # 设置坐标轴比例为1:1
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=300)  # 保存图表到指定路径
    print(f"图表已保存到: {save_path}")  # 打印保存路径
    plt.close(fig)  # 关闭当前图形，防止在循环中打开过多窗口


def plot_residuals(y_true, y_pred, model_name, save_path):  # 定义函数，用于绘制残差图
    """绘制区分异常值的残差图"""
    residuals = y_true - y_pred  # 计算残差（真实值 - 预测值）
    std_dev = np.std(residuals)  # 计算残差的标准差
    outlier_threshold = 1.96 * std_dev  # 定义异常值阈值（约95%置信区间）

    is_outlier = np.abs(residuals) > outlier_threshold  # 创建一个布尔掩码，标记异常值

    fig, ax = plt.subplots(figsize=(10, 6))  # 创建图形和坐标轴

    ax.scatter(
        y_pred[~is_outlier], residuals[~is_outlier], c="green", alpha=0.6, label="Simple"
    )  # 绘制正常值的散点（绿色）
    ax.scatter(
        y_pred[is_outlier],
        residuals[is_outlier],
        c="red",
        alpha=0.8,
        edgecolors="k",
        label="Outliers",
    )  # 绘制异常值的散点（红色）

    ax.axhline(y=0, color="red", linestyle="--")  # 绘制y=0的参考线
    ax.fill_between(
        [min(y_pred), max(y_pred)],
        -std_dev,
        std_dev,  # 绘制标准范围的阴影区域
        color="yellow",
        alpha=0.3,
        label=f"Standard Range={std_dev:.2f}",
    )

    ax.set_xlabel("Predicted Values", fontsize=14)  # 设置x轴标签
    ax.set_ylabel("Residuals", fontsize=14)  # 设置y轴标签
    ax.set_title(f"({model_name}) Residual Plot", fontsize=16)  # 设置图表标题
    ax.legend()  # 显示图例
    ax.grid(True)  # 显示网格线
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=300)  # 保存图表
    print(f"残差图已保存到: {save_path}")  # 打印保存路径
    plt.close(fig)  # 关闭当前图形


def plot_shap_summary(
    model, X, feature_names, model_name, save_path
):  # 定义函数，用于绘制SHAP摘要图
    """绘制带背景色的SHAP特征重要性图"""
    print(f"正在为 {model_name} 进行 SHAP 分析...")  # 打印提示信息

    if any(
        s in str(type(model)).lower() for s in ["gradient", "randomforest", "xgb"]
    ):  # 判断是否为树模型
        explainer = shap.TreeExplainer(model)  # 如果是，使用高效的TreeExplainer
    else:  # 否则为其他模型
        if not hasattr(model, "predict"):  # 检查模型是否有predict方法
            # 为 SVR 等没有 predict 方法但有 decision_function 的模型添加 predict
            model.predict = lambda x: model.decision_function(x)
        X_summary = shap.sample(
            X, 100, random_state=42
        )  # 从数据中抽样作为背景数据，以加速KernelExplainer
        explainer = shap.KernelExplainer(model.predict, X_summary)  # 使用通用的KernelExplainer

    shap_values = explainer.shap_values(X)  # 计算SHAP值

    shap.summary_plot(
        shap_values, X, feature_names=feature_names, show=False
    )  # 生成SHAP摘要图，但不立即显示

    ax = plt.gca()  # 获取当前的坐标轴对象
    ax.axvspan(
        ax.get_xlim()[0], 0, color="lightblue", alpha=0.2, zorder=0
    )  # 在SHAP值小于0的区域添加浅蓝色背景
    ax.axvspan(
        0, ax.get_xlim()[1], color="lightpink", alpha=0.2, zorder=0
    )  # 在SHAP值大于0的区域添加浅粉色背景

    plt.title(f"{model_name} SHAP Summary", fontsize=16)  # 设置图表标题
    fig = plt.gcf()  # 获取当前的图形对象
    fig.tight_layout()  # 自动调整布局
    fig.savefig(save_path, dpi=300)  # 保存图表
    print(f"SHAP 图表已保存到: {save_path}")  # 打印保存路径
    plt.close(fig)  # 关闭当前图形


def add_noise(X, noise_level=0.05):  # 定义函数，为特征数据添加噪声
    """为数据集特征加入高斯噪声"""
    X_noisy = X.copy()  # 复制原始数据以避免修改原数据
    for col in X_noisy.columns:  # 遍历每一列特征
        std = X_noisy[col].std()  # 计算该列的标准差
        if std > 0:  # 检查标准差是否大于0，避免对常数特征添加噪声
            noise = np.random.normal(
                0, noise_level * std, size=X_noisy[col].shape
            )  # 生成符合正态分布的噪声
            X_noisy[col] += noise  # 将噪声添加到特征数据上
    return X_noisy  # 返回添加噪声后的数据


def print_noisy_evaluation_metrics(
    model, X_train, y_train, X_test, y_test, groups_test
):  # 定义函数，在带噪数据上评估模型
    """在加入噪声的数据上评估模型并打印指标"""
    print("\n--- 在 5% 噪声数据上的性能评估 ---")  # 打印标题

    X_train_noisy = add_noise(X_train)  # 为训练集添加噪声
    X_test_noisy = add_noise(X_test)  # 为测试集添加噪声

    model.fit(X_train_noisy, y_train)  # 使用带噪声的训练集重新训练模型
    y_pred_noisy = model.predict(X_test_noisy)  # 在带噪声的测试集上进行预测

    r2 = r2_score(y_test, y_pred_noisy)  # 计算R²分数
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_noisy))  # 计算均方根误差
    mae = mean_absolute_error(y_test, y_pred_noisy)  # 计算平均绝对误差

    residuals_df = pd.DataFrame(
        {"residuals": y_test - y_pred_noisy, "group": groups_test}
    )  # 创建包含残差和分组信息的DataFrame
    group_mean_residuals = residuals_df.groupby("group")["residuals"].mean()  # 按组计算平均残差
    between_group_variance = group_mean_residuals.var()  # 计算组间平均残差的方差

    print(f"  R2: {r2:.4f}")  # 打印R²
    print(f"  RMSE: {rmse:.4f}")  # 打印RMSE
    print(f"  MAE: {mae:.4f}")  # 打印MAE
    print(
        f"  组间残差均值方差: {between_group_variance:.4f} (越小表示模型对各区域越公平)"
    )  # 打印组间方差


def plot_all_learning_curves(all_curves_data, save_path):  # 定义函数，绘制所有模型的学习曲线
    """在无噪声数据上，绘制所有模型的学习曲线对比图"""
    plt.figure(figsize=(12, 10))  # 创建一个较大的图形

    colors = matplotlib.colormaps["tab10"]  # 使用新的API获取颜色映射，避免警告

    min_score, max_score = 1.0, -2.0  # 初始化Y轴范围的最小和最大值

    for i, (model_name, train_sizes, train_scores, test_scores) in enumerate(
        all_curves_data
    ):  # 遍历每个模型的数据
        test_scores_mean = np.mean(test_scores, axis=1)  # 计算交叉验证得分的平均值
        if test_scores_mean.min() < min_score:
            min_score = test_scores_mean.min()  # 更新Y轴最小值
        if test_scores_mean.max() > max_score:
            max_score = test_scores_mean.max()  # 更新Y轴最大值

        train_scores_mean = np.mean(train_scores, axis=1)  # 计算训练得分的平均值
        train_scores_std = np.std(train_scores, axis=1)  # 计算训练得分的标准差
        test_scores_std = np.std(test_scores, axis=1)  # 计算交叉验证得分的标准差

        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color=colors(i),  # 绘制训练得分曲线
            label=f"{model_name} Training",
        )
        plt.fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,  # 绘制训练得分的标准差范围
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color=colors(i),
        )

        plt.plot(
            train_sizes,
            test_scores_mean,
            "s-",
            color=colors(i),
            alpha=0.7,  # 绘制交叉验证得分曲线
            label=f"{model_name} Cross-validation",
        )
        plt.fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,  # 绘制交叉验证得分的标准差范围
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color=colors(i),
        )

    plt.title("Learning Curves for All Models", fontsize=18)  # 设置图表标题
    plt.xlabel("Training examples", fontsize=14)  # 设置x轴标签
    plt.ylabel("$R^2$", fontsize=14)  # 设置y轴标签

    padding = (max_score - min_score) * 0.1  # 计算Y轴的缓冲范围
    y_min = min_score - padding  # 确定Y轴最小值
    y_max = min(1.05, max_score + padding)  # 确定Y轴最大值，但不超过1.05
    plt.ylim(y_min, y_max)  # 设置Y轴范围

    plt.legend(loc="best")  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.tight_layout()  # 自动调整布局
    plt.savefig(save_path, dpi=300)  # 保存图表
    print(f"\n所有模型的学习曲线对比图已保存到: {save_path}")  # 打印保存路径
    plt.close("all")  # 关闭所有图形窗口


# -------------------------------------------------------------------
# 第 5 步: 主循环
# -------------------------------------------------------------------
all_learning_curves_data = []  # 初始化一个列表，用于存储所有模型的学习曲线数据

for model_config in models_to_run:  # 开始遍历 `models_to_run` 列表中的每一个模型配置
    model_name = model_config["name"]  # 获取模型名称
    estimator = model_config["estimator"]  # 获取模型估算器实例
    param_grid = model_config["param_grid"]  # 获取超参数网格
    needs_scaling = model_config["needs_scaling"]  # 获取是否需要数据标准化的标志

    print("\n" + "=" * 80)  # 打印分隔线
    print(f"🚀 开始处理模型: {model_name}")  # 打印当前正在处理的模型名称
    print("=" * 80)  # 打印分隔线

    if needs_scaling:  # 判断当前模型是否需要标准化数据
        X_train_to_use, X_val_to_use, X_test_to_use, X_train_val_to_use = (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            X_train_val_scaled,
        )  # 使用标准化后的数据
    else:  # 如果不需要
        X_train_to_use, X_val_to_use, X_test_to_use, X_train_val_to_use = (
            X_train,
            X_val,
            X_test,
            X_train_val,
        )  # 使用原始数据

    print(f"\n开始为 {model_name} 进行超参数搜索...")  # 打印提示信息
    start_time = time.time()  # 记录开始时间
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=SEARCH_CV,
        scoring="r2",
        n_jobs=WORKERS,
        verbose=1,
    )  # 配置网格搜索对象
    grid_search.fit(X_train_to_use, y_train)  # 在训练数据上执行网格搜索
    end_time = time.time()  # 记录结束时间
    print(f"\n✅ {model_name} 超参数搜索完成！耗时: {end_time - start_time:.2f} 秒")  # 打印耗时
    print(f"找到的最佳参数组合: {grid_search.best_params_}")  # 打印找到的最佳超参数
    print(f"在交叉验证中得到的最佳 R² 分数: {grid_search.best_score_:.4f}")  # 打印最佳交叉验证分数

    best_model = grid_search.best_estimator_  # 获取经过优化的最佳模型实例

    print(f"\n--- 在无噪声数据上进行标准绘图 ---")  # 打印提示信息
    y_train_pred = best_model.predict(X_train_to_use)  # 在训练集上进行预测
    y_val_pred = best_model.predict(X_val_to_use)  # 在验证集上进行预测
    y_test_pred = best_model.predict(X_test_to_use)  # 在测试集上进行预测

    plot_prediction_results(
        y_train,
        y_train_pred,
        groups_train,
        f"{model_name} 模型在训练集上的表现",  # 绘制训练集预测结果图
        os.path.join(output_folder, f"{model_name}_performance_TRAIN.png"),
    )
    plot_prediction_results(
        y_val,
        y_val_pred,
        groups_val,
        f"{model_name} 模型在验证集上的表现",  # 绘制验证集预测结果图
        os.path.join(output_folder, f"{model_name}_performance_VALIDATION.png"),
    )
    plot_prediction_results(
        y_test,
        y_test_pred,
        groups_test,
        f"{model_name} 模型在测试集上的表现",  # 绘制测试集预测结果图
        os.path.join(output_folder, f"{model_name}_performance_TEST.png"),
    )

    plot_residuals(
        y_test, y_test_pred, model_name, os.path.join(output_folder, f"{model_name}_residuals.png")
    )  # 绘制残差图

    plot_shap_summary(
        best_model,
        X_test_to_use,
        feature_names=X_test.columns,
        model_name=model_name,  # 绘制SHAP摘要图
        save_path=os.path.join(output_folder, f"{model_name}_shap_summary.png"),
    )

    # 打印带噪声的评估指标
    print_noisy_evaluation_metrics(
        best_model,  # 传入最佳模型
        X_train=X_train_val_to_use,
        y_train=y_train_val,  # 传入完整训练验证集用于带噪训练
        X_test=X_test_to_use,
        y_test=y_test,  # 传入测试集用于带噪评估
        groups_test=groups_test,  # 传入测试集的分组信息
    )

    # 计算无噪声的学习曲线数据以供最后绘图
    print(f"\n正在为 {model_name} 计算(无噪声)学习曲线数据...")  # 打印提示信息
    train_sizes, train_scores, test_scores = learning_curve(  # 调用sklearn的学习曲线函数
        estimator=best_model,
        X=X_train_val_to_use,
        y=y_train_val,  # 传入模型和完整训练验证集
        cv=SEARCH_CV,
        n_jobs=WORKERS,
        train_sizes=CURVE_SIZES,
        scoring="r2",  # 配置交叉验证、并行数、训练集大小划分和评分标准
    )
    all_learning_curves_data.append(
        (model_name, train_sizes, train_scores, test_scores)
    )  # 将计算结果存入列表
    print(f"✅ {model_name} 学习曲线数据计算完成！")  # 打印完成信息

# -------------------------------------------------------------------
# 第 6 步: 绘制学习曲线对比图
# -------------------------------------------------------------------
plot_all_learning_curves(
    all_learning_curves_data, os.path.join(output_folder, "ALL_MODELS_learning_curves.png")
)  # 调用函数绘制所有模型的学习曲线

print("\n🎉🎉🎉 所有模型处理及分析完毕！🎉🎉🎉")  # 打印最终完成信息
print(f"所有结果图表已保存至: {output_folder}")  # 告知用户结果保存位置
print("=" * 80)  # 打印分隔线

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

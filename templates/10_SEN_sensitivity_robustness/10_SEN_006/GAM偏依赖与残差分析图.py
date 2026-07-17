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

# ==============================================================================
# 步骤 0: 导入所有需要的库
# ==============================================================================
import matplotlib  # 导入matplotlib库，用于绘图。

matplotlib.use("Agg")  # 设置matplotlib的图形渲染后端为'TkAgg'，以确保在不同系统上的兼容性。
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，并简写为plt，用于创建图表。
import numpy as np  # 导入NumPy库，并简写为np，用于高效的数值计算。
import pandas as pd  # 导入Pandas库，并简写为pd，用于数据处理和分析，特别是读取Excel文件。
from pygam import (  # 从pygam库导入线性广义加性模型(LinearGAM)以及样条函数(s)和线性项(l)。
    LinearGAM,
    l,
    s,
)
from scipy.stats import gaussian_kde  # 从SciPy库导入高斯核密度估计函数，用于密度散点图。
from sklearn.metrics import (  # 从scikit-learn库导入多种模型性能评估指标。
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (  # 从scikit-learn库导入数据划分和学习曲线的工具。
    learning_curve,
    train_test_split,
)

# --- 全局字体设置 ---
plt.rcParams["font.family"] = (
    "Times New Roman"  # 设置所有matplotlib图表的全局字体为'Times New Roman'。
)
print("--- 正在从Excel文件加载数据... ---")  # 在控制台输出提示信息，表示开始加载数据。
# 定义Excel文件路径
excel_filepath = str(
    DATA_DIR / "LST_simulation_data.xlsx"
)  # 定义一个字符串变量，存储Excel文件的完整路径。
try:  # 开始一个try-except代码块，用于捕获和处理可能发生的错误。
    df = pd.read_excel(
        excel_filepath
    )  # 使用pandas的read_excel函数读取指定路径的Excel文件，并存入名为df的DataFrame中。
    print(f"数据已成功从 {excel_filepath} 加载。")  # 如果文件成功加载，打印确认信息和文件路径。
    print("数据预览:\n", df.head())  # 打印DataFrame的前5行，以便快速查看数据内容和结构。
    print("\n")  # 在控制台输出一个空行，使输出格式更清晰。
except FileNotFoundError:  # 如果在try代码块中发生了“文件未找到”的错误，则执行以下代码。
    print(f"错误: 文件未找到，请检查路径 '{excel_filepath}'")  # 打印文件未找到的错误提示。
    exit()  # 终止程序的执行。
except Exception as e:  # 捕获除了文件未找到之外的所有其他类型的异常。
    print(f"加载Excel文件时发生错误: {e}")  # 打印一个通用的错误信息，并显示具体的异常内容。
    exit()  # 终止程序的执行。

# ==============================================================================
# --- 步骤 2: 动态确定特征(X)和目标(Y) ---
# ==============================================================================
# 根据您的要求，不再手动指定变量列表，而是根据列的位置自动确定。
# 假定: 最后一列是目标变量(Y)，其余所有列是特征变量(X)。
if df.shape[1] < 2:  # 检查DataFrame的列数是否小于2。
    print("错误: Excel文件至少需要包含两列 (一个特征, 一个目标)。")  # 如果列数不足，打印错误信息。
    exit()  # 终止程序的执行。

y = df.iloc[:, -1]  # 使用iloc基于整数位置的索引，选取所有行的最后一列(索引-1)作为目标变量Y。
X = df.iloc[:, :-1]  # 使用iloc，选取所有行以及从第一列到倒数第二列(索引:-1)的所有列作为特征变量X。

# 动态获取变量名称，以供后续绘图和分析使用
target_variable = y.name  # 从目标变量Series的.name属性中获取其原始的列名。
feature_variables = (
    X.columns.tolist()
)  # 从特征变量DataFrame的.columns属性中获取所有列名，并转换为一个列表。

print("--- 变量已根据列位置动态确定 ---")  # 打印提示信息，告知变量已确定。
print(f"目标变量 (y): '{target_variable}'")  # 打印已识别的目标变量名称。
print(f"特征变量 (X): {feature_variables}")  # 打印已识别的特征变量名称列表。
print("\n")  # 输出一个空行用于格式化。

# ==============================================================================
# --- 步骤 3: 划分训练集和测试集 ---
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)  # 使用train_test_split函数将X和Y划分为训练集和测试集。test_size=0.25表示测试集占25%。random_state=42确保每次划分结果都一样，便于复现。

# ==============================================================================
# --- 步骤 4: 构建并训练广义加性模型 (GAM) ---
# ==============================================================================
print("--- 正在构建和训练GAM模型... ---")  # 打印流程状态信息。

# 动态构建GAM公式
# s()表示平滑项, l()表示线性项。这里我们对所有特征使用平滑项，
# 并为第7个特征(Aspect，坡向)使用循环平滑样条'cp'，因为它是一个循环变量(0-360度)。
# 注意: 这个公式假定您有10个特征。如果特征数量变化，可能需要调整。
# 为了通用性，我们为所有特征都使用平滑项s()
gam_formula = s(0)  # 初始化GAM公式，对第一个特征（索引为0）应用一个平滑项s()。
for i in range(1, len(feature_variables)):  # 遍历除第一个以外的所有特征的索引。
    # 特别处理坡向（Aspect），如果它存在并且是循环的
    if feature_variables[i].lower() == "aspect":  # 检查当前特征的名称（转换为小写）是否为'aspect'。
        gam_formula += s(i, basis="cp")  # 如果是，则为该特征添加一个使用循环样条基('cp')的平滑项。
    else:  # 如果特征名称不是'aspect'。
        gam_formula += s(i)  # 为该特征添加一个标准的平滑项。

gam = LinearGAM(gam_formula).fit(
    X_train, y_train
)  # 使用构建好的公式初始化一个线性GAM模型，并立即使用训练数据(X_train, y_train)对其进行拟合。
print("模型训练完成。")  # 打印模型训练完成的消息。
print("\n")  # 输出一个空行。

# ==============================================================================
# --- 步骤 5: 模型评估与摘要 ---
# ==============================================================================
# 对测试集进行预测
y_pred = gam.predict(X_test)  # 使用训练好的GAM模型对测试集特征X_test进行预测，得到预测值y_pred。
mse = mean_squared_error(
    y_test, y_pred
)  # 计算测试集真实值y_test和预测值y_pred之间的均方误差(MSE)。
r2 = r2_score(y_test, y_pred)  # 计算决定系数(R²)，评估模型的拟合优度。
rmse = np.sqrt(mse)  # 通过对MSE开方来计算均方根误差(RMSE)。
mae = mean_absolute_error(y_test, y_pred)  # 计算平均绝对误差(MAE)。

print("--- 模型摘要 ---")  # 打印标题。
gam.summary()  # 调用模型的summary()方法，打印出详细的统计摘要，包括每个特征的系数、p值等。
print("\n")  # 输出一个空行。
print("--- 模型在测试集上的性能 ---")  # 打印标题。
print(f"均方误差 (MSE): {mse:.4f}")  # 打印计算出的MSE，格式化为保留4位小数。
print(f"均方根误差 (RMSE): {rmse:.4f}")  # 打印计算出的RMSE，格式化为保留4位小数。
print(f"平均绝对误差 (MAE): {mae:.4f}")  # 打印计算出的MAE，格式化为保留4位小数。
print(f"决定系数 (R²): {r2:.4f}")  # 打印计算出的R²，格式化为保留4位小数。
print("\n")  # 输出一个空行。


# ==============================================================================
# --- 步骤 6: 定义高级密度散点图函数 ---
# ==============================================================================
def plot_density_scatter(
    y_true, y_pred, rmse_val, mae_val, r2_val, n_val, save_path, name, target_name
):  # 定义一个函数，用于绘制高级密度散点图，并接收多个参数。
    """
    绘制带有高斯核密度着色的散点图。
    """
    plt.figure(figsize=(15, 12))  # 创建一个新的图形实例，并设置其尺寸为15x12英寸。

    # 将y_true从pandas Series转换为numpy array，以进行vstack
    y_true_np = y_true.to_numpy()  # 将pandas Series对象转换为NumPy数组，以便进行后续的数值计算。

    # 计算密度估计
    xy = np.vstack([y_true_np, y_pred])  # 将真实值和预测值两个一维数组垂直堆叠成一个2xN的二维数组。
    density = gaussian_kde(xy)(xy)  # 使用高斯核密度估计(KDE)计算每个(x, y)数据点的密度值。

    # 按密度排序，以便将密度最高的点绘制在最上层
    idx = density.argsort()  # 获取根据密度值从小到大排序的索引。
    x_sorted, y_sorted, density_sorted = (
        y_true_np[idx],
        y_pred[idx],
        density[idx],
    )  # 使用排序后的索引来重新排列x、y和密度值。

    # 绘制散点图
    scatter = plt.scatter(
        x_sorted, y_sorted, c=density_sorted, cmap="viridis", s=50, alpha=0.7
    )  # 绘制散点图，其中点的颜色(c)由其密度决定，使用'viridis'色谱，点大小(s)为50，透明度(alpha)为0.7。

    # 绘制1:1对角线
    min_val = (
        min(y_true.min(), y_pred.min()) - 5
    )  # 计算真实值和预测值中的最小值，并减去5作为绘图下限。
    max_val = (
        max(y_true.max(), y_pred.max()) + 5
    )  # 计算真实值和预测值中的最大值，并加上5作为绘图上限。
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="black",
        label="y=x Line",
        linewidth=3,
        linestyle="--",
    )  # 绘制一条从(min_val, min_val)到(max_val, max_val)的黑色虚线，作为y=x参考线。

    # 自定义 x 和 y 轴的范围和刻度
    axis_min = np.floor(min_val / 10) * 10  # 将坐标轴最小值向下取整到最近的10的倍数。
    axis_max = np.ceil(max_val / 10) * 10  # 将坐标轴最大值向上取整到最近的10的倍数。
    plt.xlim([axis_min, axis_max])  # 设置x轴的显示范围。
    plt.ylim([axis_min, axis_max])  # 设置y轴的显示范围。
    # 确保刻度步长至少为1，避免范围过小时出错
    step = max(1, np.round((axis_max - axis_min) / 5, -1))  # 计算一个合适的坐标轴刻度步长。
    plt.xticks(
        np.arange(axis_min, axis_max + 1, step=step), fontsize=40
    )  # 设置x轴的刻度位置和标签字体大小。
    plt.yticks(
        np.arange(axis_min, axis_max + 1, step=step), fontsize=40
    )  # 设置y轴的刻度位置和标签字体大小。

    # 设置刻度线、坐标轴标签和图框样式
    plt.tick_params(
        axis="both", direction="out", length=10, width=3
    )  # 自定义刻度线的样式（方向、长度、宽度）。
    plt.xlabel(f"Actual {target_name}", fontsize=40)  # 设置x轴的标签，并使用动态的目标变量名称。
    plt.ylabel(f"Predicted {target_name}", fontsize=40)  # 设置y轴的标签，并使用动态的目标变量名称。
    for spine in plt.gca().spines.values():  # 遍历图表的四个边框（spines）。
        spine.set_linewidth(3)  # 将每个边框的线宽设置为3。

    # 添加精度评估指标文本
    eval_metrics = f"RMSE: {rmse_val:.3f}\nMAE: {mae_val:.3f}\nR²: {r2_val:.3f}\nn: {n_val:.0f}"  # 创建一个包含所有评估指标的多行字符串。
    plt.text(
        0.95,
        0.05,
        eval_metrics,
        transform=plt.gca().transAxes,
        fontsize=40,
        verticalalignment="bottom",
        horizontalalignment="right",
    )  # 在图表的右下角（相对坐标0.95, 0.05）添加评估指标文本。
    plt.text(
        0.05,
        0.9,
        name,
        transform=plt.gca().transAxes,
        fontsize=40,
        verticalalignment="top",
        horizontalalignment="left",
    )  # 在图表的左上角（相对坐标0.05, 0.9）添加图的名称（如'Training Set'）。

    # 添加 colorbar
    cb = plt.colorbar(scatter, pad=0.03)  # 为散点图添加一个颜色条(colorbar)，并设置其与图的间距。
    cb.set_label("Point Density", fontsize=35)  # 设置颜色条的标签。
    cb.ax.tick_params(labelsize=30)  # 设置颜色条刻度标签的字体大小。
    cb.set_ticks(
        [density_sorted.min(), density_sorted.max()]
    )  # 设置颜色条的刻度只显示最小值和最大值。
    cb.set_ticklabels(["Low", "High"])  # 将颜色条的刻度标签设置为'Low'和'High'。

    # 保存并显示图形
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight"
    )  # 将当前图形保存到指定的路径，设置分辨率为300dpi，并自动裁剪边框。
    plt.close("all")  # Interactive display removed; assets were exported above.


# ==============================================================================
# --- 步骤 7: 可视化 - 调用高级密度散点图函数 ---
# ==============================================================================
print("--- 正在生成高级密度散点图... ---")  # 打印流程状态。
# 7.1 为训练集绘图
y_train_pred = gam.predict(X_train)  # 对训练集特征进行预测，以评估模型在训练集上的表现。
r2_train = r2_score(y_train, y_train_pred)  # 计算训练集的R²。
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 计算训练集的RMSE。
mae_train = mean_absolute_error(y_train, y_train_pred)  # 计算训练集的MAE。
train_save_path = str(
    OUTPUT_DIR / "Advanced_Density_Scatter_Train.png"
)  # 定义训练集散点图的保存路径。
plot_density_scatter(
    y_train,
    y_train_pred,
    rmse_train,
    mae_train,
    r2_train,
    len(y_train),
    train_save_path,
    "Training Set",
    target_variable,
)  # 调用绘图函数，为训练集数据生成并保存密度散点图。
print(f"--- 训练集高级密度散点图已保存至 {train_save_path} ---")  # 打印保存成功的消息。

# 7.2 为测试集绘图
test_save_path = str(
    OUTPUT_DIR / "Advanced_Density_Scatter_Test.png"
)  # 定义测试集散点图的保存路径。
plot_density_scatter(
    y_test, y_pred, rmse, mae, r2, len(y_test), test_save_path, "Test Set", target_variable
)  # 调用绘图函数，为测试集数据生成并保存密度散点图。
print(f"--- 测试集高级密度散点图已保存至 {test_save_path} ---")  # 打印保存成功的消息。

# ==============================================================================
# --- 步骤 8: 可视化 - 美化后的部分依赖图 ---
# ==============================================================================
fig1, axes = plt.subplots(2, 5, figsize=(22, 9))  # 创建一个包含2行5列共10个子图的图形窗口。
fig1.suptitle("GAM Partial Dependence Plots", fontsize=18)  # 为整个图形设置一个主标题。
axes = axes.ravel()  # 将2x5的二维子图数组展平为一维数组，方便循环。

for i, ax in enumerate(axes):  # 遍历每个子图及其索引。
    # 检查索引是否在特征数量范围内
    if i < len(feature_variables):  # 如果当前子图索引小于特征的总数，则进行绘图。
        pdp, conf_int = gam.partial_dependence(
            term=i, X=X_train, width=0.95
        )  # 计算第i个特征的部分依赖(pdp)和95%置信区间(conf_int)。

        # 使用X_train中对应列的值进行排序绘图
        feature_col = feature_variables[i]  # 获取当前特征的名称。
        sorted_indices = X_train[feature_col].argsort()  # 获取根据当前特征值排序的索引。

        # 使用 .iloc 确保索引对齐
        sorted_x = X_train[feature_col].iloc[sorted_indices]  # 根据排序索引获取排序后的特征值。
        sorted_pdp = pdp[sorted_indices]  # 根据排序索引获取排序后的部分依赖值。
        sorted_conf_int = conf_int[sorted_indices]  # 根据排序索引获取排序后的置信区间。

        ax.plot(sorted_x, sorted_pdp, "b-", lw=2)  # 在当前子图上绘制部分依赖曲线。
        ax.fill_between(
            sorted_x, sorted_conf_int[:, 0], sorted_conf_int[:, 1], color="blue", alpha=0.2
        )  # 填充置信区间区域，并设置透明度。
        ax.set_title(feature_variables[i], fontsize=12)  # 设置子图的标题为特征名称。
        ax.set_xlabel("Feature Value", fontsize=11)  # 设置子图的x轴标签。
        if i % 5 == 0:  # 如果是每行的第一个子图（索引为0或5）。
            ax.set_ylabel(
                f"Impact on {target_variable}", fontsize=11
            )  # 设置y轴标签，并使用动态目标名称。
        ax.grid(True, linestyle="--", alpha=0.6)  # 显示网格线。
    else:  # 如果特征数量少于10个。
        # 如果特征数量少于10个，则隐藏多余的子图
        ax.set_visible(False)  # 将多余的子图设置为不可见。

fig1.tight_layout(rect=[0, 0, 1, 0.96])  # 自动调整子图布局，防止重叠，并为主标题留出空间。
save_path_8 = str(
    OUTPUT_DIR / "GAM_Partial_Dependence_Plots_Styled.png"
)  # 定义部分依赖图的保存路径。
plt.savefig(save_path_8, dpi=300, bbox_inches="tight")  # 保存图形。
print(f"--- 美化后的部分依赖图已保存至 {save_path_8} ---")  # 打印保存成功的消息。

# ==============================================================================
# --- 步骤 9: 可视化 - 残差分析图 ---
# ==============================================================================
fig_residual, ax_residual = plt.subplots(
    figsize=(8, 6)
)  # 创建一个新的图形窗口和子图，用于绘制残差图。
residuals = y_test - y_pred  # 计算残差（真实值与预测值之差）。
std_dev = np.std(residuals)  # 计算残差的标准差。
outliers = (
    np.abs(residuals) > 3 * std_dev
)  # 判断哪些点的残差绝对值超过3倍标准差，将其标记为离群点。
colors = np.where(outliers, "red", "blue")  # 根据是否为离群点，为每个点分配颜色（红色或蓝色）。
ax_residual.scatter(
    y_pred, residuals, c=colors, alpha=0.6, edgecolors="k", s=50
)  # 绘制残差散点图，x轴为预测值，y轴为残差。
if np.any(outliers):  # 如果存在任何离群点。
    ax_residual.scatter(
        [], [], c="red", alpha=0.6, label=f"Outliers (>3σ)"
    )  # 添加一个虚拟的散点用于生成离群点的图例。
ax_residual.scatter(
    [], [], c="blue", alpha=0.6, label="Normal Points"
)  # 添加一个虚拟的散点用于生成正常点的图例。
ax_residual.axhline(y=0, color="r", linestyle="--", lw=2)  # 在y=0的位置绘制一条红色虚线作为参考。
ax_residual.set_xlabel(
    f"Predicted {target_variable}", fontsize=12
)  # 设置x轴标签，并使用动态目标名称。
ax_residual.set_ylabel("Residuals (Actual - Predicted)", fontsize=12)  # 设置y轴标签。
ax_residual.set_title("Residual Analysis Plot", fontsize=14)  # 设置图表标题。
ax_residual.legend()  # 显示图例。
ax_residual.grid(True, linestyle="--", alpha=0.6)  # 显示网格线。
fig_residual.tight_layout()  # 自动调整布局。
save_path_9 = str(OUTPUT_DIR / "GAM_Residual_Analysis.png")  # 定义残差图的保存路径。
plt.savefig(save_path_9, dpi=300)  # 保存图形。
print(f"--- 残差分析图已保存至 {save_path_9} ---")  # 打印保存成功的消息。

# ==============================================================================
# --- 步骤 10: 可视化 - 各因子独立回归拟合图 (Partial Residual Plots) ---
# ==============================================================================
fig_pr, axes_pr = plt.subplots(2, 5, figsize=(20, 8))  # 创建一个新的包含10个子图的图形窗口。
fig_pr.suptitle("Partial Residual Plots for Each Feature", fontsize=18)  # 设置整个图形的主标题。
axes_pr = axes_pr.ravel()  # 将子图数组展平为一维，方便循环。

# 使用上面计算过的训练集预测值y_train_pred
train_residuals = y_train - y_train_pred  # 计算训练集的残差。

for i, ax in enumerate(axes_pr):  # 遍历每个子图及其索引。
    # 检查索引是否在特征数量范围内
    if i < len(feature_variables):  # 如果当前子图索引小于特征的总数。
        pdp, conf_int = gam.partial_dependence(
            term=i, X=X_train, width=0.95
        )  # 再次计算第i个特征的部分依赖。
        partial_residuals = (
            train_residuals + pdp
        )  # 计算部分残差（等于普通残差加上该特征的部分依赖）。

        feature_col = feature_variables[i]  # 获取当前特征的名称。
        ax.scatter(
            X_train[feature_col], partial_residuals, edgecolor="k", facecolor="c", alpha=0.5, s=40
        )  # 绘制部分残差散点图。

        # 使用 .iloc 确保索引对齐
        sorted_indices = X_train[feature_col].argsort()  # 获取根据特征值排序的索引。
        sorted_x = X_train[feature_col].iloc[sorted_indices]  # 获取排序后的特征值。
        sorted_pdp = pdp[sorted_indices]  # 获取排序后的部分依赖值。

        ax.plot(sorted_x, sorted_pdp, "k-", lw=2.5)  # 叠加绘制部分依赖曲线。

        ax.set_xlabel(feature_col, fontsize=11)  # 设置x轴标签。
        if i % 5 == 0:  # 如果是每行的第一个子图。
            ax.set_ylabel(
                f"Partial Effect on {target_variable}", fontsize=11
            )  # 设置y轴标签，并使用动态目标名称。
        ax.grid(True, linestyle="--", alpha=0.4)  # 显示网格线。
    else:  # 如果特征数少于10。
        # 如果特征数量少于10个，则隐藏多余的子图
        ax.set_visible(False)  # 将多余的子图设置为不可见。

fig_pr.tight_layout(rect=[0, 0, 1, 0.96])  # 自动调整布局。
save_path_10 = str(OUTPUT_DIR / "GAM_Partial_Residual_Plots.png")  # 定义部分残差图的保存路径。
plt.savefig(save_path_10, dpi=300, bbox_inches="tight")  # 保存图形。
print(f"--- 部分残差图已保存至 {save_path_10} ---")  # 打印保存成功的消息。

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

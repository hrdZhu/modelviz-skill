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

import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,  # 分类报告
    cohen_kappa_score,  # Cohen's Kappa系数
    confusion_matrix,  # 混淆矩阵
    roc_auc_score,  # ROC曲线下面积
    roc_curve,  # ROC曲线
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=FutureWarning)
# ==============================================================================
# 数据准备
# ==============================================================================
print("------------------------------------- 1. 准备数据 ---------------------------------------")
# !!! 请将这里更改为您自己的Excel文件路径 !!!
file_path = str(DATA_DIR / "XGBoost_MultiClass_Output_Final/林型分类数据.xlsx")
df = pd.read_excel(file_path)
y = df.iloc[:, 1]
x_df = df.iloc[:, 2:]
# 获取所有特征的名称，并将其转换为一个列表
feature_names = x_df.columns.tolist()
# 使用numpy的unique函数获取y中的所有唯一值，然后计算其数量，即为类别数
num_classes = len(np.unique(y))
# !!! 请在这里指定您自己的类别名称，顺序应与标签值 0, 1, 2, ... 对应 !!!
# 例如，如果标签0代表'落叶阔叶林'，1代表'常绿阔叶林'，以此类推
class_names = [
    "落叶阔叶林",
    "常绿阔叶林",
    "针阔混交林",
    "常绿针叶林",
    "落叶针叶林",
    "竹林",
    "灌木林",
    "经济林",
]
# 检查用户指定的类别名称数量是否与数据中检测到的类别数量匹配
if len(class_names) != num_classes:
    # 如果不匹配，则抛出一个ValueError异常，并提示用户
    raise ValueError(
        f"指定的类别名称数量 ({len(class_names)}) 与数据中的类别数量 ({num_classes}) 不匹配。请检查 class_names 列表。"
    )
# 打印检测到的类别数量
print(f"检测到 {num_classes} 个类别。")
# 打印用户指定的类别名称
print(f"您指定的类别名称为: {class_names}")
# ==============================================================================
# 数据处理与模型训练
# ==============================================================================
# 打印当前执行步骤的提示信息
print(
    "------------------------------------- 2. 划分和处理数据 ---------------------------------------"
)
# 将数据集划分为训练集和测试集
# x_df: 特征数据, y: 标签数据
# test_size=0.3: 指定测试集占总数据的30%
# random_state=42: 设置随机种子，保证每次划分结果一致，便于复现
# stratify=y: 按y中的类别比例进行分层抽样，确保训练集和测试集中各类别比例与原数据一致
x_train_df, x_test_df, y_train, y_test = train_test_split(
    x_df, y, test_size=0.3, random_state=42, stratify=y
)
# 创建一个StandardScaler对象，用于对数据进行标准化（Z-score规范化）
scaler = StandardScaler()
# 对训练集的特征数据进行标准化
# fit_transform会先学习训练数据的均值和标准差，然后再对训练数据进行转换
X_train_scaled_df = pd.DataFrame(scaler.fit_transform(x_train_df), columns=feature_names)
# 对测试集的特征数据进行标准化
# transform使用在训练集上学习到的均值和标准差来转换测试数据，以防止数据泄露
X_test_scaled_df = pd.DataFrame(scaler.transform(x_test_df), columns=feature_names)
# 打印数据划分和标准化完成的消息
print("数据划分和标准化完成。")
print("------------------------------------- 3. 模型训练与保存 -----------------------------------")
# 定义XGBoost模型的超参数搜索网格，用于GridSearchCV
param_grid_xgb = {
    "n_estimators": [10, 20],  # 梯度提升树的数量（基学习器的数量）
    "max_depth": [1, 2, 3],  # 每棵树的最大深度
    "learning_rate": [0.05, 0.1],  # 学习率（步长）
    "subsample": [0.8, 0.9],  # 训练每棵树时，用于训练样本的采样比例
    "colsample_bytree": [0.8, 0.9],  # 训练每棵树时，用于选择特征的采样比例
}
# 设置GridSearchCV（网格搜索交叉验证）
# estimator: 指定要调优的模型，这里是XGBClassifier
#   - random_state=42: 保证模型结果可复现
#   - objective='multi:softprob': 指定多分类问题，输出每个样本属于各个类别的概率
#   - num_class=num_classes: 告诉模型类别的总数
#   - eval_metric='mlogloss': 指定评估指标为多分类对数损失
#   - use_label_encoder=False: 禁用旧的标签编码器，避免警告
# param_grid: 指定要搜索的超参数网格
# cv=5: 使用5折交叉验证
# n_jobs=-1: 使用所有可用的CPU核心并行运行，加快搜索速度
# scoring='roc_auc_ovr_weighted': 指定评估最终模型好坏的指标为加权的、一对多(One-vs-Rest)的ROC AUC分数
# verbose=0: 控制搜索过程中的信息输出量，0表示不输出
gd = GridSearchCV(
    estimator=xgb.XGBClassifier(
        random_state=42,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
    ),
    param_grid=param_grid_xgb,
    cv=5,
    n_jobs=-1,
    scoring="roc_auc_ovr_weighted",
    verbose=0,
)
# 打印开始搜索最佳超参数的消息
print("使用GridSearchCV搜索最佳超参数...")
# 在标准化的训练数据上执行网格搜索
gd.fit(X_train_scaled_df, y_train)
# 打印搜索完成的消息
print("搜索完成。")
# 打印在交叉验证中找到的最佳模型的加权ROC AUC分数，保留4位小数
print(f"交叉验证中的最佳加权ROC AUC分数: {gd.best_score_:.4f}")
# !!! 请修改为您自己的输出目录 !!!
model_save_dir = str(OUTPUT_DIR)
# 检查指定的输出目录是否存在，如果不存在，则创建它
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
# 使用os.path.join构建模型的完整保存路径，这样做可以跨平台兼容
model_path = os.path.join(
    model_save_dir,
    "XGB_MultiClassifier_model.joblib",
)
# 使用joblib.dump将网格搜索找到的最佳模型（gd.best_estimator_）保存到文件
joblib.dump(gd.best_estimator_, model_path)
# 打印模型已保存的消息和路径
print(f"最佳模型已保存至: {model_path}")
# 从文件加载刚才保存的模型，用于后续评估
loaded_model = joblib.load(model_path)


# ==============================================================================
# 可视化函数定义
# ==============================================================================
# 定义一个函数，用于绘制并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_labels, save_path, title):
    # 使用sklearn的confusion_matrix函数计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    # 创建一个matplotlib画布，大小为10x8英寸
    plt.figure(figsize=(10, 8))
    # 使用seaborn的heatmap函数绘制热力图来可视化混淆矩阵
    # annot=True: 在每个格子上显示数值
    # fmt='d': 数值的格式为整数
    # cmap='Blues': 使用蓝色系配色
    # xticklabels, yticklabels: 设置x轴和y轴的刻度标签为类别名称
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels
    )
    # 设置图表标题，并指定字体大小
    plt.title(title, fontsize=16)
    # 设置y轴标签，并指定字体大小
    plt.ylabel("真实标签", fontsize=12)
    # 设置x轴标签，并指定字体大小
    plt.xlabel("预测标签", fontsize=12)
    # 保存图像到指定路径
    # dpi=300: 设置图像分辨率为300点/英寸，保证清晰度
    # bbox_inches='tight': 自动调整边界框，防止标签被截断
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # 关闭当前画布，释放内存
    plt.close()
    # 打印保存路径信息
    print(f"混淆矩阵图已保存至: {save_path}")


# 定义一个函数，用于绘制并保存多分类ROC曲线
def plot_multiclass_roc_curve(y_true, y_proba, class_labels, save_path, title):
    # 创建一个matplotlib画布，大小为12x10英寸
    plt.figure(figsize=(12, 10))
    # 遍历每个类别，为其单独绘制一条ROC曲线
    for i, class_label in enumerate(class_labels):
        # 使用roc_curve函数计算当前类别的假正率（fpr）和真正率（tpr）
        # pos_label=i 指定当前的正类是哪一个
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, i], pos_label=i)
        # 计算当前类别的AUC值
        # 这里需要将y_true转换为二元形式（是当前类 vs 不是当前类）
        class_auc = roc_auc_score((y_true == i).astype(int), y_proba[:, i])
        # 使用plt.plot绘制ROC曲线，并在线条上标注类别名称和AUC值
        plt.plot(fpr, tpr, lw=2, label=f"{class_label} 的ROC曲线 (AUC = {class_auc:.3f})")
    # 绘制一条从(0,0)到(1,1)的虚线，代表随机猜测的性能基准
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # 设置x轴的范围为[0.0, 1.0]
    plt.xlim([0.0, 1.0])
    # 设置y轴的范围为[0.0, 1.05]，稍微留出一点顶部空间
    plt.ylim([0.0, 1.05])
    # 设置x轴标签和字体大小
    plt.xlabel("假正率", fontsize=14)
    # 设置y轴标签和字体大小
    plt.ylabel("真正率", fontsize=14)
    # 设置图表标题和字体大小
    plt.title(title, fontsize=16)
    # 显示图例，并将其放置在右下角，设置字体大小
    plt.legend(loc="lower right", fontsize=10)
    # 显示网格线
    plt.grid(True)
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # 关闭当前画布
    plt.close()
    # 打印保存路径信息
    print(f"多分类ROC曲线图已保存至: {save_path}")


# 定义一个函数，用于绘制并保存训练集和测试集的性能指标对比柱状图
def plot_metrics_comparison(train_metrics, test_metrics, save_path):
    # 将传入的训练集和测试集指标字典整合成一个pandas DataFrame
    metrics_df = pd.DataFrame({"训练集": train_metrics, "测试集": test_metrics})
    # 使用DataFrame的plot方法绘制柱状图
    # kind='bar': 指定图表类型为柱状图
    # figsize=(14, 8): 设置画布大小
    # colormap='viridis': 使用'viridis'配色方案
    # rot=0: 设置x轴刻度标签不旋转
    ax = metrics_df.plot(kind="bar", figsize=(14, 8), colormap="viridis", rot=0)
    # 设置图表标题和字体大小
    plt.title("模型整体性能对比", fontsize=18)
    # 设置y轴标签和字体大小
    plt.ylabel("分数", fontsize=14)
    # 设置y轴的范围和x轴刻度的字体大小
    plt.ylim(0, 1.05)
    plt.xticks(fontsize=12)
    plt.legend(fontsize=12)
    # 遍历每个柱子（patch），在其顶部添加数值标签
    for p in ax.patches:
        # 使用ax.annotate添加文本
        # p.get_height(): 获取柱子的高度
        # (p.get_x() + p.get_width() / 2., p.get_height()): 确定文本的位置
        ax.annotate(
            f"{p.get_height():.4f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=10,
        )
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # 关闭当前画布
    plt.close()
    # 打印保存路径信息
    print(f"整体性能指标对比图已保存至: {save_path}")


# 定义一个函数，用于绘制并保存每个类别的详细性能指标（精确率、召回率、F1分数）
def plot_per_class_metrics(report_dict, class_names, title, save_path):
    # 从classification_report生成的字典中，只提取与我们指定的类别相关的数据
    class_metrics = {k: v for k, v in report_dict.items() if k in class_names}
    # 将提取出的字典转换为pandas DataFrame，并进行转置，使类别名成为行索引
    df = pd.DataFrame(class_metrics).transpose()
    # 只保留'precision', 'recall', 'f1-score'这三列我们关心的指标
    df = df[["precision", "recall", "f1-score"]]
    # 重命名列名，使其在图表中显示为中文
    df.rename(
        columns={"precision": "精确率", "recall": "召回率", "f1-score": "F1分数"}, inplace=True
    )
    # 绘制分组柱状图
    ax = df.plot(kind="bar", figsize=(14, 8), rot=0, width=0.8)
    # 设置图表标题和坐标轴标签
    plt.title(title, fontsize=18)
    plt.ylabel("分数", fontsize=14)
    plt.xlabel("类别", fontsize=14)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=12)
    # 设置图例标题和字体，并添加y轴的网格线
    plt.legend(title="指标", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # 遍历每个柱子，在其顶部添加数值标签
    for p in ax.patches:
        # 添加数值标签，保留两位小数
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=9,
        )
    # 保存图像到指定路径
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # 关闭当前画布
    plt.close()
    # 打印保存路径信息
    print(f"各类别详细指标图已保存至: {save_path}")


# 定义一个主函数，用于在指定数据集上评估模型并生成所有报告和图表
def evaluate_and_plot(model, X_df, y_true, set_name, class_names, save_dir):
    # 打印当前正在评估的数据集名称（如“训练集”或“测试集”）
    print(f"\n---------------------- 在 [{set_name}] 上进行评估 ----------------------")
    # 检查保存图表的目录是否存在，如果不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 使用模型对输入特征进行预测，得到预测的类别
    y_pred_class = model.predict(X_df)
    # 使用模型对输入特征进行预测，得到每个样本属于每个类别的概率
    y_pred_proba = model.predict_proba(X_df)
    # 生成分类报告（字典格式），便于后续提取数据
    report_dict = classification_report(
        y_true, y_pred_class, target_names=class_names, output_dict=True
    )
    # 生成分类报告（文本格式），便于直接打印查看
    report_text = classification_report(y_true, y_pred_class, target_names=class_names)
    # 计算并汇总一系列整体性能指标
    overall_metrics = {
        "准确率": report_dict["accuracy"],
        "加权平均精确率": report_dict["weighted avg"]["precision"],
        "加权平均召回率": report_dict["weighted avg"]["recall"],
        "加权平均F1分数": report_dict["weighted avg"]["f1-score"],
        "科恩Kappa系数": cohen_kappa_score(y_true, y_pred_class),
        "加权ROC AUC": roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted"),
    }
    # 打印文本格式的分类报告
    print(f"---------- {set_name} 分类报告 ----------")
    print(report_text)
    # 打印汇总的整体性能指标
    print("----------------- 整体指标 -----------------")
    # 遍历字典，格式化输出每个指标的名称和值
    for name, score in overall_metrics.items():
        print(f"{name:<25}: {score:.4f}")
    # 调用之前定义的绘图函数，生成并保存混淆矩阵图
    plot_confusion_matrix(
        y_true,
        y_pred_class,
        class_names,
        os.path.join(save_dir, f"混淆矩阵_{set_name}.png"),
        f"混淆矩阵 ({set_name})",
    )
    # 调用之前定义的绘图函数，生成并保存多分类ROC曲线图
    plot_multiclass_roc_curve(
        y_true,
        y_pred_proba,
        class_names,
        os.path.join(save_dir, f"ROC曲线_{set_name}.png"),
        f"ROC曲线 ({set_name})",
    )
    # 调用之前定义的绘图函数，生成并保存各类别性能指标图
    plot_per_class_metrics(
        report_dict,
        class_names,
        f"各类别指标 ({set_name})",
        os.path.join(save_dir, f"各类别指标_{set_name}.png"),
    )
    # 函数返回计算出的整体性能指标字典，用于后续对比
    return overall_metrics


# ==============================================================================
# 模型评估与可视化执行
# ==============================================================================
# 在训练集上评估模型，并获取性能指标，图表保存在 'Plots_Train_Set' 文件夹中
train_metrics = evaluate_and_plot(
    loaded_model,
    X_train_scaled_df,
    y_train,
    "训练集",
    class_names,
    os.path.join(model_save_dir, "Plots_Train_Set"),
)
# 在测试集上评估模型，并获取性能指标，图表保存在 'Plots_Test_Set' 文件夹中
test_metrics = evaluate_and_plot(
    loaded_model,
    X_test_scaled_df,
    y_test,
    "测试集",
    class_names,
    os.path.join(model_save_dir, "Plots_Test_Set"),
)
# 调用绘图函数，绘制并保存训练集与测试集的性能对比图
plot_metrics_comparison(train_metrics, test_metrics, os.path.join(model_save_dir, "指标对比图.png"))
# ==============================================================================
# SHAP 模型可解释性分析
# ==============================================================================
# 打印开始SHAP分析的提示信息
print(
    "\n----------------------------------------开始SHAP分析-----------------------------------------"
)
# 定义一个用于保存SHAP相关图表的目录路径
shap_save_dir = os.path.join(model_save_dir, "SHAP_Plots")
# 检查该目录是否存在，如果不存在则创建
if not os.path.exists(shap_save_dir):
    os.makedirs(shap_save_dir)
# 打印SHAP图表的保存位置
print(f"SHAP相关图表将保存至: {shap_save_dir}")
# 打印正在创建SHAP解释器的消息
print("创建SHAP TreeExplainer...")
# 为树模型（如XGBoost, LightGBM等）创建一个SHAP解释器，传入我们训练好的模型
explainer = shap.TreeExplainer(loaded_model)
# 打印正在计算SHAP值的消息
print("计算SHAP值...")
# 在（标准化的）测试集上计算SHAP值
# 这会返回一个Explanation对象，其中包含了每个样本、每个特征对于每个类别的SHAP值
explanation = explainer(X_test_scaled_df)
# 打印SHAP解释对象已创建的消息
print("SHAP解释对象已创建。")
# --- 全局SHAP摘要图 ---
# 打印当前分析步骤的提示信息
print("\n----------------------全局SHAP分析 ----------------------")
# 打印正在绘制全局SHAP摘要图的消息
print("绘制全局SHAP摘要图...")
# SHAP的summary_plot可以直接接受多分类的Explanation对象，但为了更好地控制，我们也可以手动传入一个SHAP值列表
# 这里的列表推导式为每个类别提取出其对应的SHAP值矩阵
shap_values_list = [explanation.values[:, :, i] for i in range(explanation.values.shape[2])]
# 绘制所有类别叠加在一起的全局特征重要性摘要图（蜂群图）
# 这个图可以展示特征的全局重要性排序，以及特征值的大小和方向对模型输出的影响
shap.summary_plot(shap_values_list, X_test_scaled_df, class_names=class_names, show=False)
# 为图表设置标题
plt.title("全局SHAP特征重要性 (所有类别叠加)", fontsize=16)
# 定义保存路径
save_path_global_summary = os.path.join(shap_save_dir, str(OUTPUT_DIR / "SHAP_Summary_Global.png"))
# 保存图像
plt.savefig(save_path_global_summary, dpi=300, bbox_inches="tight")
# 关闭当前画布
plt.close()
# 打印图表已保存的消息
print(f"全局SHAP摘要图已保存至: {save_path_global_summary}")

# --- 各类别特征重要性与依赖图 ---
# 打印当前分析步骤的提示信息
print("\n---------------------- 各类别SHAP重要性与依赖性分析 ----------------------")
# 创建一个子目录，用于保存每个类别的SHAP分析图表
per_class_shap_dir = os.path.join(shap_save_dir, "Per_Class_Analysis")
# 检查该目录是否存在，如果不存在则创建
if not os.path.exists(per_class_shap_dir):
    os.makedirs(per_class_shap_dir)
# 打印各类别图表的保存位置
print(f"各类别SHAP图表将保存至: {per_class_shap_dir}")

# 首先，根据全局重要性获取排名靠前的特征，用于后续绘制依赖图
# 计算全局特征重要性：对所有样本和所有类别上的SHAP值的绝对值取平均，然后按特征求和
global_shap_importance = np.abs(explanation.values).mean(0).sum(1)
# 创建一个DataFrame，包含特征名称和其全局重要性，并按重要性降序排列
global_importance_df = pd.DataFrame(
    {"Feature": explanation.feature_names, "Importance": global_shap_importance}
).sort_values(by="Importance", ascending=False)
# 提取全局重要性排名前3的特征
top_features_for_dep_plots = global_importance_df["Feature"].tolist()[:3]
# 打印将要为哪些特征绘制依赖图
print(f"\n将为全局重要性排名前3的特征绘制依赖图: {top_features_for_dep_plots}")

# 遍历每个类别，进行单独的详细分析
for i, class_name in enumerate(class_names):
    # 清理类别名称中的空格，以便用作文件名
    class_name_safe = class_name.replace(" ", "_")
    # 打印当前正在分析的类别名称
    print(f"\n---------- 分析类别: {class_name} ----------")

    # 1. 计算并打印当前类别下各特征的重要性
    # 从explanation对象中提取当前类别的SHAP值矩阵
    class_shap_values = explanation.values[:, :, i]
    # 计算当前类别下每个特征的平均绝对SHAP值，作为其对该类别的重要性度量
    class_importance = np.mean(np.abs(class_shap_values), axis=0)
    # 创建一个DataFrame，包含特征名称和其对当前类别的重要性，并按重要性降序排列
    class_importance_df = pd.DataFrame(
        {"Feature": explanation.feature_names, "Importance": class_importance}
    ).sort_values(by="Importance", ascending=False)
    # 打印当前类别下重要性排名前10的特征及其重要性值
    print(f"'{class_name}' 的前10个最重要特征:")
    print(class_importance_df.head(10).to_string(index=False))

    # 2. 绘制当前类别的SHAP摘要图（蜂群图）
    # `show=False`表示不立即显示图像，而是等待后续保存
    shap.summary_plot(class_shap_values, X_test_scaled_df, show=False)
    # 设置图表标题
    plt.title(f'对"{class_name}"预测的SHAP特征影响', fontsize=16)
    # 定义保存路径
    save_path_beeswarm = os.path.join(per_class_shap_dir, f"SHAP_Beeswarm_{class_name_safe}.png")
    # 保存图像
    plt.savefig(save_path_beeswarm, dpi=300, bbox_inches="tight")
    # 关闭当前画布
    plt.close()
    # 打印图表已保存的消息
    print(f"'{class_name}' 的SHAP蜂群图已保存。")

    # 3. 为当前类别绘制一个平均绝对SHAP值的条形图
    # 创建新画布
    plt.figure(figsize=(10, 8))
    # 选择重要性排名前15的特征进行绘图
    top_n = 15
    # 提取排名前n的特征及其重要性
    df_to_plot = class_importance_df.head(top_n)
    # 绘制水平条形图；[::-1]将数据顺序反转，使最重要的特征显示在图表顶部
    plt.barh(df_to_plot["Feature"][::-1], df_to_plot["Importance"][::-1])
    # 设置X轴标签
    plt.xlabel("平均绝对SHAP值 (对该类别预测的影响)", fontsize=12)
    # 设置图表标题
    plt.title(f'"{class_name}" 的特征重要性', fontsize=16)
    # 调整布局以防止标签重叠或被截断
    plt.tight_layout()
    # 定义保存路径
    save_path_bar = os.path.join(per_class_shap_dir, f"SHAP_Bar_Chart_{class_name_safe}.png")
    # 保存图像
    plt.savefig(save_path_bar, dpi=300)
    # 关闭当前画布
    plt.close()
    # 打印图表已保存的消息
    print(f"'{class_name}' 的SHAP条形图已保存。")

# 最后，为之前选出的最重要的几个特征创建依赖图
print(f"\n正在为排名靠前的特征创建依赖图...")
# 遍历全局重要性最高的几个特征
for feature in top_features_for_dep_plots:
    # 对于每个重要特征，再遍历所有类别
    for i, class_name in enumerate(class_names):
        # 清理类别名称以用于文件名
        class_name_safe = class_name.replace(" ", "_")
        # 创建新画布
        plt.figure()
        # 绘制依赖图，展示选定特征(feature)的值如何影响模型对特定类别(i)的预测
        # interaction_index="auto": SHAP会自动选择一个与`feature`有最强交互作用的特征来进行着色，以揭示交互效应
        shap.dependence_plot(
            feature, shap_values_list[i], X_test_scaled_df, show=False, interaction_index="auto"
        )
        # 设置图表标题
        plt.title(f'SHAP 依赖图: {feature}\n(对 "{class_name}" 的影响)', fontsize=14)
        # 定义保存路径
        save_path_dep = os.path.join(
            per_class_shap_dir, f"SHAP_Dependence_{feature}_on_{class_name_safe}.png"
        )
        # 保存图像
        plt.savefig(save_path_dep, dpi=300, bbox_inches="tight")
        # 关闭当前画布，防止后续绘图叠加在同一张图上
        plt.close()

# 打印各类别依赖图已保存的消息
print(f"各类别依赖图已保存。")
# 打印所有分析和绘图完成的最终消息
print(
    "\n----------------------------------------所有分析和绘图完成-----------------------------------------"
)

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

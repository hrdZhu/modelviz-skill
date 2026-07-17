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
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,  # <--- 导入Kappa值
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# --- 初始化设置 ---
matplotlib.use("Agg")
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", category=FutureWarning)
# ==============================================================================
# 1. 数据准备与模型训练
# ==============================================================================
print("-------------------------------------准备数据---------------------------------------")
# !!! 请确保您的目标变量(y)是0和1的二元标签 !!!
file_path = str(DATA_DIR / "二分类数据.xlsx")
df = pd.read_excel(file_path)
y = df.iloc[:, 0]
x_df = df.iloc[:, 1:]
feature_names = x_df.columns.tolist()
print("-------------------------------------划分数据集---------------------------------------")
# 确保分层抽样，保持训练集和测试集中类别比例一致
x_train_df, x_test_df, y_train, y_test = train_test_split(
    x_df, y, test_size=0.3, random_state=42, stratify=y
)
print("-------------------------------------数据标准化---------------------------------------")
scaler = StandardScaler()
X_train_scaled_np = scaler.fit_transform(x_train_df)
X_train_scaled_df = pd.DataFrame(X_train_scaled_np, columns=feature_names)
X_test_scaled_np = scaler.transform(x_test_df)
X_test_scaled_df = pd.DataFrame(X_test_scaled_np, columns=feature_names)
print(
    "-------------------------------------定义XGBoost分类模型超参数范围-----------------------------------"
)
param_grid_xgb = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
}
print(
    "-------------------------------------搜索XGBoost最佳超参数-------------------------------------"
)
gd = GridSearchCV(
    estimator=xgb.XGBClassifier(
        random_state=0, objective="binary:logistic", eval_metric="logloss", use_label_encoder=False
    ),
    param_grid=param_grid_xgb,
    cv=5,
    n_jobs=-1,
    scoring="roc_auc",
    verbose=0,
)
gd.fit(X_train_scaled_df, y_train)
print("-------------------------------------输出最佳模型---------------------------------------")
print(f"在交叉验证中验证的最好结果 (ROC AUC): {gd.best_score_:.4f}")
print("最好的参数模型:", gd.best_estimator_)
print(
    "-------------------------------------保存和加载最佳模型---------------------------------------"
)
# !!! 请务必修改为您自己的保存路径 !!!
model_save_dir = str(OUTPUT_DIR)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
model_path = os.path.join(model_save_dir, "XGB_Classifier_model.joblib")
joblib.dump(gd.best_estimator_, model_path)
print(f"XGBoost分类模型已保存至: {model_path}")
loaded_model = joblib.load(model_path)
print("XGBoost分类模型已成功加载。")


# ==============================================================================
# 2. 评价指标的可视化函数 (可复用)
# ==============================================================================
def plot_confusion_matrix(cm, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["预测为 0", "预测为 1"],
        yticklabels=["实际为 0", "实际为 1"],
    )
    plt.title(title, fontsize=16)
    plt.ylabel("实际标签", fontsize=12)
    plt.xlabel("预测标签", fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵图已保存至: {save_path}")


def plot_roc_curve(y_true, y_proba, roc_auc, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC 曲线 (面积 = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假正例率", fontsize=14)
    plt.ylabel("真正例率", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC曲线图已保存至: {save_path}")


def plot_pr_curve(y_true, y_proba, avg_precision, save_path, title):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(
        recall_vals,
        precision_vals,
        color="royalblue",
        lw=2,
        label=f"P-R 曲线 (平均精确率 = {avg_precision:.2f})",
    )
    plt.xlabel("召回率", fontsize=14)
    plt.ylabel("精确率", fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"P-R曲线图已保存至: {save_path}")


def plot_metrics_bar_chart(metrics, save_path, title):
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    plt.ylim([0, 1.05])
    plt.ylabel("分数", fontsize=12)
    plt.title(title, fontsize=16)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom",
            ha="center",
            fontsize=12,
        )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"核心指标对比图已保存至: {save_path}")


# ==============================================================================
# 3. 对 验证集(Test Set) 进行评估与绘图
# ==============================================================================
print(
    "\n-------------------------------------对【验证集】进行评估---------------------------------------"
)
plot_save_dir_test = os.path.join(model_save_dir, "Plots_Test_Set")
if not os.path.exists(plot_save_dir_test):
    os.makedirs(plot_save_dir_test)
y_test_pred_class = loaded_model.predict(X_test_scaled_df)
y_test_pred_proba = loaded_model.predict_proba(X_test_scaled_df)[:, 1]
# 计算指标
test_accuracy = accuracy_score(y_test, y_test_pred_class)
test_precision = precision_score(y_test, y_test_pred_class)
test_recall = recall_score(y_test, y_test_pred_class)
test_f1 = f1_score(y_test, y_test_pred_class)
test_roc_auc = roc_auc_score(y_test, y_test_pred_proba)
test_kappa = cohen_kappa_score(y_test, y_test_pred_class)  # <--- 计算Kappa值
test_avg_precision = average_precision_score(y_test, y_test_pred_proba)
test_cm = confusion_matrix(y_test, y_test_pred_class)
# 打印报告
print(classification_report(y_test, y_test_pred_class))
print(f"Cohen's Kappa: {test_kappa:.4f}")
# 绘图
plot_confusion_matrix(
    test_cm, os.path.join(plot_save_dir_test, "Confusion_Matrix_Test.png"), "混淆矩阵 (验证集)"
)
plot_roc_curve(
    y_test,
    y_test_pred_proba,
    test_roc_auc,
    os.path.join(plot_save_dir_test, "ROC_Curve_Test.png"),
    "ROC 曲线 (验证集)",
)
plot_pr_curve(
    y_test,
    y_test_pred_proba,
    test_avg_precision,
    os.path.join(plot_save_dir_test, "PR_Curve_Test.png"),
    "P-R 曲线 (验证集)",
)
test_metrics = {
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1 Score": test_f1,
    "Kappa": test_kappa,
}
plot_metrics_bar_chart(
    test_metrics,
    os.path.join(plot_save_dir_test, "Metrics_Comparison_Test.png"),
    "核心分类指标对比 (验证集)",
)
# ==============================================================================
# 4. 对 训练集(Train Set) 进行评估与绘图
# ==============================================================================
print(
    "\n-------------------------------------对【训练集】进行评估---------------------------------------"
)
plot_save_dir_train = os.path.join(model_save_dir, "Plots_Train_Set")
if not os.path.exists(plot_save_dir_train):
    os.makedirs(plot_save_dir_train)
y_train_pred_class = loaded_model.predict(X_train_scaled_df)
y_train_pred_proba = loaded_model.predict_proba(X_train_scaled_df)[:, 1]
# 计算指标
train_accuracy = accuracy_score(y_train, y_train_pred_class)
train_precision = precision_score(y_train, y_train_pred_class)
train_recall = recall_score(y_train, y_train_pred_class)
train_f1 = f1_score(y_train, y_train_pred_class)
train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
train_kappa = cohen_kappa_score(y_train, y_train_pred_class)  # <--- 计算Kappa值
train_avg_precision = average_precision_score(y_train, y_train_pred_proba)
train_cm = confusion_matrix(y_train, y_train_pred_class)
# 打印报告
print(classification_report(y_train, y_train_pred_class))
print(f"Cohen's Kappa: {train_kappa:.4f}")
# 绘图
plot_confusion_matrix(
    train_cm, os.path.join(plot_save_dir_train, "Confusion_Matrix_Train.png"), "混淆矩阵 (训练集)"
)
plot_roc_curve(
    y_train,
    y_train_pred_proba,
    train_roc_auc,
    os.path.join(plot_save_dir_train, "ROC_Curve_Train.png"),
    "ROC 曲线 (训练集)",
)
plot_pr_curve(
    y_train,
    y_train_pred_proba,
    train_avg_precision,
    os.path.join(plot_save_dir_train, "PR_Curve_Train.png"),
    "P-R 曲线 (训练集)",
)
train_metrics = {
    "Accuracy": train_accuracy,
    "Precision": train_precision,
    "Recall": train_recall,
    "F1 Score": train_f1,
    "Kappa": train_kappa,
}
plot_metrics_bar_chart(
    train_metrics,
    os.path.join(plot_save_dir_train, "Metrics_Comparison_Train.png"),
    "核心分类指标对比 (训练集)",
)
# ==============================================================================
# 5. SHAP 模型可解释性分析
# ==============================================================================
print(
    "\n----------------------------------------开始 SHAP 分析-----------------------------------------"
)
shap_save_dir = os.path.join(model_save_dir, "SHAP_Plots")
if not os.path.exists(shap_save_dir):
    os.makedirs(shap_save_dir)
print(f"SHAP 相关图将保存到: {shap_save_dir}")
# 1. 创建SHAP explainer
# TreeExplainer 对XGBoost等树模型效率最高
print("创建 SHAP TreeExplainer...")
explainer = shap.TreeExplainer(loaded_model)
# 2. 计算SHAP值
# 我们通常在测试集上进行解释，以了解模型在未知数据上的行为
print("正在计算SHAP值 (在测试集上)...")
shap_values = explainer.shap_values(X_test_scaled_df)
print("SHAP值计算完成。")
# 对于二分类，SHAP值为模型输出（log-odds）的变化。正值推向类别1，负值推向类别0。
# --- 5.1 SHAP Summary Plot (条形图) ---
print("绘制 SHAP Summary Plot (条形图)...")
shap.summary_plot(shap_values, X_test_scaled_df, plot_type="bar", show=False)
plt.title("SHAP 特征重要性 (平均绝对SHAP值)", fontsize=16)
plt.savefig(
    os.path.join(shap_save_dir, str(OUTPUT_DIR / "SHAP_Summary_Bar.png")),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
# --- 5.2 SHAP Summary Plot (散点/蜂群图) ---
print("绘制 SHAP Summary Plot (散点分布图)...")
shap.summary_plot(shap_values, X_test_scaled_df, show=False)
plt.title("SHAP 特征影响概览", fontsize=16)
plt.savefig(
    os.path.join(shap_save_dir, str(OUTPUT_DIR / "SHAP_Summary_Scatter.png")),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
# --- 5.3 SHAP Dependence Plots ---
print("绘制 SHAP Dependence Plots (针对最重要的几个特征)...")
# 计算全局重要性以选择特征
mean_abs_shap = np.abs(shap_values).mean(0)
shap_importance_df = pd.DataFrame(
    list(zip(feature_names, mean_abs_shap)), columns=["Feature", "Importance"]
)
shap_importance_df.sort_values(by="Importance", ascending=False, inplace=True)
top_features_for_shap = shap_importance_df["Feature"].tolist()[:5]  # 选择最重要的5个特征
for feature in top_features_for_shap:
    shap.dependence_plot(
        feature, shap_values, X_test_scaled_df, interaction_index="auto", show=False
    )
    plt.title(f"SHAP 依赖图: {feature}", fontsize=14)
    plt.savefig(
        os.path.join(shap_save_dir, f"SHAP_Dependence_{feature}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()
# --- 5.4 SHAP Waterfall Plot ---
print("绘制 SHAP Waterfall Plot (针对验证集第一个样本)...")
# explainer.expected_value 是SHAP图的基线值
# 对于二分类，TreeExplainer的expected_value可能是一个数组，我们取第一个元素
base_value = (
    explainer.expected_value[0]
    if isinstance(explainer.expected_value, np.ndarray)
    else explainer.expected_value
)

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=X_test_scaled_df.iloc[0],
        feature_names=feature_names,
    ),
    show=False,
)
plt.title("SHAP 瀑布图 (单个样本预测解释)", fontsize=16)
plt.savefig(
    os.path.join(shap_save_dir, str(OUTPUT_DIR / "SHAP_Waterfall_Sample0.png")),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(
    "\n----------------------------------------所有分析与绘图完成-----------------------------------------"
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

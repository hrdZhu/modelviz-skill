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

# =========================================================================================
# ====================================== 1. 库的导入 =========================================
# =========================================================================================
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ======================================2.设置颜色库 =========================================
# =========================================================================================
color_schemes = {
    1: {
        "Logistic Regression": "#1f77b4",
        "SVM (RBF Kernel)": "#d62728",
        "Random Forest": "#ff7f0e",
        "Gradient Boosting": "#2ca02c",
        "diag_line": "navy",
        "xy_label": "#8B0000",
        "bg_color": "#F5F5DC",
    },
    2: {
        "Logistic Regression": "#e41a1c",
        "SVM (RBF Kernel)": "#377eb8",
        "Random Forest": "#4daf4a",
        "Gradient Boosting": "#984ea3",
        "diag_line": "black",
        "xy_label": "black",
        "bg_color": "white",
    },
    3: {
        "Logistic Regression": "#66c2a5",
        "SVM (RBF Kernel)": "#fc8d62",
        "Random Forest": "#8da0cb",
        "Gradient Boosting": "#e78ac3",
        "diag_line": "#444444",
        "xy_label": "#222222",
        "bg_color": "#F0F0F0",
    },
    4: {
        "Logistic Regression": "#268bd2",
        "SVM (RBF Kernel)": "#d33682",
        "Random Forest": "#cb4b16",
        "Gradient Boosting": "#859900",
        "diag_line": "#93a1a1",
        "xy_label": "#586e75",
        "bg_color": "#fdf6e3",
    },
    5: {
        "Logistic Regression": "#fbb4ae",
        "SVM (RBF Kernel)": "#b3cde3",
        "Random Forest": "#ccebc5",
        "Gradient Boosting": "#decbe4",
        "diag_line": "#007A8C",
        "xy_label": "#005662",
        "bg_color": "#E6F7FF",
    },
    6: {
        "Logistic Regression": "#1b9e77",
        "SVM (RBF Kernel)": "#d95f02",
        "Random Forest": "#7570b3",
        "Gradient Boosting": "#e7298a",
        "diag_line": "#555555",
        "xy_label": "#333333",
        "bg_color": "#EAEAF2",
    },
    7: {
        "Logistic Regression": "#7fc97f",
        "SVM (RBF Kernel)": "#beaed4",
        "Random Forest": "#fdc086",
        "Gradient Boosting": "#ffff99",
        "diag_line": "#BCAAA4",
        "xy_label": "#5D4037",
        "bg_color": "#FEFBF3",
    },
    8: {
        "Logistic Regression": "#8dd3c7",
        "SVM (RBF Kernel)": "#ffffb3",
        "Random Forest": "#bebada",
        "Gradient Boosting": "#fb8072",
        "diag_line": "#D98080",
        "xy_label": "#7D0000",
        "bg_color": "#FFF5F5",
    },
    9: {
        "Logistic Regression": "#a6cee3",
        "SVM (RBF Kernel)": "#1f78b4",
        "Random Forest": "#b2df8a",
        "Gradient Boosting": "#33a02c",
        "diag_line": "#2E8B57",
        "xy_label": "#006400",
        "bg_color": "#F0FFF4",
    },
    10: {
        "Logistic Regression": "#9467bd",
        "SVM (RBF Kernel)": "#8c564b",
        "Random Forest": "#e377c2",
        "Gradient Boosting": "#7f7f7f",
        "diag_line": "#8A2BE2",
        "xy_label": "#4B0082",
        "bg_color": "#E6E6FA",
    },
    11: {
        "Logistic Regression": "#80b1d3",
        "SVM (RBF Kernel)": "#fdb462",
        "Random Forest": "#b3de69",
        "Gradient Boosting": "#fccde5",
        "diag_line": "#C71585",
        "xy_label": "#8B008B",
        "bg_color": "#FFF0F5",
    },
    12: {
        "Logistic Regression": "#e41a1c",
        "SVM (RBF Kernel)": "#377eb8",
        "Random Forest": "#4daf4a",
        "Gradient Boosting": "#984ea3",
        "diag_line": "#4682B4",
        "xy_label": "#000080",
        "bg_color": "#F0F8FF",
    },
    13: {
        "Logistic Regression": "#a6cee3",
        "SVM (RBF Kernel)": "#1f78b4",
        "Random Forest": "#b2df8a",
        "Gradient Boosting": "#33a02c",
        "diag_line": "#A0522D",
        "xy_label": "#8B4513",
        "bg_color": "#FAF0E6",
    },
    14: {
        "Logistic Regression": "#b3e2cd",
        "SVM (RBF Kernel)": "#fdcdac",
        "Random Forest": "#cbd5e8",
        "Gradient Boosting": "#f4cae4",
        "diag_line": "#3CB371",
        "xy_label": "#2E8B57",
        "bg_color": "#F5FFFA",
    },
    15: {
        "Logistic Regression": "#fb9a99",
        "SVM (RBF Kernel)": "#e31a1c",
        "Random Forest": "#fdbf6f",
        "Gradient Boosting": "#ff7f00",
        "diag_line": "#DAA520",
        "xy_label": "#B8860B",
        "bg_color": "#FFFACD",
    },
    16: {
        "Logistic Regression": "#7f3b08",
        "SVM (RBF Kernel)": "#b35806",
        "Random Forest": "#e08214",
        "Gradient Boosting": "#fdb863",
        "diag_line": "#A0522D",
        "xy_label": "#8B4513",
        "bg_color": "#FDF5E6",
    },
    17: {
        "Logistic Regression": "#003f5c",
        "SVM (RBF Kernel)": "#58508d",
        "Random Forest": "#bc5090",
        "Gradient Boosting": "#ff6361",
        "diag_line": "#7F8C8D",
        "xy_label": "#2C3E50",
        "bg_color": "#F0F4F8",
    },
    18: {
        "Logistic Regression": "#d73027",
        "SVM (RBF Kernel)": "#fc8d59",
        "Random Forest": "#fee090",
        "Gradient Boosting": "#91bfdb",
        "diag_line": "#CD5C5C",
        "xy_label": "#800000",
        "bg_color": "#FFE4E1",
    },
    19: {
        "Logistic Regression": "#fed9a6",
        "SVM (RBF Kernel)": "#ffffcc",
        "Random Forest": "#e5d8bd",
        "Gradient Boosting": "#fddaec",
        "diag_line": "#BDB76B",
        "xy_label": "#556B2F",
        "bg_color": "#FFFFF0",
    },
    20: {
        "Logistic Regression": "#fb4934",
        "SVM (RBF Kernel)": "#b8bb26",
        "Random Forest": "#fabd2f",
        "Gradient Boosting": "#83a598",
        "diag_line": "#7c6f64",
        "xy_label": "#3c3836",
        "bg_color": "#fbf1c7",
    },
}
selected_scheme = 20  # 选择颜色方案
SELECTED_COLORS = color_schemes[selected_scheme]  # 获取颜色方案
# =========================================================================================
# ======================================3.绘图函数 =========================================
# =========================================================================================


def plot_roc_curves(y_true, all_scores, plot_title, colors_dict, filename=None):

    fig, ax = plt.subplots(figsize=(8, 7))  # 画布
    ax.set_facecolor(colors_dict["bg_color"])  # 设置的背景颜色
    # 网格线
    ax.grid(True, which="both", linestyle="--", linewidth=2, color="lightgrey", dashes=(3, 5))
    # 存储每个模型AUC值的文本标注位置
    auc_text_positions = {
        "Logistic Regression": (0.65, 0.95),
        "Random Forest": (0.6, 0.82),
        "Gradient Boosting": (0.35, 0.68),
        "SVM (RBF Kernel)": (0.5, 0.55),
    }

    # 循环绘制每条ROC曲线
    for name, scores in all_scores.items():  # 遍历每一个模型名称, 预测分数
        fpr, tpr, _ = roc_curve(y_true, scores)  # 计算假正率和真正率
        roc_auc = auc(fpr, tpr)  # 计算AUC
        line_color = colors_dict[name]  # 获取对应的线条颜色
        ax.plot(fpr, tpr, color=line_color, lw=3, label=f"{name}")  # 绘制ROC曲线

        # 在图上标注AUC值
        pos = auc_text_positions[name]  # 获取标注坐标位置
        # 在指定位置添加文本
        ax.text(
            pos[0],
            pos[1],
            f"AUC={roc_auc:.2f}",
            transform=ax.transAxes,
            fontsize=16,
            color=line_color,
            ha="center",
        )
        # 绘制对角虚线/参考线
    ax.plot([0, 1], [0, 1], color=colors_dict["diag_line"], lw=2, linestyle="--")
    ax.set_xlim([-0.02, 1.0])  # 设置X轴的范围
    ax.set_ylim([0, 1.05])  # 设置Y轴的范围

    ax.set_xlabel("Specificity", fontsize=18, color=colors_dict["xy_label"])  # x轴标题
    ax.set_ylabel("Sensitivity", fontsize=18, color=colors_dict["xy_label"])  # Y轴标题
    ax.set_title(plot_title, fontsize=20, loc="left", pad=10)  # 设置主图标题

    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))  # 设置X轴主刻度的间隔
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))  # 设置Y轴主刻度的间隔
    ax.tick_params(axis="both", which="major", labelsize=14)  # 主刻度标签的字体大小

    # 添加图例
    ax.legend(loc="lower right", fontsize=14, frameon=True, edgecolor="black")

    # 遍历图框
    for spine in ax.spines.values():
        spine.set_edgecolor("#6B8E23")  # 颜色
        spine.set_linewidth(2)  # 图框的粗细设置

    plt.tight_layout()  # 自动调整子图参数，使其填充整个图窗区域，防止标签重叠
    # 保存
    fig.savefig(str(OUTPUT_DIR / f"{filename}_{selected_scheme}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(OUTPUT_DIR / f"{filename}_{selected_scheme}.pdf"), bbox_inches="tight")


# =========================================================================================
# ======================================4.数据的加载与处理=========================================
# =========================================================================================
excel_filename = str(DATA_DIR / "simulated_senescence_data.xlsx")  # 原始数据路径

df = pd.read_excel(excel_filename, engine="openpyxl")  # 读取数据
# 提取所有特征和标签
y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values
# 划分数据集为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# =========================================================================================
# ======================================5.模型构建=========================================
# =========================================================================================
print(f"正在为 'Logistic Regression' 进行网格搜索")
# 初始化逻辑回归模型
model_lr = LogisticRegression(random_state=42, solver="liblinear")
# 逻辑回归的超参数搜索网格
param_grid_lr = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
# 配置网格搜索
grid_search_lr = GridSearchCV(
    estimator=model_lr, param_grid=param_grid_lr, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
)
# 执行网格搜索
grid_search_lr.fit(X_train_scaled, y_train)
best_model_lr = grid_search_lr.best_estimator_  # 最佳模型
print(f"'Logistic Regression' 网格搜索完成！最佳超参数: {grid_search_lr.best_params_}")
# 最佳逻辑回归模型预测正类的概率
scores_lr_train = best_model_lr.predict_proba(X_train_scaled)[:, 1]
scores_lr_val = best_model_lr.predict_proba(X_val_scaled)[:, 1]


print(f"\n--- 正在为 'SVM (RBF Kernel)' 进行网格搜索")
# 初始化SVC模型
model_svm = SVC(random_state=42, probability=True)
# 超参数搜索网格
param_grid_svm = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
# 网格搜索
grid_search_svm = GridSearchCV(
    estimator=model_svm, param_grid=param_grid_svm, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
)
grid_search_svm.fit(X_train_scaled, y_train)
# 最佳SVM模型
best_model_svm = grid_search_svm.best_estimator_
print(f"SVM最佳超参数: {grid_search_svm.best_params_}")
# 预测正类的概率
scores_svm_train = best_model_svm.predict_proba(X_train_scaled)[:, 1]
scores_svm_val = best_model_svm.predict_proba(X_val_scaled)[:, 1]


print(f"\n--- 正在为 'Random Forest' 进行网格搜索")
# 初始化RF模型
model_rf = RandomForestClassifier(random_state=42)
# 超参数搜索网格
param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]}
# 执行网格搜索
grid_search_rf = GridSearchCV(
    estimator=model_rf, param_grid=param_grid_rf, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
)
grid_search_rf.fit(X_train_scaled, y_train)
# 最佳RF模型
best_model_rf = grid_search_rf.best_estimator_
print(f"'RF最佳超参数: {grid_search_rf.best_params_}")
# 预测正类的概率
scores_rf_train = best_model_rf.predict_proba(X_train_scaled)[:, 1]
scores_rf_val = best_model_rf.predict_proba(X_val_scaled)[:, 1]


print(f"\n--- 正在为 'Gradient Boosting' 进行网格搜索 ---")
# 初始化GBT模型
model_gbt = GradientBoostingClassifier(random_state=42)
param_grid_gbt = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
# 执行网格搜索
grid_search_gbt = GridSearchCV(
    estimator=model_gbt, param_grid=param_grid_gbt, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
)
grid_search_gbt.fit(X_train_scaled, y_train)
# 最佳GBT模型
best_model_gbt = grid_search_gbt.best_estimator_
print(f"GBT最佳超参数: {grid_search_gbt.best_params_}")
# 预测正类的概率
scores_gbt_train = best_model_gbt.predict_proba(X_train_scaled)[:, 1]
scores_gbt_val = best_model_gbt.predict_proba(X_val_scaled)[:, 1]

# =========================================================================================
# ======================================6.绘图=========================================
# =========================================================================================
# 存储训练集上所有模型的预测分数
train_scores_dict = {
    "Logistic Regression": scores_lr_train,
    "SVM (RBF Kernel)": scores_svm_train,
    "Random Forest": scores_rf_train,
    "Gradient Boosting": scores_gbt_train,
}
# 存储验证集上所有模型的预测分数
val_scores_dict = {
    "Logistic Regression": scores_lr_val,
    "SVM (RBF Kernel)": scores_svm_val,
    "Random Forest": scores_rf_val,
    "Gradient Boosting": scores_gbt_val,
}
# 绘制训练集的ROC曲线
plot_roc_curves(
    y_train,
    train_scores_dict,
    "(C) ROC Curves on Training Set (Model Comparison)",
    SELECTED_COLORS,
    filename="train",
)
# 绘制验证集的ROC曲线
plot_roc_curves(
    y_val,
    val_scores_dict,
    "(C) ROC Curves on Validation Set (Model Comparison)",
    SELECTED_COLORS,
    filename="test",
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

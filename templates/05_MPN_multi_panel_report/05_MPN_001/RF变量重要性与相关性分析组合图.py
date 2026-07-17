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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

plt.rcParams["font.family"] = "Times New Roman"
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# =========================================================================================
# ======================================2.颜色库设置========================================
# =========================================================================================
selected_scheme = 19  # 选择配色方案
color_schemes = {
    0: ("viridis", "#440154"),
    1: ("plasma", "#f0f921"),
    2: ("inferno", "#fcffa4"),
    3: ("magma", "#fcfdbf"),
    4: ("cividis", "#fde725"),
    5: ("coolwarm", "#6788ee"),
    6: ("bwr", "#3a68b4"),
    7: ("RdBu_r", "#2166ac"),
    8: ("PiYG", "#8e0152"),
    9: ("PRGn", "#40004b"),
    10: ("BrBG", "#543005"),
    11: ("PuOr", "#b35806"),
    12: ("RdYlBu", "#313695"),
    13: ("RdYlGn", "#1a9641"),
    14: ("Spectral", "#9e0142"),
    15: ("terrain", "#33a02c"),
    16: ("ocean", "#99d8c9"),
    17: ("gist_earth", "#a19342"),
    18: ("gnuplot", "#f6c940"),
    19: ("cubehelix", "#d4e157"),
}
# =========================================================================================
# ======================================3.数据的读取及预处理========================================
# =========================================================================================
# 原始数据的路径
file_path = str(DATA_DIR / "data.xlsx")

X = pd.read_excel(file_path, sheet_name="Environmental Variables")  # 特征
Y = pd.read_excel(file_path, sheet_name="COG Categories")  # 目标

# 从读取的数据中动态获取标签列表
env_vars_labels = X.columns.tolist()  # 获取特征名称
# env_vars_labels.reverse()  #顺序反转
cog_cats_labels = Y.columns.tolist()  # 目标名称

# =========================================================================================
# ======================================4.分析========================================
# =========================================================================================
importances_df = pd.DataFrame(index=env_vars_labels, columns=cog_cats_labels)  # 用于存储特征重要性
correlations_df = pd.DataFrame(index=env_vars_labels, columns=cog_cats_labels)  # 用于存储相关性系数
variation_explained = pd.Series(
    index=cog_cats_labels, dtype=float
)  # 用于存储模型解释的方差（R2分数）

for cog in cog_cats_labels:  # 逐个目标进行分析
    print(f"正在分析{cog}")
    X_ordered = X[env_vars_labels]
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_ordered, Y[cog], test_size=0.3, random_state=42
    )
    # 定义超参数网格
    param_grid = {"n_estimators": [50, 100], "max_depth": [10], "min_samples_leaf": [1, 2]}
    # 初始化RF回归器实例
    rf = RandomForestRegressor(random_state=42)
    gd = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring="r2"
    )  # 设置网格搜索
    # 在训练数据上执行网格搜索
    gd.fit(X_train, y_train)
    # 最佳模型
    best_rf = gd.best_estimator_
    variation_explained[cog] = best_rf.score(X_test, y_test)  # 计算R2
    print("R2", variation_explained[cog])

    explainer = shap.TreeExplainer(best_rf)  # 为训练好的最佳随机森林模型创建一个SHAP解释器
    shap_values = explainer.shap_values(X_test)  # SHAP值
    # 将重要性按正确的顺序存入DataFrame
    importances_df[cog] = np.abs(shap_values).mean(axis=0)
    print("重要性", importances_df[cog])

    for env_var in env_vars_labels:  # 始一个内层循环，遍历每一个环境变量
        corr, _ = spearmanr(X[env_var], Y[cog])  # 计算当前变量与目标之间的斯皮尔曼相关系数
        correlations_df.loc[env_var, cog] = corr  # 保存相关系数
        print("相关系数", correlations_df[cog])

# =========================================================================================
# ======================================5.保存分析结果========================================
# =========================================================================================
# 分析结果保存路径
output_excel_path = str(OUTPUT_DIR / "analysis_results.xlsx")
# 保存
with pd.ExcelWriter(output_excel_path, engine="xlsxwriter") as writer:
    variation_explained.to_excel(writer, sheet_name="Variation Explained (R2)", header=["R2_Score"])
    importances_df.to_excel(writer, sheet_name="SHAP Feature Importance")
    correlations_df.to_excel(writer, sheet_name="Spearman Correlations")


# =========================================================================================
# ======================================6.绘图函数=======================================
# =========================================================================================
def plot_correlation_heatmap(
    variation_explained, correlations_df, importances_df, selected_cmap, bar_color="skyblue"
):

    plot_data = correlations_df.reset_index().melt(
        id_vars="index", var_name="COG", value_name="Correlation"
    )  # 格式转换，方便绘图
    plot_data.rename(columns={"index": "Variable"}, inplace=True)  # 重命名列
    plot_data["Importance"] = importances_df.reset_index().melt(
        id_vars="index", value_name="Importance"
    )["Importance"]  # 转换

    min_imp = plot_data["Importance"].min()  # 重要性的最小值
    max_imp = plot_data["Importance"].max()  # 重要性的最大值
    print("最小值", min_imp)
    print("最大值", max_imp)
    # 对重要性进行归一化处理，用于映射气泡大小。如果最大最小值相等，则设为0.5
    plot_data["Normalized_Importance"] = (
        (plot_data["Importance"] - min_imp) / (max_imp - min_imp) if max_imp > min_imp else 0.5
    )
    # 创建一个包含两个子图的画布
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 18),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 8]},
        constrained_layout=True,
    )

    axes[0].bar(
        variation_explained.index,
        variation_explained.values * 100,
        color=bar_color,
        edgecolor="none",
    )  # 在上面的子图绘制R2的柱状图
    axes[0].set_ylabel("Variation explained (%)", fontsize=20)  # 第一个子图的Y轴标题
    axes[0].set_ylim(0, 100)  # 第一个子图的Y轴范围
    axes[0].tick_params(axis="y", labelsize=16)  # 设置第一个子图Y轴刻度字体
    # 在柱状图上添加数值标注
    for index, value in enumerate(variation_explained.values):
        # 将R2值乘以100保留两位小数
        label = f"{value * 100:.2f}"
        # 添加数值标注
        axes[0].text(
            index, value * 100 + 1, label, ha="center", va="bottom", fontsize=10, color="black"
        )

    # 调整Y轴的上限
    axes[0].set_ylim(0, 110)

    MIN_BUBBLE_SIZE = 20  # 设置最小的气泡尺寸，确保最小重要性也可见
    MAX_BUBBLE_SIZE = 400  # 设置最大的气泡尺寸
    # 计算气泡大小
    sizes = MIN_BUBBLE_SIZE + plot_data["Normalized_Importance"] * (
        MAX_BUBBLE_SIZE - MIN_BUBBLE_SIZE
    )
    # 绘制下面的气泡图
    scatter = axes[1].scatter(
        x=plot_data["COG"],  # x轴为目标
        y=plot_data["Variable"],  # y轴为特征
        s=sizes,  # 大小由重要性决定
        c=plot_data["Correlation"],  # 气泡的颜色由相关系数决定
        cmap=selected_cmap,  # 颜色映射
        norm=mcolors.Normalize(vmin=-1, vmax=1),  # 颜色映射的范围
        edgecolor="grey",  # 边框颜色
        linewidth=0,  # 边框宽度
        zorder=3,
    )
    axes[1].grid(
        True, which="both", linestyle="--", linewidth=0.5, color="lightgrey", zorder=0
    )  # 添加气泡图背景网格线

    axes[1].set_xticks(np.arange(len(cog_cats_labels)))  # 设置X轴的刻度位置
    axes[1].set_xticklabels(cog_cats_labels, rotation=90)  # 设置X轴的刻度标注
    axes[1].set_yticks(np.arange(len(env_vars_labels)))  # 设置Y轴的刻度位置
    axes[1].set_yticklabels(env_vars_labels)  # Y轴的刻度标签
    axes[1].set_ylim(-1, len(env_vars_labels) + 0.5)  # Y轴的范围
    axes[1].tick_params(axis="both", which="major", labelsize=20)  # 气泡图刻度的字体设置

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=axes, location="right", shrink=0.6)
    cbar.set_label("Correlation (%)", fontsize=20)  # 颜色条的标题
    cbar.set_ticks([-0.8, -0.4, 0, 0.4, 0.8])  # 自定义颜色条上的刻度位置
    cbar.set_ticklabels(["-80", "-40", "0", "40", "80"])  # 自定义颜色条上的刻度标签
    # 设置颜色条刻度标签的字体
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(20)

    legend_handles = []  # 用于存放气泡图图例的元素

    # 根据真实的重要性范围，自动生成图例的标签
    legend_importance_values = np.linspace(min_imp, max_imp, 4)
    # 循环创建图例元素
    for value in legend_importance_values:
        display_size = MIN_BUBBLE_SIZE + value * (
            MAX_BUBBLE_SIZE - MIN_BUBBLE_SIZE
        )  # 该值在图例中对应的气泡显示大小
        # 创建一个用于图例的散点句柄：坐标为空使其不在图上显示，大小s，颜色c
        legend_handles.append(plt.scatter([], [], s=display_size, c="black", label=f"{value:.2f}"))
    # 添加气泡的图例
    legend = fig.legend(
        handles=legend_handles,  # 元素
        title="Variable\nimportance",  # 标题
        loc="center right",  # 位置
        bbox_to_anchor=(1.01, 0.18),  # 精确位置
        frameon=True,  # 边框
        edgecolor="black",  # 边框颜色
        labelspacing=2.5,  # 行间距
        borderpad=1.2,  # 内边距
    )
    plt.setp(legend.get_title(), fontsize="20")  # 设置图例标题

    plt.savefig(str(OUTPUT_DIR / f"heatmap_{selected_scheme}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"heatmap_{selected_scheme}.pdf"), bbox_inches="tight")


# =========================================================================================
# ======================================7.绘图========================================
# =========================================================================================
# 从统一的颜色库中获取选定的配色方案
cmap_name, bar_color = color_schemes[selected_scheme]

# 根据名称获取颜色映射对象
selected_palette = plt.get_cmap(cmap_name)

# 调用绘图函数
plot_correlation_heatmap(
    variation_explained, correlations_df, importances_df, selected_palette, bar_color=bar_color
)  # 将选定的颜色传递给绘图函数

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

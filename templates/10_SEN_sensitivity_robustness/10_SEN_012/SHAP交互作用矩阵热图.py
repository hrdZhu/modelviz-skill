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
# ====================================== 1. 环境设置 =======================================
# =========================================================================================
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ======================================2.颜色库=======================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"],
    2: ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
    3: ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
    4: ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
    5: ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
    6: ["#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04"],
    7: ["#FFFFE0", "#FFFACD", "#FAFAD2", "#FFEFD5", "#FFE4B5", "#FFDAB9", "#EEE8AA", "#BDB76B"],
    8: ["#E0F7FA", "#B2EBF2", "#80DEEA", "#4DD0E1", "#26C6DA", "#00BCD4", "#0097A7", "#006064"],
    9: ["#F1F8E9", "#DCEDC8", "#C5E1A5", "#AED581", "#9CCC65", "#8BC34A", "#689F38", "#33691E"],
    10: ["#ECEFF1", "#CFD8DC", "#B0BEC5", "#90A4AE", "#78909C", "#607D8B", "#546E7A", "#455A64"],
    11: ["#FFEB3B", "#FBC02D", "#FFA000", "#F57C00", "#E64A19", "#D84315", "#BF360C", "#870000"],
    12: ["#CCFF90", "#76FF03", "#64DD17", "#00C853", "#00BFA5", "#00B8D4", "#0091EA", "#2962FF"],
    13: ["#F8BBD0", "#F48FB1", "#F06292", "#EC407A", "#E91E63", "#C2185B", "#880E4F", "#4A0025"],
    14: ["#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC", "#9C27B0", "#7B1FA2", "#4A148C", "#311B92"],
    15: ["#F9FBE7", "#F0F4C3", "#E6EE9C", "#DCE775", "#D4E157", "#CDDC39", "#AFB42B", "#827717"],
    16: ["#EFEBE9", "#D7CCC8", "#BCAAA4", "#A1887F", "#8D6E63", "#795548", "#6D4C41", "#4E342E"],
    17: ["#F3E5F5", "#E1BEE7", "#CE93D8", "#BA68C8", "#AB47BC", "#9C27B0", "#7B1FA2", "#4A148C"],
    18: ["#E0F2F1", "#B2DFDB", "#80CBC4", "#4DB6AC", "#26A69A", "#009688", "#00796B", "#004D40"],
    19: ["#212121", "#424242", "#616161", "#757575", "#9E9E9E", "#BDBDBD", "#E0E0E0", "#F5F5F5"][
        ::-1
    ],
    20: ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#b10026"],
}
scheme_index = 1  # 定义颜色方案
selected_colors = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])  # 获取颜色
# =========================================================================================
# ======================================3.形状标记库=======================================
# =========================================================================================
STYLE_SCHEMES = {
    1: "o",
    2: "s",
    3: r"$\spadesuit$",
    4: r"$\clubsuit$",
    5: "*",
    6: "X",
    7: r"$\heartsuit$",
    8: r"$\infty$",
    9: r"$\odot$",
    10: r"$\oplus$",
    11: r"$\otimes$",
}
style_index = 11  # 定义标记形状
selected_marker = STYLE_SCHEMES.get(style_index, "o")  # 获取
# =========================================================================================
# ======================================4.数据加载=======================================
# =========================================================================================
# 原始数据路径
csv_path = str(DATA_DIR / "simulation_data.csv")
df = pd.read_csv(csv_path)  # 读取
y = df.iloc[:, -1]  # 目标变量
X = df.iloc[:, :-1]  # 特征变量
feature_names = X.columns.tolist()  # 获取特征变量的列名
# =========================================================================================
# ======================================5.数据划分及模型构建=======================================
# =========================================================================================
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_base = xgb.XGBRegressor(
    objective="reg:squarederror", random_state=42, n_jobs=-1
)  # 初始化XGBoost
# 超参数范围
param_grid = {
    "n_estimators": [100, 200, 300],
    # 'max_depth': [3, 5, 7],
}
# 初始化网格搜索对象
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)  # 执行网格搜索训练
xgb_model = grid_search.best_estimator_  # 获取最佳模型
print(f"最佳参数: {grid_search.best_params_}")
# =========================================================================================
# ======================================6.SHAP分析=======================================
# =========================================================================================
explainer = shap.TreeExplainer(xgb_model)  # 创建基于树模型的SHAP解释器
shap_interaction_values = explainer.shap_interaction_values(X_test)  # 测试集样本的SHAP交互值
mean_signed_interaction_matrix = shap_interaction_values.mean(0)  # 交互值的均值
mean_abs_interaction_matrix = np.abs(
    mean_signed_interaction_matrix
)  # 均值的绝对值，用于绘图颜色深浅


# =========================================================================================
# ======================================7.蜂群图辅助函数=======================================
# =========================================================================================
def simple_beeswarm(x_values, nbins=40, width=0.1):
    hist_range = (np.min(x_values), np.max(x_values))  # 数据的最小值和最大值范围
    if hist_range[0] == hist_range[1]:  # 如果最大值等于最小值
        hist_range = (hist_range[0] - 0.1, hist_range[1] + 0.1)  # 手动扩展范围
    counts, edges = np.histogram(
        x_values, bins=nbins, range=hist_range
    )  # 计算直方图，获取各区间的计数和边界
    bin_indices = np.digitize(x_values, edges) - 1  # 计算每个数据点所属的箱子索引
    bin_indices = np.clip(bin_indices, 0, nbins - 1)  # 索引范围
    y_values = np.zeros_like(x_values)  # 初始化Y轴抖动值
    max_count = counts.max()  # 获取直方图中的最大计数值
    if max_count == 0:  # 如果最大计数为0
        return np.random.uniform(-0.1, 0.1, len(x_values))  # 返回随机噪声
    for i in range(len(counts)):  # 遍历每一个直方图箱子
        idxs = np.where(bin_indices == i)[0]  # 找到当前箱子内的所有数据点索引
        if len(idxs) == 0:  # 如果当前箱子为空
            continue
        current_width = (counts[i] / max_count) * width  # 根据当前箱子的密度计算抖动宽度
        ys = np.linspace(-current_width, current_width, len(idxs))  # 在宽度范围内生成均匀分布的Y值
        np.random.shuffle(ys)  # 打乱Y值顺序
        y_values[idxs] = ys  # 将计算好的Y值赋给对应的数据点
    return y_values  # 返回计算好的Y轴抖动坐标


# =========================================================================================
# ======================================8.绘图函数=======================================
# =========================================================================================
def plot_shap_interaction_matrix(
    shap_interaction_values,
    mean_abs_interaction_matrix,
    mean_signed_interaction_matrix,
    X_test,
    feature_names,
    custom_colors,
    marker_symbol,
):
    labels = feature_names  # 获取特征名称作为标签
    n = len(labels)  # 特征数量
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", custom_colors
    )  # 创建线性分段颜色映射
    max_off_diag = np.max(
        mean_abs_interaction_matrix[~np.eye(n, dtype=bool)]
    )  # 找出非对角线元素中的绝对值最大值
    # 防止全零矩阵导致除零错误
    if max_off_diag == 0:
        max_off_diag = 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=max_off_diag)  # 设置颜色的归一化范围
    col_xlims = []  # 初始化每列X轴范围列表
    col_xticks = []  # 初始化每列X轴刻度列表

    for j in range(n):  # 遍历每一列特征
        all_col_values = []  # 初始化当前列的所有SHAP值列表
        for i in range(j, n):  # 遍历下三角区域
            vals = shap_interaction_values[:, i, j]  # 获取对应特征对的SHAP交互值
            all_col_values.extend(vals)  # 添加到列表中

        if len(all_col_values) > 0:  # 如果有数据
            max_val = np.max(np.abs(all_col_values))  # 计算该列绝对值的最大值
            limit = max(max_val * 1.1, 0.01)  # 设定显示范围限制，保留一定边距
            col_xlims.append((-limit, limit))  # 添加到X轴范围列表
            # 动态步长
            if limit <= 0.05:
                step = 0.02  # 步长
            elif limit <= 0.2:
                step = 0.1
            elif limit <= 0.8:
                step = 0.2
            elif limit <= 2.0:
                step = 0.5
            elif limit <= 5.0:
                step = 1.0
            else:
                step = 2.0

            ticks = np.arange(0, limit + 0.0001, step)  # 正半轴刻度
            ticks = np.unique(np.concatenate((-ticks, ticks)))  # 生成对称刻度并去重
            ticks = ticks[(ticks >= -limit) & (ticks <= limit)]  # 过滤超出范围的刻度
            if len(ticks) <= 1:
                ticks = np.array([-limit, 0, limit])  # 如果刻度太少，强制设定起止点
            col_xticks.append(ticks)  # 添加到刻度列表
        else:  # 如果没有数据
            col_xlims.append((-1, 1))  # 设置默认范围
            col_xticks.append([-1, 0, 1])  # 设置默认刻度

    # 创建画布
    fig = plt.figure(figsize=(14, 14))

    gs = gridspec.GridSpec(n, n, figure=fig, wspace=0.08, hspace=0.08)  # 定义网格布局，设置子图间距

    for i in range(n):  # 遍历行
        for j in range(n):  # 遍历列
            ax = fig.add_subplot(gs[i, j])  # 在网格位置添加子图
            ax.set_box_aspect(1)  # 强制子图比例为正方形
            # 上三角
            if i < j:
                # 获取绝对值用于颜色
                abs_val = mean_abs_interaction_matrix[i, j]
                # 获取有符号真实值用于文本显示
                signed_val = mean_signed_interaction_matrix[i, j]
                ax.set_facecolor("#f9f9f9")  # 背景色
                ax.set_xlim(0, 1)  # X轴范围
                ax.set_ylim(0, 1)  # Y轴范围
                # 绘制热图
                ax.scatter(
                    0.5, 0.5, s=9000, c=[abs_val], cmap=cmap, norm=norm, marker=marker_symbol
                )

                for spine in ax.spines.values():  # 遍历边框
                    spine.set_edgecolor("black")  # 设置边框颜色
                    spine.set_linewidth(1)  # 设置边框线宽

                text_color = "black"  # 文本颜色
                # 添加数值标注
                ax.text(
                    0.5,
                    0.5,
                    f"{signed_val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    color=text_color,
                    fontweight="bold",
                )
                ax.set_xticks([])  # 隐藏X轴刻度
                ax.set_yticks([])  # 隐藏Y轴刻度

            # 下三角
            else:
                ax.set_facecolor("#f2f2f2")  # 背景色
                xlim = col_xlims[j]  # X轴范围
                xticks = col_xticks[j]  # X轴刻度
                for x_loc in xticks:  # 遍历刻度位置
                    # 绘制白色垂直参考线
                    ax.axvline(x_loc, color="white", linestyle="-", linewidth=1.5, zorder=0)
                # 绘制X=0处的灰色参考线
                ax.axvline(0, color="#888888", linestyle="-", linewidth=1.0, zorder=1)

                x_vals = shap_interaction_values[:, i, j]  # 获取该对特征的交互值数据
                y_vals = simple_beeswarm(x_vals, nbins=35, width=0.15)  # 计算蜂群图的Y轴抖动

                feature_values = X_test.iloc[:, j].values  # 获取对应的原始特征值
                c_norm = plt.Normalize(
                    vmin=np.min(feature_values), vmax=np.max(feature_values)
                )  # 对原始特征值进行归一化用于颜色映射
                # 绘制散点图
                ax.scatter(
                    x_vals,
                    y_vals,
                    c=feature_values,
                    cmap=cmap,
                    norm=c_norm,
                    s=25,
                    alpha=0.9,
                    edgecolors="none",
                    zorder=2,
                    marker=marker_symbol,
                )

                ax.set_yticks([])  # 隐藏Y轴刻度
                ax.set_ylim(-0.5, 0.5)  # Y轴范围
                ax.set_xlim(xlim)  # X轴范围
                for spine in ax.spines.values():  # 遍历边框
                    spine.set_edgecolor("black")  # 边框颜色
                    spine.set_linewidth(1)  # 边框线宽
                if i == n - 1:  # 如果是最后一行
                    ax.set_xticks(xticks)  # 设置X轴刻度
                    ax.tick_params(
                        axis="x", direction="out", length=3, width=1, colors="black", labelsize=18
                    )  # 设置刻度样式
                    plt.setp(ax.get_xticklabels(), ha="center")  # 设置刻度标签对齐方式

                    if len(xticks) > 5:  # 如果刻度过多
                        for label in ax.xaxis.get_ticklabels()[::2]:  # 每隔一个隐藏标签
                            label.set_visible(False)  # 隐藏标签
                else:  # 如果不是最后一行
                    ax.set_xticks([])  # 隐藏X轴刻度
            if i == 0:  # 如果是第一行
                ax.set_title(labels[j], fontsize=18, pad=8, color="#333333")  # 设置列标题
            if j == 0:  # 如果是第一列
                ax.set_ylabel(labels[i], fontsize=18, labelpad=8, color="#333333")  # 设置行标题

    fig.text(0.5, 0.05, "SHAP interaction value", ha="center", va="center", fontsize=18)  # X轴标题

    cbar_ax = fig.add_axes([0.92, 0.28, 0.02, 0.45])  # 定义颜色条的位置和大小
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))  # 创建用于生成颜色条的映射对象
    sm.set_array([])  # 设置空数组
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")  # 绘制垂直颜色条
    cbar.set_label("Raw feature value / |Interaction|", fontsize=18, labelpad=10)  # 颜色条标题
    cbar.set_ticks([])  # 去掉颜色条的刻度数值
    cbar.ax.text(1.5, 0, "Low", ha="left", va="center", fontsize=18)  # 颜色条底部标记
    cbar.ax.text(1.5, 1, "High", ha="left", va="center", fontsize=18)  # 颜色条顶部标记
    plt.subplots_adjust(left=0.08, right=0.9, top=0.92, bottom=0.1)  # 调整子图布局
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"shap_int_{scheme_index}_{style_index}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        str(OUTPUT_DIR / f"shap_int_{scheme_index}_{style_index}.png"),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    # 调用绘图函数
    plot_shap_interaction_matrix(
        shap_interaction_values,  # SHAP交互值矩阵
        mean_abs_interaction_matrix,  # 绝对值
        mean_signed_interaction_matrix,  # 交互值均值
        X_test,  # 测试集特征数据
        feature_names,  # 特征名称列表
        custom_colors=selected_colors,  # 选定的颜色方案
        marker_symbol=selected_marker,  # 选定的标记形状
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

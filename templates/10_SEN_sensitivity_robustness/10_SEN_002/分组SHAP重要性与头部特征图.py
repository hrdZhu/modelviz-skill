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
# ===== 1. 库导入 ==============================================================
# =============================================================================
import lightgbm as lgb
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Wedge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =============================================================================
# ===== 2. 颜色库与数据加载 =====================================================
# =============================================================================
color_schemes = {
    1: {
        "cmap": ["#EBF5FB", "#85C1E9", "#2874A6"],
        "category_colors": {
            "Adaptive capacity": "#AED6F1",
            "Socioeconomic": "#85C1E9",
            "Spatial structure": "#5DADE2",
        },
    },
    2: {
        "cmap": ["#F5EFE6", "#AF8F6F", "#74512D"],
        "category_colors": {
            "Adaptive capacity": "#E3D5CA",
            "Socioeconomic": "#D7C0AE",
            "Spatial structure": "#AF8F6F",
        },
    },
    3: {
        "cmap": ["#E8F5E9", "#81C784", "#2E7D32"],
        "category_colors": {
            "Adaptive capacity": "#A5D6A7",
            "Socioeconomic": "#81C784",
            "Spatial structure": "#66BB6A",
        },
    },
    4: {
        "cmap": ["#F3E5F5", "#BA68C8", "#6A1B9A"],
        "category_colors": {
            "Adaptive capacity": "#CE93D8",
            "Socioeconomic": "#BA68C8",
            "Spatial structure": "#AB47BC",
        },
    },
    5: {
        "cmap": ["#FFEBEE", "#E57373", "#C62828"],
        "category_colors": {
            "Adaptive capacity": "#EF9A9A",
            "Socioeconomic": "#E57373",
            "Spatial structure": "#EF5350",
        },
    },
    6: {
        "cmap": ["#F5F5F5", "#9E9E9E", "#212121"],
        "category_colors": {
            "Adaptive capacity": "#E0E0E0",
            "Socioeconomic": "#BDBDBD",
            "Spatial structure": "#9E9E9E",
        },
    },
    7: {
        "cmap": ["#FFF3E0", "#FFB74D", "#E65100"],
        "category_colors": {
            "Adaptive capacity": "#FFCC80",
            "Socioeconomic": "#FFB74D",
            "Spatial structure": "#FFA726",
        },
    },
    8: {
        "cmap": ["#E0F7FA", "#4DD0E1", "#006064"],
        "category_colors": {
            "Adaptive capacity": "#80DEEA",
            "Socioeconomic": "#4DD0E1",
            "Spatial structure": "#26C6DA",
        },
    },
    9: {
        "cmap": ["#FCE4EC", "#F06292", "#C2185B"],
        "category_colors": {
            "Adaptive capacity": "#F48FB1",
            "Socioeconomic": "#F06292",
            "Spatial structure": "#EC407A",
        },
    },
    10: {
        "cmap": ["#E0F2F1", "#4DB6AC", "#004D40"],
        "category_colors": {
            "Adaptive capacity": "#80CBC4",
            "Socioeconomic": "#4DB6AC",
            "Spatial structure": "#26A69A",
        },
    },
    11: {
        "cmap": ["#EDE7F6", "#9575CD", "#4527A0"],
        "category_colors": {
            "Adaptive capacity": "#B39DDB",
            "Socioeconomic": "#9575CD",
            "Spatial structure": "#7E57C2",
        },
    },
    12: {
        "cmap": ["#FFFDE7", "#FFF176", "#F57F17"],
        "category_colors": {
            "Adaptive capacity": "#FFF59D",
            "Socioeconomic": "#FFF176",
            "Spatial structure": "#FFEE58",
        },
    },
    13: {
        "cmap": ["#F1F8E9", "#AED581", "#558B2F"],
        "category_colors": {
            "Adaptive capacity": "#C5E1A5",
            "Socioeconomic": "#AED581",
            "Spatial structure": "#9CCC65",
        },
    },
    14: {
        "cmap": ["#E3F2FD", "#64B5F6", "#1565C0"],
        "category_colors": {
            "Adaptive capacity": "#90CAF9",
            "Socioeconomic": "#64B5F6",
            "Spatial structure": "#42A5F5",
        },
    },
    15: {
        "cmap": ["#FBE9E7", "#FF8A65", "#D84315"],
        "category_colors": {
            "Adaptive capacity": "#FFAB91",
            "Socioeconomic": "#FF8A65",
            "Spatial structure": "#FF7043",
        },
    },
    16: {
        "cmap": ["#ECEFF1", "#90A4AE", "#37474F"],
        "category_colors": {
            "Adaptive capacity": "#B0BEC5",
            "Socioeconomic": "#90A4AE",
            "Spatial structure": "#78909C",
        },
    },
    17: {
        "cmap": ["#FFF8E1", "#FFD54F", "#FF8F00"],
        "category_colors": {
            "Adaptive capacity": "#FFE082",
            "Socioeconomic": "#FFD54F",
            "Spatial structure": "#FFCA28",
        },
    },
    18: {
        "cmap": ["#EFEBE9", "#A1887F", "#4E342E"],
        "category_colors": {
            "Adaptive capacity": "#BCAAA4",
            "Socioeconomic": "#A1887F",
            "Spatial structure": "#8D6E63",
        },
    },
    19: {
        "cmap": ["#F9FBE7", "#DCE775", "#9E9D24"],
        "category_colors": {
            "Adaptive capacity": "#E6EE9C",
            "Socioeconomic": "#DCE775",
            "Spatial structure": "#D4E157",
        },
    },
    20: {
        "cmap": ["#FCE4EC", "#F06292", "#AD1457"],
        "category_colors": {
            "Adaptive capacity": "#F48FB1",
            "Socioeconomic": "#F06292",
            "Spatial structure": "#EC407A",
        },
    },
    21: {
        "cmap": ["#FCE4EC", "#EF9A9A", "#E57373", "#D32F2F", "#C62828"],
        "category_colors": {
            "Adaptive capacity": "#EF9A9A",
            "Socioeconomic": "#E57373",
            "Spatial structure": "#C62828",
        },
    },
}
selected_scheme = 12  # 选择配色方案
selected_colors = color_schemes[selected_scheme]  # 从颜色库中提取对应的颜色方案

file_path = str(DATA_DIR / "simulated_data.xlsx")  # 定义数据文件的路径
local_data = pd.read_excel(file_path, engine="openpyxl")  # 读取数据
X = local_data.iloc[:, :-1]  # 提取特征X
y = local_data.iloc[:, -1]  # 提取目标变量y
# =============================================================================
# ===== 4.数据预处理=============================================
# =============================================================================
# 获取特征名称列表
features = list(X.columns)
# 划分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# 将标准化后的数据转换回Pandas DataFrame，并保留列名
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features)
# =============================================================================
# =====4.模型构建=============================================
# =============================================================================
# 初始化LightGBM回归模型
lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
# 定义一个参数网格，用于超参数搜索
param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
# 设置网格搜索，5折交叉验证
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, n_jobs=-1, scoring="r2")
# 在标准化的训练集上执行网格搜索，寻找最佳参数
grid_search.fit(X_train_scaled_df, y_train)
# 获取最佳模型
best_model = grid_search.best_estimator_
# =============================================================================
# =====5.shap分析及重要性计算=============================================
# =============================================================================
# 创建一个TreeExplainer对象，用于计算SHAP值
explainer = shap.TreeExplainer(best_model)
# 计算验证集上每个样本的SHAP值
shap_values = explainer.shap_values(X_val_scaled_df)
# 计算每个特征SHAP值的绝对值的平均值
mean_abs_shap = np.abs(shap_values).mean(axis=0)
# 将特征重要性保存，并按降序排列
feature_importances = pd.Series(mean_abs_shap, index=features).sort_values(ascending=False)
# 对不同的特征进行分类
category_map = {
    "Adaptive capacity": ["DCA", "DCS", "DH", "DMS", "DES"],
    "Spatial structure": ["ISR", "ABS", "AGS", "BD"],
    "Socioeconomic": ["NL", "HP", "BA"],
}
# 计算每个类别的总特征重要性
category_totals = {cat: feature_importances[feats].sum() for cat, feats in category_map.items()}
# 计算所有特征重要性的总和
total_importance = sum(category_totals.values())
# 计算每个类别重要性占总重要性的百分比
category_percentages = {cat: (val / total_importance) * 100 for cat, val in category_totals.items()}


# =============================================================================
# ============================6.绘图函数==================================
# =============================================================================
def create_and_save_plot(
    feature_importances, shap_values, X_val, category_percentages, total_importance, colors_dict
):
    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), gridspec_kw={"width_ratios": [1.1, 1]})

    sorted_features_list = feature_importances.index.tolist()  # 按重要性排序的特征名列表
    shap_values_df = pd.DataFrame(shap_values, columns=features)  # 将SHAP值数组转为DataFrame
    shap_values_sorted = shap_values_df[
        sorted_features_list
    ].values  # 按特征重要性顺序重新排列SHAP值
    X_sorted = X_val[sorted_features_list]  # 按特征重要性顺序重新排列验证集特征值

    # 获取颜色
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", colors_dict["cmap"])

    plt.sca(ax1)  # 设置当前绘图区域为第一个子图，用于绘制蜂巢图
    # 绘制蜂巢图
    shap.summary_plot(
        shap_values_sorted,
        X_sorted,
        plot_type="dot",
        show=False,
        color_bar=False,
        plot_size=None,
        cmap=custom_cmap,
    )

    norm = mcolors.Normalize(
        vmin=X_sorted.values.min(), vmax=X_sorted.values.max()
    )  # 创建一个归一化对象，范围为特征值的最小和最大值
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)  # 创建一个可映射对象，用于设置颜色条
    sm.set_array([])  # 设置一个空数组，这是创建颜色条所必需的
    # 添加颜色条
    cbar = fig.colorbar(sm, ax=ax1, aspect=40, pad=0.02)
    cbar.set_label("Feature value", rotation=270, labelpad=15)  # 颜色条标题
    cbar.set_ticks([norm.vmin, norm.vmax])  # 颜色条的刻度位置
    cbar.set_ticklabels(["Low", "High"])  # 刻度标签
    cbar.outline.set_visible(False)  # 去掉颜色条的边框

    ax1.set_xlabel("SHAP value (impact on model output)", fontsize=12)  # 蜂巢图x周标题
    ax1.tick_params(axis="y", labelsize=11)  # 蜂巢图y周标题
    for spine in ax1.spines.values():  # 设置图框
        spine.set_visible(True)
    # 添加文本标注
    ax1.text(0.05, 0.15, "SHE", fontsize=14, fontweight="bold", transform=ax1.transAxes)
    ax1.text(0.95, 0.02, "(a2)", fontsize=12, ha="center", transform=ax1.transAxes)

    # 右侧子图，绘制特征重要性图
    sorted_features_bar = feature_importances.index  # 获取按重要性排序的特征名
    y_pos_bar = np.arange(len(sorted_features_bar))  # 创建y轴的位置
    category_color_map = colors_dict["category_colors"]  # 获取类别颜色
    bar_color_map = {}  # 用于存储每个特征的颜色
    for category, feats in category_map.items():  # 遍历特征类别映射
        color = category_color_map.get(category, "#9E9E9E")  # 获取该类别的颜色
        for feature in feats:  # 遍历该类别下的所有特征
            bar_color_map[feature] = color  # 将特征和其对应的颜色存入字典

    bar_colors = [
        bar_color_map.get(feature, "#9E9E9E") for feature in sorted_features_bar
    ]  # 根据排序后的特征列表生成颜色列表
    # 绘制特征重要性水平条形图
    ax2.barh(y_pos_bar, feature_importances, color=bar_colors, align="center")
    ax2.set_yticks(y_pos_bar)  # 设置y轴的刻度位置
    ax2.set_yticklabels(sorted_features_bar, fontsize=11)  # 设置y轴的标注，就是特征名
    ax2.invert_yaxis()  # 反转y轴，使最重要的特征显示在顶部
    ax2.set_xlabel(
        "mean(|SHAP value|) (average impact on model output magnitude)", fontsize=12
    )  # x轴标题
    ax2.tick_params(axis="y", length=0)  # 去掉y轴的刻度线
    # 设置图框
    for spine in ax2.spines.values():
        spine.set_visible(True)

    for i, v in enumerate(feature_importances):  # 遍历特征重要性值及其索引
        ax2.text(
            v + (ax2.get_xlim()[1] * 0.006), i, f"{v:.2f}", color="grey", va="center"
        )  # 在条形图的右侧加上平均绝对数值

    # 添加一个水平分割线，先上面的特征占比超过10%
    feature_percentages = (
        feature_importances / total_importance
    ) * 100  # 计算每个特征的重要性百分比
    num_high_importance_features = (feature_percentages > 10).sum()  # 计算重要性大于10%的特征数量

    if (
        0 < num_high_importance_features < len(feature_importances)
    ):  # 如果存在重要性大于10%的特征，且不是全部特征
        line_pos_y = num_high_importance_features - 0.5  # 计算分割线的y轴位置
        ax2.axhline(y=line_pos_y, color="black", linestyle="--", linewidth=1)  # 绘制水平虚线
        # 在线上方添加文本
        ax2.text(
            x=ax2.get_xlim()[1] * 0.73,
            y=line_pos_y - 0.8,
            s="Features with FI > 10%",
            ha="center",
            va="top",
            fontsize=10,
        )
        # 绘制一个向上的箭头
        ax2.arrow(
            x=ax2.get_xlim()[1] * 0.6,  # 箭头的起始x坐标
            y=line_pos_y - 0.2,  # 箭头的起始y坐标
            dx=0,  # 箭头在x方向上的长度变化
            dy=-0.5,  # 箭头在y方向上的长度变化
            head_width=0.01,  # 箭头头部的宽度
            head_length=0.15,  # 箭头头部的长度
            fc="black",  # 箭头的填充颜色
            ec="black",
        )  # 箭头的边框颜色
        # 在线下方添加文本
        ax2.text(
            x=ax2.get_xlim()[1] * 0.73,
            y=line_pos_y + 0.8,
            s="Features with FI < 10%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        # 绘制一个向下的箭头
        ax2.arrow(
            x=ax2.get_xlim()[1] * 0.6,
            y=line_pos_y + 0.2,
            dx=0,
            dy=0.5,
            head_width=0.01,
            head_length=0.15,
            fc="black",
            ec="black",
        )

    # 绘制表示不同特征占比的环形图
    ax_donut = ax2.inset_axes([0.35, 0.1, 0.5, 0.4])  # 创建一个新的坐标轴用于绘制环形图
    donut_color_map = colors_dict["category_colors"]  # 获取类别颜色映射
    cat_order = sorted(
        category_percentages, key=category_percentages.get, reverse=True
    )  # 按百分比降序排列类别
    donut_colors = [donut_color_map[cat] for cat in cat_order]  # 根据排序后的类别生成颜色列表
    sorted_donut_sizes = [
        category_percentages[cat] for cat in cat_order
    ]  # 根据排序后的类别生成百分比大小列表

    inner_radius, min_outer_radius, max_outer_radius = 0.2, 0.6, 1.0  # 定义内径、最小外径和最大外径
    percentages = pd.Series(category_percentages)
    min_perc, max_perc = percentages.min(), percentages.max()  # 获取百分比的最小值和最大值

    outer_radii_map = {}  # 存储每个类别的外径
    for cat, perc in category_percentages.items():  # 遍历每个类别及其百分比
        scale_factor = (
            (perc - min_perc) / (max_perc - min_perc) if (max_perc - min_perc) != 0 else 0.5
        )  # 计算缩放因子
        radius = (
            min_outer_radius + (max_outer_radius - min_outer_radius) * scale_factor
        )  # 根据百分比计算外径
        outer_radii_map[cat] = radius  # 存储计算出的外径
    outer_radii = [outer_radii_map[cat] for cat in cat_order]  # 根据排序后的类别生成外径列表

    start_angle = 90  # 设置第一个扇形的起始角度
    sizes_sum = sum(sorted_donut_sizes)  # 计算所有百分比的总和
    angles = [360 * size / sizes_sum for size in sorted_donut_sizes]  # 计算每个扇形对应的角度

    ax_donut.set_aspect("equal", adjustable="box")  # 设置横纵比
    for i in range(len(cat_order)):  # 遍历每个类别来绘制扇形
        category, outer_radius, color = (
            cat_order[i],
            outer_radii[i],
            donut_colors[i],
        )  # 获取类别、外径和颜色
        width, end_angle = (
            outer_radius - inner_radius,
            start_angle - angles[i],
        )  # 计算扇形的宽度和结束角度

        # 创建一个楔形/扇区对象
        wedge = Wedge(
            center=(0, 0),  # 中心点坐标
            r=outer_radius,  # 外径
            theta1=end_angle,  # 起始角度
            theta2=start_angle,  # 结束角度
            width=width,  # 扇区的宽度
            facecolor=color,  # 扇区的填充颜色
            edgecolor="white",  # 边框的颜色
            linewidth=1.5,
        )  # 边框的线宽
        ax_donut.add_patch(wedge)  # 添加环形图

        mid_angle_rad = np.deg2rad((start_angle + end_angle) / 2)  # 计算扇形中间角度的弧度值
        label_radius = inner_radius + (outer_radius - inner_radius) * 0.5  # 计算标签放置的半径位置
        text_x, text_y = (
            label_radius * np.cos(mid_angle_rad),
            label_radius * np.sin(mid_angle_rad),
        )  # 计算标签的x, y坐标
        label_text = f"{category}\n({category_percentages[category]:.2f}%)"  # 准备标签文本
        # 添加文本标签
        ax_donut.text(
            text_x,  # x坐标
            text_y,  # y坐标
            label_text,  # 文本内容
            ha="center",  # 水平对齐方式
            va="center",  # 垂直对齐方式
            fontsize=9,  # 字体大小
            linespacing=1.3,  # 行间距
            fontname="Times New Roman",  # 字体
            color="black",
        )  # 文本颜色
        start_angle = end_angle  # 更新下一个扇形的起始角度

    ax_donut.set_xlim(-1.4, 1.4)  # 环形图x轴范围
    ax_donut.set_ylim(-1.4, 1.4)  # 环形图y轴范围
    ax_donut.axis("off")
    ax2.text(0.95, 0.02, "(b2)", fontsize=12, ha="center", transform=ax2.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 保存组合图
    plt.savefig(str(OUTPUT_DIR / f"combined_{selected_scheme}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"combined_{selected_scheme}.pdf"), bbox_inches="tight")
    # 保存第一个子图
    # 隐藏第二个子图
    ax2.set_visible(False)
    plt.savefig(
        str(OUTPUT_DIR / f"subplot_summary_{selected_scheme}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(str(OUTPUT_DIR / f"subplot_summary_{selected_scheme}.pdf"), bbox_inches="tight")
    # 恢复显示第二个子图
    ax2.set_visible(True)
    # 保存第二个子图
    # 隐藏第一个子图和它的颜色条
    ax1.set_visible(False)
    cbar.ax.set_visible(False)
    plt.savefig(
        str(OUTPUT_DIR / f"subplot_importance_{selected_scheme}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(str(OUTPUT_DIR / f"subplot_importance_{selected_scheme}.pdf"), bbox_inches="tight")


# =============================================================================
# ===================================7.条用绘图函数绘图=============================================
# =============================================================================
# 调用绘图函数进行绘图
create_and_save_plot(
    feature_importances, shap_values, X_val, category_percentages, total_importance, selected_colors
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

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
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
COLOR_LIBRARY = {
    1: {
        "ring_colors": [
            "#8E44AD",
            "#2980B9",
            "#3498DB",
            "#27AE60",
            "#F1C40F",
            "#E67E22",
            "#E74C3C",
            "#1ABC9C",
            "#9B59B6",
            "#34495E",
        ],
        "bar_color": "#d3c0a3",
    },
    2: {
        "ring_colors": [
            "#f7fbff",
            "#f1f7ff",
            "#ebf3ff",
            "#eff3ff",
            "#bdd7e7",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#08519c",
            "#08306b",
        ],
        "bar_color": "#8c8c8c",
    },
    3: {
        "ring_colors": [
            "#f7fcf5",
            "#f1f9f0",
            "#ebf6e9",
            "#edf8e9",
            "#bae4b3",
            "#74c476",
            "#41ab5d",
            "#238b45",
            "#006d2c",
            "#00441b",
        ],
        "bar_color": "#a6761d",
    },
    4: {
        "ring_colors": [
            "#fff5eb",
            "#feeade",
            "#fedecb",
            "#feedde",
            "#fdd0a2",
            "#fdae6b",
            "#fd8d3c",
            "#f16913",
            "#d94801",
            "#8c2d04",
        ],
        "bar_color": "#7570b3",
    },
    5: {
        "ring_colors": [
            "#fbb4ae",
            "#b3cde3",
            "#ccebc5",
            "#decbe4",
            "#fed9a6",
            "#ffffcc",
            "#e5d8bd",
            "#fde0dd",
            "#d4e6f1",
            "#d1f2eb",
        ],
        "bar_color": "#bebada",
    },
    6: {
        "ring_colors": [
            "#e41a1c",
            "#377eb8",
            "#4daf4a",
            "#984ea3",
            "#ff7f00",
            "#ffff33",
            "#a65628",
            "#f781bf",
            "#999999",
            "#66c2a5",
        ],
        "bar_color": "#f781bf",
    },
    7: {
        "ring_colors": [
            "#67001f",
            "#b2182b",
            "#d6604d",
            "#f4a582",
            "#fddbc7",
            "#f7f7f7",
            "#d1e5f0",
            "#92c5de",
            "#4393c3",
            "#053061",
        ],
        "bar_color": "#878787",
    },
    8: {
        "ring_colors": [
            "#40004b",
            "#7b3294",
            "#c2a5cf",
            "#e7d4e8",
            "#f1eef6",
            "#f7f7f7",
            "#d9f0d3",
            "#a6dba0",
            "#5aae61",
            "#00441b",
        ],
        "bar_color": "#bababa",
    },
    9: {
        "ring_colors": [
            "#008080",
            "#54bebe",
            "#76c8c8",
            "#98d1d1",
            "#badbdb",
            "#f5f5f5",
            "#dedad2",
            "#e4bcad",
            "#df979e",
            "#c76274",
        ],
        "bar_color": "#c7c1a8",
    },
    10: {
        "ring_colors": [
            "#a50026",
            "#d73027",
            "#fc8d59",
            "#fee090",
            "#ffffbf",
            "#abdda4",
            "#e0f3f8",
            "#91bfdb",
            "#4575b4",
            "#313695",
        ],
        "bar_color": "#a1a1a1",
    },
    11: {
        "ring_colors": [
            "#ffffd4",
            "#f3e79b",
            "#fac484",
            "#f8a07e",
            "#eb7f86",
            "#ce6693",
            "#a75aa4",
            "#7454a2",
            "#4f3d81",
            "#3b2a63",
        ],
        "bar_color": "#66545e",
    },
    12: {
        "ring_colors": [
            "#002159",
            "#00429d",
            "#4771b2",
            "#73a2c6",
            "#a6d3d8",
            "#ffffe0",
            "#ffdfd5",
            "#ffbcaf",
            "#f4777f",
            "#d43d3d",
        ],
        "bar_color": "#b0a8b9",
    },
    13: {
        "ring_colors": [
            "#00441b",
            "#184524",
            "#2d5e3c",
            "#488055",
            "#66a36f",
            "#86c88b",
            "#a8eea0",
            "#cbf4b5",
            "#edfbd0",
            "#f7fcf5",
        ],
        "bar_color": "#705d3b",
    },
    14: {
        "ring_colors": [
            "#300054",
            "#4b0082",
            "#660099",
            "#8a2be2",
            "#9932cc",
            "#ba55d3",
            "#dda0dd",
            "#e6e6fa",
            "#f8f8ff",
            "#fdfdff",
        ],
        "bar_color": "#ffd700",
    },
    15: {
        "ring_colors": [
            "#a6761d",
            "#666666",
            "#d95f02",
            "#7570b3",
            "#e7298a",
            "#66a61e",
            "#e6ab02",
            "#1b9e77",
            "#e6f5c9",
            "#fdb462",
        ],
        "bar_color": "#8d8d8d",
    },
    16: {
        "ring_colors": [
            "#440154",
            "#482878",
            "#3e4a89",
            "#31688e",
            "#26828e",
            "#1f9e89",
            "#35b779",
            "#6dcd59",
            "#b4de2c",
            "#fde725",
        ],
        "bar_color": "#9e9e9e",
    },
    17: {
        "ring_colors": [
            "#0d0887",
            "#46039f",
            "#7201a8",
            "#9c179e",
            "#bd3786",
            "#d8576b",
            "#ed7953",
            "#fca636",
            "#fcce25",
            "#f0f921",
        ],
        "bar_color": "#fca636",
    },
    18: {
        "ring_colors": [
            "#000000",
            "#252525",
            "#525252",
            "#737373",
            "#969696",
            "#bdbdbd",
            "#d9d9d9",
            "#f0f0f0",
            "#f7f7f7",
            "#ffffff",
        ],
        "bar_color": "#f0f0f0",
    },
    19: {
        "ring_colors": [
            "#800000",
            "#9a6324",
            "#e6194B",
            "#f58231",
            "#ffe119",
            "#bfef45",
            "#3cb44b",
            "#4363d8",
            "#911eb4",
            "#46f0f0",
        ],
        "bar_color": "#a9a9a9",
    },
    20: {
        "ring_colors": [
            "#9e0142",
            "#d53e4f",
            "#f46d43",
            "#fdae61",
            "#fee08b",
            "#f7f7f7",
            "#e6f598",
            "#abdda4",
            "#66c2a5",
            "#3288bd",
        ],
        "bar_color": "#3288bd",
    },
}


# =========================================================================================
# ====================================== 4.绘图函数 =========================================
# =========================================================================================
def plot_multi_omics_circos(
    gene_names, heatmap_data, layer_labels, color_scheme_id=1
):  # 定义绘图函数，接收基因名、热图数据、图层标签和配色方案ID
    num_layers, num_genes = heatmap_data.shape  # 从热图数据获取图层数和特征数
    if len(layer_labels) != num_layers:  # 检查图层标签的数量是否与热图数据的图层数匹配
        raise ValueError(
            f"heatmap_data 的图层数 ({num_layers}) 与 layer_labels 的长度 ({len(layer_labels)}) 不匹配。"
        )
    # 创建一个16x16英寸的极坐标图窗和子图
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection="polar"))
    selected_scheme = COLOR_LIBRARY[color_scheme_id]  # 获取选定的配色方案字典
    ring_colors = selected_scheme["ring_colors"]  # 从字典中获取环形颜色列表
    bar_color = selected_scheme["bar_color"]  # 从字典中获取外部条形图的颜色
    # 确保颜色列表足够长
    if len(ring_colors) < num_layers:
        print(f"配色方案 {color_scheme_id} 颜色不足 ({len(ring_colors)})，将循环使用。")
        ring_colors = (ring_colors * (num_layers // len(ring_colors) + 1))[
            :num_layers
        ]  # 通过重复和切片来扩展颜色列表
    else:
        ring_colors = ring_colors[:num_layers]
    ring_thickness = 1.2  # 每个环的厚度
    radial_gap = 0.2  # 环之间的径向间隙
    central_hole_radius = 13.0  # 中心空白区域的半径
    step_with_gap = ring_thickness + radial_gap  # 计算每个图层（环+间隙）的总径向步进
    ring_positions = np.arange(
        central_hole_radius, central_hole_radius + num_layers * step_with_gap, step_with_gap
    )  # 计算每个环的起始径向位置

    num_features_in_gap = 3  # 在开口处留出相当于3个特征的空白位置
    total_positions = num_genes + num_features_in_gap  # 计算总共的角位置（特征+开口）
    angular_width_per_position = 2 * np.pi / total_positions  # 计算每个特征所占的角度
    gap_size_rad = num_features_in_gap * angular_width_per_position  # 计算总开口大小
    start_angle = np.pi / 2 + gap_size_rad / 2  # 计算绘图的起始角度，在顶部开口的右侧
    end_angle = start_angle + (2 * np.pi - gap_size_rad - 0.02)  # 计算绘图的结束角度
    theta = np.linspace(start_angle, end_angle, num_genes)  # 为每个特征生成等间距的角度位置
    width = angular_width_per_position * 0.8  # 设置每个特征条形的角宽度
    # ------------------------------------------------------------- 绘制层环状热图 -------------------------------------------------------------------
    for layer_idx in range(num_layers):  # 遍历每一层
        for gene_idx in range(num_genes):  # 遍历该层中的每一个特征
            alpha_value = (
                1.0 if heatmap_data[layer_idx, gene_idx] == 1 else 0.2
            )  # 根据热图数据设置透明度
            ax.bar(
                x=theta[gene_idx],  # 条形的角度位置
                height=ring_thickness,  # 条形的高度，即环的厚度
                width=width,  # 条形的角宽度
                bottom=ring_positions[layer_idx],  # 起始径向位置
                color=ring_colors[layer_idx],  # 颜色
                alpha=alpha_value,  # 透明度
                align="edge",  # 对齐方式
            )
    # -------------------------------------------------------------绘制最外层的径向堆叠条形图 -------------------------------------------------------------------
    block_height = 0.5  # 设置堆叠小方块的高度
    block_spacing = 0.05  # 设置堆叠小方块之间的间隙
    outer_bar_bottom_radius = (
        ring_positions.max() + ring_thickness + 3
    )  # 计算最外层堆叠条形图的起始径向位置
    for gene_idx in range(num_genes):  # 遍历每个特征
        num_blocks = int(
            np.sum(heatmap_data[:, gene_idx])
        )  # 计算该特征在所有图层中被选中的总次数，即堆叠方块的数量
        if num_blocks == 0:  # 如果总分为0
            continue  # 则跳过，不绘制
        current_bottom = outer_bar_bottom_radius  # 初始化当前方块的起始径向位置
        for _ in range(num_blocks):  # 根据总分绘制相应数量的方块
            ax.bar(
                x=theta[gene_idx],  # 角度位置
                height=block_height,  # 方块高度
                width=width * 0.7,  # 方块宽度（比热图条形稍窄）
                bottom=current_bottom,  # 起始径向位置
                color=bar_color,  # 颜色
                align="edge",  # 对齐方式
            )
            current_bottom += block_height + block_spacing  # 更新下一个方块的起始径向位置
    # -------------------------------------------------------------绘制特征名称-------------------------------------------------------------------
    label_radius = ring_positions.max() + ring_thickness - 0.7  # 计算特征名称的径向位置
    for i in range(num_genes):  # 遍历每个特征
        angle_deg = np.rad2deg(theta[i]) % 360  # 将特征的角度转换为度
        rotation = angle_deg  # 设置特征名称标注的旋转角度
        # 在图上添加特征名称
        ax.text(
            theta[i] + width / 2,  # 角度位置
            label_radius,  # 径向位置
            gene_names[i],  # 内容
            rotation=rotation,  # 旋转角度
            ha="left",  # 水平对齐
            va="center",  # 垂直对齐
            fontsize=8,  # 字体大小
            rotation_mode="anchor",  # 旋转模式
        )
    # 绘制虚线圈和开口处
    dashed_circle_radius = ring_positions.max() + ring_thickness + 2.8  # 计算虚线参考圈的径向位置
    # 绘制虚线圈
    ax.plot(
        np.linspace(start_angle, end_angle, 200),
        [dashed_circle_radius] * 200,
        color=bar_color,
        linestyle="--",
        lw=1,
        zorder=2,
    )

    label_angle = np.pi / 2  # 设置开口处角度位置
    gap_gray_color = "#f0f0f0"  # 设置开口处背景条的颜色
    gray_bar_height = (
        (ring_positions.max() + ring_thickness) - ring_positions.min() - 1.2
    )  # 计算背景条的高度
    gray_bar_width = 1.5 * angular_width_per_position  # 计算背景条的角宽度
    # 绘制开口处的背景条
    ax.bar(
        x=label_angle,  # 角度位置
        height=gray_bar_height,  # 高度
        width=gray_bar_width,  # 宽度
        bottom=ring_positions[0],  # 起始径向位置
        color=gap_gray_color,  # 颜色
        align="center",  # 居中对齐
        zorder=0,  # 绘图顺序
        edgecolor="white",  # 边框颜色
        linewidth=1,  # 边框宽度
    )
    # 动态绘制开口处的数字
    for i in range(num_layers):  # 遍历每个图层
        r = ring_positions[i] + ring_thickness / 2  # 计算数字的径向位置
        ax.text(
            label_angle,
            r,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#555555",
            zorder=3,
        )

    # 细节调整
    ax.grid(False)  # 关闭极坐标网格
    ax.set_yticklabels([])  # 去掉径向刻度标签
    ax.set_xticklabels([])  # 去掉角度刻度标签
    ax.spines["polar"].set_visible(False)  # 去掉极坐标的外框
    max_stack_height = num_layers * (
        block_height + block_spacing
    )  # 计算最外层堆叠条形图可能的最大高度
    max_radius = outer_bar_bottom_radius + max_stack_height  # 计算整个图表的最大半径
    ax.set_ylim(0, max_radius + 2)  # 设置径向范围

    # 添加图例
    ring_patches = [
        mpatches.Patch(color=ring_colors[i], label=layer_labels[i]) for i in range(num_layers)
    ]  # 为每个环创建图例
    bar_patch = mpatches.Patch(
        color=bar_color, label="Overall Evidence Score"
    )  # 为外部堆叠条形图创建图例
    all_patches = ring_patches + [bar_patch]  # 合并所有图例

    legend = ax.legend(handles=all_patches, loc="center", frameon=True, fontsize=11)  # 添加图例
    frame = legend.get_frame()  # 图例的边框
    frame.set_facecolor("white")  # 图例背景色
    frame.set_edgecolor("white")  # 图例边框色

    return fig, ax  # 返回图窗和子图对象


if __name__ == "__main__":
    # 手动设置要处理的任务类型: 'binary', 'multiclass', 或 'regression'
    task_type = "binary"
    # 指定数据文件路径
    data_file_path = str(DATA_DIR / "simulated_binary_data.xlsx")  # 指定输入数据的文件路径
    # 指定的目标列的名称
    target_column_name = "Target"
    # 选择配色方案
    selected_color_scheme = 1
    # 设置保存路径

    df = pd.read_excel(data_file_path)  # 读取数据
    # 提取X和y
    y = df[target_column_name].values
    X_df = df.drop(columns=[target_column_name])
    X = X_df.values
    # 获取特征名称
    feature_names = X_df.columns.tolist()
    n_features = X.shape[1]  # 获取特征的数量
    print(f"\n步骤 2: 正在为 {task_type} 任务准备和训练模型")
    if task_type == "binary":  # 如果任务是二分类
        model_instance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif task_type == "multiclass":  # 如果任务是多分类
        model_instance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif task_type == "regression":  # 如果任务是回归
        model_instance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        print(f"错误：未知的任务类型 '{task_type}'")
        exit()
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = model_instance.fit(X_train, y_train)  # 训练模型

    num_dimensions = 9  # 设置评估的总维度/图层数
    heatmap_data = np.zeros((num_dimensions, n_features))  # 用于存储评估结果
    layer_labels = [""] * num_dimensions  # 用于存储图例标签
    # 1.从训练好的随机森林模型中获取基尼重要性
    importances = model.feature_importances_
    heatmap_data[0, :] = (importances >= np.percentile(importances, 75)).astype(
        int
    )  # 标记重要性排名前25%的特征为1
    layer_labels[0] = "1. High Gini Importance (RF)"  # 设置图层1的标签

    # 2:置换重要性（Permutation Importance）
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )  # 计算
    heatmap_data[1, :] = (
        perm_importance.importances_mean > np.percentile(perm_importance.importances_mean, 75)
    ).astype(int)  # 平均重要性
    layer_labels[1] = "2. High Permutation Importance (Mean)"  # 设置图层的标签

    # 3:SHAP重要性
    explainer = shap.TreeExplainer(model)  # 创建SHAP解释器
    explanation = explainer(X_train)  # 计算训练集上的SHAP值
    if task_type == "regression":  # 如果是回归任务
        mean_abs_shap = np.abs(explanation.values).mean(axis=0)  # SHAP值平均绝对值
    elif task_type == "binary":  # 如果是二分类任务
        mean_abs_shap = np.abs(explanation.values[:, :, 1]).mean(axis=0)
    else:  # 如果是多分类任务
        mean_abs_shap = np.abs(explanation.values).mean(axis=(0, 2))

    heatmap_data[2, :] = (mean_abs_shap >= np.percentile(mean_abs_shap, 75)).astype(
        int
    )  # 标记平均绝对SHAP值排名前25%的特征
    layer_labels[2] = "3. High SHAP Value (Mean Abs)"

    # 4: L1 正则化 (Lasso回归 / 逻辑回归)
    scaler = StandardScaler()  # 实例化一个标准化处理器
    X_scaled = scaler.fit_transform(X_train)  # 对所有特征进行标准化

    if task_type == "regression":  # 如果是回归任务
        l1_estimator = Lasso(alpha=0.01, random_state=42)  # 使用 Lasso 回归
        layer_labels[3] = "4. Selected by L1 (Lasso)"  # 设置图层5的标签
    else:  # 如果是分类任务
        l1_estimator = LogisticRegression(
            penalty="l1", C=0.1, solver="liblinear", random_state=42
        )  # 使用带L1惩罚的逻辑回归
        layer_labels[3] = "4. Selected by L1 (Logistic)"  # 设置图层5的标签

    selector_l1 = SelectFromModel(l1_estimator, threshold="median").fit(
        X_scaled, y_train
    )  # 使用SelectFromModel和L1模型选择特征（阈值为系数中位数）
    heatmap_data[3, :] = selector_l1.get_support().astype(int)  # 标记被L1方法选中的特征

    # 5:RFE
    if task_type == "regression":  # 如果是回归任务
        rfe_estimator = DecisionTreeRegressor(
            max_depth=5, random_state=42
        )  # 使用决策树回归器作为RFE的基础评估器
    else:  # 如果是分类任务
        rfe_estimator = DecisionTreeClassifier(max_depth=5, random_state=42)

    n_top_features = int(n_features * 0.25)  # 设定RFE要选择的特征数量
    rfe = RFE(rfe_estimator, n_features_to_select=n_top_features, step=0.1)  # 实例化RFE选择器
    rfe.fit(X_train, y_train)  # 在训练集上拟合RFE

    heatmap_data[4, :] = rfe.support_.astype(int)  # 标记被RFE选中的特征
    layer_labels[4] = f"5. Selected by RFE (Top {n_top_features})"  # 设置图层6的标签

    # 6:Spearman相关性
    spearman_corrs = [spearmanr(X[:, i], y) for i in range(n_features)]  # 计算Spearman相关性
    spearman_vals = np.nan_to_num([c.correlation for c in spearman_corrs])  # 提取相关系数值
    spearman_pvals = np.nan_to_num([c.pvalue for c in spearman_corrs], nan=1.0)  # 提取P值

    spearman_significant = spearman_pvals < 0.05  # 筛选条件1：P值显著（< 0.05）
    spearman_high_value = np.abs(spearman_vals) >= np.nanpercentile(
        np.abs(spearman_vals), 75
    )  # 筛选条件2：相关性绝对值排名

    heatmap_data[5, :] = (spearman_significant & spearman_high_value).astype(
        int
    )  # 同时满足两个条件的特征
    layer_labels[5] = "6. High & Significant Spearman Correlation"  # 图层7的标签

    # 7:过滤法
    if task_type == "regression":  # 如果是回归任务
        f_values, p_values = f_regression(X, y)  # F检验
        f_test_label = "F-score (Regression)"  # 设置标签文本
    else:  # 如果是分类任务
        f_values, p_values = f_classif(X, y)  # ANOVA F检验
        f_test_label = "F-score (ANOVA)"  # 设置标签文本
    f_values, p_values = (
        np.nan_to_num(f_values),
        np.nan_to_num(p_values, nan=1.0),
    )  # 将NaN分别转为0和1.0
    ftest_significant = p_values < 0.05  # 筛选条件1，P值显著（< 0.05）
    ftest_high_score = f_values >= np.nanpercentile(f_values, 75)  # 筛选条件2，F分数排名前25%
    heatmap_data[6, :] = (ftest_significant & ftest_high_score).astype(
        int
    )  # 标记同时满足两个条件的特征
    layer_labels[6] = f"7. High & Significant {f_test_label}"  # 图层8的标签

    # 8:互信息法
    if task_type == "regression":  # 如果是回归任务
        mi_scores = mutual_info_regression(X, y, random_state=42)  # 计算回归互信息
        layer_labels[7] = "8. High Mutual Information (Regression)"  # 图层9的标签
    else:  # 如果是分类任务
        mi_scores = mutual_info_classif(X, y, random_state=42)  # 计算分类互信息
        layer_labels[7] = "8. High Mutual Information (Classification)"  # 图层9的标签
    heatmap_data[7, :] = (mi_scores >= np.percentile(mi_scores, 75)).astype(
        int
    )  # 互信息分数排名前25%的特征

    # 9:Pearson相关系数
    correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)]  # 计算Pearson相关系数
    correlations_arr = np.nan_to_num(np.array(correlations))  # 转换为Numpy数组并替换NaN
    heatmap_data[8, :] = (
        np.abs(correlations_arr) > np.nanpercentile(np.abs(correlations_arr), 75)
    ).astype(int)  # 相关系数绝对值排名前25%的特征
    layer_labels[8] = "9. High Pearson Correlation"

    # 调用函数进行绘图
    fig, ax = plot_multi_omics_circos(
        gene_names=feature_names,  # 特征名称
        heatmap_data=heatmap_data,  # 评估结果矩阵
        layer_labels=layer_labels,  # 图层标签
        color_scheme_id=selected_color_scheme,  # 配色方案
    )
    # 保存
    fig.savefig(
        str(OUTPUT_DIR / f"{task_type}{selected_color_scheme}.png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        str(OUTPUT_DIR / f"{task_type}{selected_color_scheme}.pdf"),
        bbox_inches="tight",
        transparent=True,
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

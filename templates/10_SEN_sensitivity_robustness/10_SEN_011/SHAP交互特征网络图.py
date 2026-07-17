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
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ======================================2.颜色库=======================================
# =========================================================================================
COLOR_SCHEMES = {
    1: {"nodes": plt.cm.Greens, "edges": plt.cm.Purples},
    2: {"nodes": plt.cm.Blues, "edges": plt.cm.Oranges},
    3: {"nodes": plt.cm.Reds, "edges": plt.cm.Greys},
    4: {"nodes": plt.cm.Oranges, "edges": plt.cm.Blues},
    5: {"nodes": plt.cm.Purples, "edges": plt.cm.Greens},
    6: {"nodes": plt.cm.viridis, "edges": plt.cm.magma},
    7: {"nodes": plt.cm.plasma, "edges": plt.cm.viridis},
    8: {"nodes": plt.cm.inferno, "edges": plt.cm.cividis},
    9: {"nodes": plt.cm.YlGnBu, "edges": plt.cm.PuRd},
    10: {"nodes": plt.cm.GnBu, "edges": plt.cm.YlOrBr},
    11: {"nodes": plt.cm.BuGn, "edges": plt.cm.RdPu},
    12: {"nodes": plt.cm.PuBu, "edges": plt.cm.YlGn},
    13: {"nodes": plt.cm.Greys, "edges": plt.cm.RdBu},
    14: {"nodes": plt.cm.Spectral, "edges": plt.cm.copper},
    15: {"nodes": plt.cm.cool, "edges": plt.cm.Wistia},
    16: {"nodes": plt.cm.winter, "edges": plt.cm.spring},
    17: {"nodes": plt.cm.summer, "edges": plt.cm.autumn},
    18: {"nodes": plt.cm.bone, "edges": plt.cm.pink},
    19: {"nodes": plt.cm.ocean, "edges": plt.cm.hot},
    20: {"nodes": plt.cm.tab20c, "edges": plt.cm.tab20b},
}
scheme_index = 1  # 颜色方案
# 获取当前颜色方案
current_color_scheme = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])
# =========================================================================================
# ======================================3.形状标记库=======================================
# =========================================================================================
STYLE_SCHEMES = {
    1: {"marker": "o", "linestyle": "-"},
    2: {"marker": "s", "linestyle": "--"},
    3: {"marker": "*", "linestyle": "-"},
    4: {"marker": "D", "linestyle": "-."},
    5: {"marker": "p", "linestyle": "-"},
    6: {"marker": "h", "linestyle": "--"},
    7: {"marker": "*", "linestyle": ":"},
    8: {"marker": "*", "linestyle": "-."},
    9: {"marker": "8", "linestyle": "-"},
    10: {"marker": "X", "linestyle": "--"},
}

style_index = 1  # 形状标记方案
# 获取样式方案
current_style_scheme = STYLE_SCHEMES.get(style_index, STYLE_SCHEMES[1])
# =========================================================================================
# ======================================4.数据加载=======================================
# =========================================================================================
# 原始数据路径
file_path = str(DATA_DIR / "mock_data.xlsx")
# 读取数据
df = pd.read_excel(file_path)
# 目标变量
y = df.iloc[:, -1]
# 特征变量
X = df.iloc[:, :-1]
# 获取特征列的名称并转换为列表
features = X.columns.tolist()
print(f"特征: {features}")
print(f"数据类型: {X.shape}")
# =========================================================================================
# ======================================5.数据划分及模型构建=======================================
# =========================================================================================
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 超参数网格
param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
}
# 初始化XGBoost 回归模型
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
# 初始化网格搜索对象
grid_search = GridSearchCV(
    estimator=xgb_model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error", verbose=1
)
# 在训练集上拟合
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
# 获取最佳模型
best_model = grid_search.best_estimator_
# =========================================================================================
# ======================================6.SHAP分析=======================================
# =========================================================================================
# 使用最佳模型创建SHAP树解释器
explainer = shap.TreeExplainer(best_model)
# 测试集的SHAP交互值
shap_interaction_values = explainer.shap_interaction_values(X_test)
# 测试集的SHAP值
shap_values = explainer.shap_values(X_test)
# 特征重要性,绝对值后的平均值
feature_importance = np.abs(shap_values).mean(axis=0)
# 对特征重要性进行归一化处理
importance_scaled = feature_importance

# 计算平均交互矩阵,绝对值后的平均值
mean_interaction_matrix = np.abs(shap_interaction_values).mean(axis=0)
# 将对角线元素填充为 0,忽略特征自身的交互
np.fill_diagonal(mean_interaction_matrix, 0)


# =========================================================================================
# ======================================7.绘图函数=======================================
# =========================================================================================
def plot_circular_interaction(features, importance, interaction_matrix):
    # 获取节点的颜色映射
    cmap_nodes = current_color_scheme["nodes"]
    # 获取连线的颜色映射
    cmap_edges = current_color_scheme["edges"]
    # 获取节点形状标记
    node_marker = current_style_scheme["marker"]
    # 获取连线样式
    edge_linestyle = current_style_scheme["linestyle"]
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"aspect": "equal"})
    # 获取特征的数量
    n_features = len(features)
    # 创建一个 NetworkX 图对象
    G = nx.Graph()
    # 向图中添加节点
    G.add_nodes_from(features)
    # 生成节点的环形布局坐标
    pos = nx.circular_layout(G)
    # 标签的坐标
    label_pos = {k: (v * 1.1) for k, v in pos.items()}
    # 定义节点颜色的归一化范围
    norm_nodes = mcolors.Normalize(vmin=0, vmax=feature_importance.max())
    # 获取交互矩阵中的最大值
    max_interaction = interaction_matrix.max()
    # 定义边颜色的归一化范围
    norm_edges = mcolors.Normalize(vmin=0, vmax=max_interaction)
    # 初始化交互列表
    interactions = []
    # 遍历特征
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # 两个特征之间的交互强度
            strength = interaction_matrix[i, j]
            # 如果交互强度大于 0
            if strength > 0:
                # 将交互对和强度添加到列表中
                interactions.append((features[i], features[j], strength))
    # 根据强度对交互列表进行排序
    interactions.sort(key=lambda x: x[2])
    # 遍历排序后的交互列表
    for u, v, strength in interactions:
        # 根据强度获取线的颜色
        color = cmap_edges(norm_edges(strength))
        # 线的粗细
        width = 0.5 + (strength / max_interaction) * 8
        # 线的透明度
        alpha = 0.1 + (strength / max_interaction) * 0.9
        # 绘制线
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=[color],
            style=edge_linestyle,
            alpha=alpha,
            ax=ax,
        )
    # 初始化节点颜色列表
    node_colors = []
    # 初始化节点大小列表
    node_sizes = []
    # 遍历每个特征
    for i, feat in enumerate(features):
        # 获取该特征的重要性
        imp = importance[i]
        # 计算并添加节点颜色
        node_colors.append(cmap_nodes(norm_nodes(imp)))
        # 计算并添加节点大小
        node_sizes.append(imp * 30.0)
    # 绘制节点
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="grey",
        linewidths=0.5,
        node_shape=node_marker,
        ax=ax,
    )
    # 遍历标签位置字典
    for node, (x, y) in label_pos.items():
        ha = "center"  # 水平对齐方式
        # 如果 x 坐标在右侧
        if x > 0.1:
            ha = "left"  # 设置左对齐
        # 如果 x 坐标在左侧
        elif x < -0.1:
            ha = "right"  # 设置右对齐
        # 绘制标签文本
        plt.text(x, y, node, size=12, horizontalalignment=ha, verticalalignment="center")
    # 关闭坐标轴
    ax.axis("off")
    # x轴显示范围
    ax.set_xlim(-1.5, 1.5)
    # y轴显示范围
    ax.set_ylim(-1.5, 1.5)
    # 标题
    plt.title(
        "(a) Green Ecological -> Agricultural Production",
        y=0.95,
        fontsize=16,
        fontname="Times New Roman",
    )
    # 定义边颜色条的位置，左,下,宽,高
    cbar_edge_pos = [0.82, 0.55, 0.015, 0.25]
    # 创建一个新的轴用于放颜色条
    cax_edge = fig.add_axes(cbar_edge_pos)
    # 创建边颜色的标量映射对象
    sm_edge = plt.cm.ScalarMappable(
        cmap=cmap_edges, norm=mcolors.Normalize(vmin=0, vmax=int(max_interaction))
    )
    # 设置空数组
    sm_edge.set_array([])
    # 绘制线的颜色条
    cbar_edge = plt.colorbar(sm_edge, cax=cax_edge)
    # 设置线的颜色条的标签
    cbar_edge.set_label(
        "Interaction Strength", rotation=270, labelpad=15, fontsize=10, fontname="Times New Roman"
    )
    # 去掉线的颜色条的轮廓线
    cbar_edge.outline.set_visible(False)
    # 节点颜色条的位置
    cbar_node_pos = [0.82, 0.20, 0.015, 0.25]
    # 添加节点颜色条的轴
    cax_node = fig.add_axes(cbar_node_pos)
    # 创建节点颜色的标量映射对象
    sm_node = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    # 设置空数组
    sm_node.set_array([])
    # 绘制节点颜色条
    cbar_node = plt.colorbar(sm_node, cax=cax_node)
    # 设置节点颜色条的标签
    cbar_node.set_label(
        "Importance", rotation=270, labelpad=15, fontsize=10, fontname="Times New Roman"
    )
    # 去掉节点颜色条的轮廓线
    cbar_node.outline.set_visible(False)
    # 保存
    save_path_png = str(OUTPUT_DIR / f"{style_index}_scheme{scheme_index}.png")
    save_path_pdf = str(OUTPUT_DIR / f"{style_index}_scheme{scheme_index}.pdf")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(save_path_pdf, bbox_inches="tight")


if __name__ == "__main__":
    # 调用绘图函数
    plot_circular_interaction(features, importance_scaled, mean_interaction_matrix)

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

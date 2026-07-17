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
import factor_analyzer.factor_analyzer as _factor_analyzer_module
import matplotlib
import matplotlib.colorbar as colorbar
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _factor_analyzer_check_array_compat(*args, force_all_finite=True, **kwargs):
    """Bridge factor_analyzer 0.5.x to scikit-learn's renamed argument."""
    kwargs.setdefault("ensure_all_finite", force_all_finite)
    return _factor_analyzer_original_check_array(*args, **kwargs)


_factor_analyzer_original_check_array = _factor_analyzer_module.check_array
_factor_analyzer_module.check_array = _factor_analyzer_check_array_compat

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: {
        "neg_loading": "#D3D3D3",
        "weak_pos_loading": "#B0C4DE",
        "strong_pos_loading": "#F08080",
        "corr_cmap": "RdBu_r",
    },
    2: {
        "neg_loading": "#c7e9c0",
        "weak_pos_loading": "#74c476",
        "strong_pos_loading": "#238b45",
        "corr_cmap": "PiYG",
    },
    3: {
        "neg_loading": "#d1e5f0",
        "weak_pos_loading": "#67a9cf",
        "strong_pos_loading": "#2166ac",
        "corr_cmap": "viridis",
    },
    4: {
        "neg_loading": "#fde0ef",
        "weak_pos_loading": "#f1b6da",
        "strong_pos_loading": "#de77ae",
        "corr_cmap": "PRGn",
    },
    5: {
        "neg_loading": "#E0E0E0",
        "weak_pos_loading": "#FFC107",
        "strong_pos_loading": "#F57C00",
        "corr_cmap": "plasma",
    },
    6: {
        "neg_loading": "#dadaeb",
        "weak_pos_loading": "#9e9ac8",
        "strong_pos_loading": "#6a51a3",
        "corr_cmap": "BrBG",
    },
    7: {
        "neg_loading": "#fee0d2",
        "weak_pos_loading": "#fc9272",
        "strong_pos_loading": "#de2d26",
        "corr_cmap": "inferno",
    },
    8: {
        "neg_loading": "#E3F2FD",
        "weak_pos_loading": "#64B5F6",
        "strong_pos_loading": "#1976D2",
        "corr_cmap": "seismic",
    },
    9: {
        "neg_loading": "#f7f7f7",
        "weak_pos_loading": "#cccccc",
        "strong_pos_loading": "#969696",
        "corr_cmap": "coolwarm_r",
    },
    10: {
        "neg_loading": "#F3E5F5",
        "weak_pos_loading": "#BA68C8",
        "strong_pos_loading": "#7B1FA2",
        "corr_cmap": "magma",
    },
    11: {
        "neg_loading": "#fff5eb",
        "weak_pos_loading": "#fdbf6f",
        "strong_pos_loading": "#ff7f00",
        "corr_cmap": "PuOr",
    },
    12: {
        "neg_loading": "#fcfbfd",
        "weak_pos_loading": "#efedf5",
        "strong_pos_loading": "#756bb1",
        "corr_cmap": "RdYlGn",
    },
    13: {
        "neg_loading": "#E8F5E9",
        "weak_pos_loading": "#A5D6A7",
        "strong_pos_loading": "#43A047",
        "corr_cmap": "cividis",
    },
    14: {
        "neg_loading": "#FBE9E7",
        "weak_pos_loading": "#FFAB91",
        "strong_pos_loading": "#F4511E",
        "corr_cmap": "Spectral_r",
    },
    15: {
        "neg_loading": "#e5f5e0",
        "weak_pos_loading": "#a1d99b",
        "strong_pos_loading": "#31a354",
        "corr_cmap": "RdGy_r",
    },
    16: {
        "neg_loading": "#E0F7FA",
        "weak_pos_loading": "#4DD0E1",
        "strong_pos_loading": "#0097A7",
        "corr_cmap": "ocean",
    },
    17: {
        "neg_loading": "#F1F8E9",
        "weak_pos_loading": "#C5E1A5",
        "strong_pos_loading": "#7CB342",
        "corr_cmap": "terrain",
    },
    18: {
        "neg_loading": "#FFF8E1",
        "weak_pos_loading": "#FFD54F",
        "strong_pos_loading": "#FFA000",
        "corr_cmap": "gist_stern",
    },
    19: {
        "neg_loading": "#F9FBE7",
        "weak_pos_loading": "#E6EE9C",
        "strong_pos_loading": "#C0CA33",
        "corr_cmap": "cubehelix",
    },
    20: {
        "neg_loading": "#e0f2f1",
        "weak_pos_loading": "#80cbc4",
        "strong_pos_loading": "#26a69a",
        "corr_cmap": "GnBu_r",
    },
}
SELECTED_SCHEME = 20  # 配色方案
N_COMPONENTS = 4  # 设置主成分分析（PCA）要提取的主成分数量


# =========================================================================================
# ====================================== 3. 绘图函数 =========================================
# =========================================================================================
def plot_correlation_and_pca(corr_matrix, loadings, metals, scheme_id, n_components):
    # 接收相关矩阵、载荷、特征名称、配色方案ID和主成分数量作为参数
    n_features = len(metals)  # 获取特征的数量
    scheme = COLOR_SCHEMES.get(scheme_id, COLOR_SCHEMES[1])  # 配色方案
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    COLOR_NEG_LOADING = scheme["neg_loading"]  # 从选定的配色方案中获取用于表示负PCA载荷的颜色
    COLOR_WEAK_POS_LOADING = scheme["weak_pos_loading"]  # 从配色方案中获取用于表示弱正PCA载荷的颜色
    COLOR_STRONG_POS_LOADING = scheme[
        "strong_pos_loading"
    ]  # 从配色方案中获取用于表示强正PCA载荷的颜色
    corr_cmap = plt.get_cmap(
        scheme["corr_cmap"]
    )  # 从配色方案中获取相关性热图的颜色映射名称，并生成颜色映射对象
    corr_norm = Normalize(
        vmin=-1, vmax=1
    )  # 创建一个归一化对象，将相关系数的值域（-1到1）映射到颜色范围（0到1）

    grid_line_width = 1.5  # 定义网格线的宽度
    grid_line_color = "lightgray"  # 定义网格线的颜色
    z_order = 1.5  # 定义网格线的绘制层级
    ax.plot(
        [0, 0], [1, n_features], color=grid_line_color, linewidth=grid_line_width, zorder=z_order
    )  # 绘制最左侧的垂直网格线
    ax.plot(
        [0, n_features - 1],
        [n_features, n_features],
        color=grid_line_color,
        linewidth=grid_line_width,
        zorder=z_order,
    )  # 绘制最底部的水平网格线
    # 绘制内部水平线
    for i in range(1, n_features):  # 循环绘制从第1条到倒数第2条的水平线
        ax.plot(
            [0, i], [i, i], color=grid_line_color, linewidth=grid_line_width, zorder=z_order
        )  # 每条水平线从x=0延伸至对角线位置x=i
    # 绘制内部垂直线
    for j in range(1, n_features):  # 循环绘制从第1条到倒数第2条的垂直线
        ax.plot(
            [j, j],
            [j, n_features],
            color=grid_line_color,
            linewidth=grid_line_width,
            zorder=z_order,
        )  # 每条垂直线从对角线位置y=j延伸至底部y=n_features

    # 填充背景色块和数据色块
    for i in range(n_features):  # 外层循环，遍历特征的行索引
        for j in range(n_features):  # 内层循环，遍历特征的列索引
            if i >= j:  # 判断当前单元格是否在主对角线及其左下方区域
                rect_bg = patches.Rectangle(
                    (j, i), 1, 1, facecolor="white", edgecolor="none", zorder=1
                )  # 创建一个无边框的白色背景矩形，作为底层背景
                ax.add_patch(rect_bg)  # 将这个白色背景矩形添加到子图中
            if i > j:  # 判断当前单元格是否在主对角线以下的区域
                x, y = j + 0.5, i + 0.5  # 计算当前单元格的中心点坐标
                corr_val = corr_matrix.iloc[i, j]  # 从相关性矩阵中获取(i, j)位置的相关系数值
                size = (
                    abs(corr_val) * 0.9
                )  # 根据相关系数的绝对值计算方块的大小，乘以0.9使其不完全填满单元格
                color = corr_cmap(
                    corr_norm(corr_val)
                )  # 根据相关系数值、归一化规则和颜色映射，计算出方块的颜色
                bottom_left_x = x - size / 2  # 计算前景方块左下角的x坐标
                bottom_left_y = y - size / 2  # 计算前景方块左下角的y坐标
                rect_fg = patches.Rectangle(
                    (bottom_left_x, bottom_left_y),
                    size,
                    size,  # 创建一个代表相关性强弱和正负的前景方块
                    linewidth=1.5,
                    edgecolor="none",  # 设置线宽为1.5，边框颜色为无
                    facecolor=color,
                    zorder=2,
                )  # 设置填充颜色，并设置一个更高的绘制层级（z-order=2），使其在背景之上
                ax.add_patch(rect_fg)  # 将这个前景方块添加到子图中

    ax.set_xticks(np.arange(n_features) + 0.5)  # 设置x轴的刻度位置
    ax.set_xticklabels(metals, fontsize=14)  # 设置x轴的刻度标签指定字体大小为14
    ax.set_yticks(np.arange(n_features) + 0.5)  # 设置y轴的刻度位置
    ax.set_yticklabels(metals, fontsize=14)  # 设置y轴的刻度标签并指定字体大小为14
    ax.tick_params(
        axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, pad=-15
    )  # x轴刻度
    ax.tick_params(
        axis="y", left=True, right=False, labelleft=True, labelright=False, pad=-15
    )  # y轴刻度
    ax.invert_yaxis()  # 翻转y轴，使(0,0)坐标位于图形的左上角，这符合矩阵的常规表示方式

    # ==================== 绘制PCA载荷连接 ====================
    metal_node_coords = [
        (i + 0.5, i + 0.5) for i in range(n_features)
    ]  # 计算每个特征在对角线上的节点坐标（即每个对角单元格的中心）

    pc_center_x = n_features - 0.5  # 设置PC节点群的中心x坐标
    pc_center_y = n_features / 2 - 1.7  # 设置PC节点群的中心y坐标
    spread_per_node = 1.5  # 定义PC节点之间的间距因子，保持原始值

    total_spread_range = (n_components - 1) * spread_per_node  # 计算所有PC节点分布的总范围
    start_offset = -total_spread_range / 2  # 计算第一个PC节点的起始偏移量
    end_offset = total_spread_range / 2  # 计算最后一个PC节点的结束偏移量
    offsets = np.linspace(
        start_offset, end_offset, n_components
    )  # 在起始和结束偏移量之间生成等间距的n_components个偏移值

    pc_node_coords = [
        (pc_center_x + offset, pc_center_y + offset) for offset in offsets
    ]  # 根据中心点和每个节点的偏移量计算出所有PC节点的最终坐标
    pc_names = [
        f"PC{i + 1}" for i in range(n_components)
    ]  # 动态生成PC名称列表，例如 ['PC1', 'PC2', 'PC3']

    metal_nodes_x = [coord[0] for coord in metal_node_coords]  # 提取所有特征的x坐标
    metal_nodes_y = [coord[1] for coord in metal_node_coords]  # 提取所有特征的y坐标
    ax.scatter(
        metal_nodes_x,
        metal_nodes_y,
        s=80,
        facecolor="red",
        edgecolor="blue",
        linewidth=1.5,
        zorder=10,
    )  # 在图上绘制特征的散点，设置大小、颜色、边框等属性，zorder设为10使其在最上层

    pc_nodes_x = [coord[0] for coord in pc_node_coords]  # 提取所有PC节点的x坐标
    pc_nodes_y = [coord[1] for coord in pc_node_coords]  # 提取所有PC节点的y坐标
    ax.scatter(
        pc_nodes_x, pc_nodes_y, facecolor="red", edgecolor="blue", s=60, zorder=10
    )  # 在图上绘制PC节点的散点
    for i, name in enumerate(pc_names):  # 循环遍历每个PC节点的名称和索引
        ax.text(
            pc_node_coords[i][0] + 0.4,
            pc_node_coords[i][1],
            name,
            ha="left",
            va="center",
            fontsize=14,
        )  # 在每个PC节点的右侧添加文本标签，并设置对齐方式和字体大小

    base_rad = 0.15  # 设置连接线的基本曲率半径

    for j in range(n_components):  # 外层循环，遍历每个主成分
        for i in range(n_features):  # 内层循环，遍历每个特征
            loading_val = loadings[i, j]  # 获取第i个特征在第j个主成分上的载荷值
            if loading_val > 0.5:  # 如果载荷值大于0.5
                color = COLOR_STRONG_POS_LOADING  # 设置颜色为强正载荷颜色
            elif loading_val >= 0:  # 如果载荷值在0和0.5之间（包括0）
                color = COLOR_WEAK_POS_LOADING  # 设置颜色为弱正载荷颜色
            else:  # 如果载荷值为负数
                color = COLOR_NEG_LOADING  # 设置颜色为负载荷颜色
            curve_rad = base_rad if i > 3 else -base_rad  # 根据特征元素的索引i设置弧线的弯曲方向
            con = patches.ConnectionPatch(  # 在两个点之间绘制连接线
                xyA=metal_node_coords[i],  # 特征
                xyB=pc_node_coords[j],  # PC节点
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,  # 指定坐标系为数据坐标系，并关联到当前子图
                arrowstyle="-",
                connectionstyle=f"arc3,rad={curve_rad}",  # 设置连接样式为弧线，并指定曲率半径
                color=color,
                linewidth=1.5,
                zorder=5,  # 设置线条颜色、宽度和绘制层级
            )
            ax.add_patch(con)  # 将创建的连接线添加到子图中

    legend_elements = [  # 图例
        Line2D([0], [0], color=COLOR_NEG_LOADING, lw=2, label="< 0"),
        Line2D([0], [0], color=COLOR_WEAK_POS_LOADING, lw=2, label="0 - 0.5"),
        Line2D([0], [0], color=COLOR_STRONG_POS_LOADING, lw=2, label="> 0.5"),
    ]
    ax.legend(
        handles=legend_elements,
        title="Factor loading",  # 在子图上创建图例
        loc="upper left",
        bbox_to_anchor=(-0.2, 0.9),  # 设置图例位置在左上角
        fontsize=12,
        title_fontsize=14,
        frameon=False,
    )  # 设置字体大小，并去掉图例边框

    cbar_ax = fig.add_axes([0.2, 0.15, 0.02, 0.45])  # [左, 下, 宽, 高] 放置颜色条
    cb = colorbar.ColorbarBase(
        cbar_ax, cmap=corr_cmap, norm=corr_norm, orientation="vertical"
    )  # 在新创建的坐标轴上绘制颜色条

    cb.outline.set_visible(False)  # 去掉颜色条边框
    cb.ax.tick_params(length=0)  # 去掉刻度线

    cb.ax.set_title("Pearson's r", size=14, pad=10)  # 设置标题
    cb.ax.yaxis.set_ticks_position("right")  # 将颜色条的刻度标签放在右侧
    cb.ax.tick_params(labelsize=12)  # 设置颜色条刻度标签的字体大小

    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)  # 隐藏主图四个坐标轴边框
    ax.tick_params(axis="both", which="both", length=0)  # 隐藏主图所有刻度线
    ax.set_aspect("equal", adjustable="box")  # 设置坐标轴的纵横比为1:1

    # 保存路径
    output_filename_png = str(OUTPUT_DIR / f"correlation_pca_{scheme_id}.png")
    output_filename_pdf = str(OUTPUT_DIR / f"correlation_pca_{scheme_id}.pdf")
    plt.savefig(output_filename_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_filename_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"绘图成功！结果已保存为 '{output_filename_png}' 和 '{output_filename_pdf}'")


# =========================================================================================
# ======================================4.数据的加载及预处理=========================================
# =========================================================================================
# ==================== 1. 从本地Excel文件加载数据 ====================
excel_path = str(DATA_DIR / "data.xlsx")  # 数据文件的路径
df = pd.read_excel(excel_path)  # 读取数据
print(f"成功从 '{excel_path}' 加载数据。")
metals = df.columns.tolist()  # 获取DataFrame的所有列名，并转换为一个列表
# ==================== 2. 执行统计分析 ====================
corr_matrix = df.corr()  # 计算DataFrame中各列之间的皮尔逊相关系数，并生成相关系数矩阵
scaler = StandardScaler()  # 创建一个StandardScaler对象，用于数据标准化
numeric_df = df.select_dtypes(include=np.number)  # 从原始DataFrame中选择所有数值类型的列
metals = numeric_df.columns.tolist()
scaled_data = scaler.fit_transform(numeric_df)  # 对数值数据进行拟合和转换，即进行标准化处理
# =========================================================================================
# ====================================== 6. 主成分分析=========================================
# =========================================================================================
# pca = PCA(n_components=N_COMPONENTS) # 创建一个PCA对象，并指定要提取的主成分数量
# pca.fit(scaled_data) # 使用标准化后的数据来训练PCA模型
# loadings = pca.components_.T # 获取PCA的载荷矩阵，.components_的形状是(n_components, n_features)，需要转置（.T）得到(n_features, n_components)形状
# 初始化模型
# n_factors主成分数量
# method='principal'用主成分法
# rotation='varimax'用“方差最大化”旋转
fa = FactorAnalyzer(n_factors=N_COMPONENTS, method="principal", rotation="varimax")
# 训练模型
fa.fit(scaled_data)
# 获取旋转后的载荷矩阵
loadings = fa.loadings_
# =========================================================================================
# ====================================== 7. 绘图 =========================================
# =========================================================================================
plot_correlation_and_pca(
    corr_matrix, loadings, metals, scheme_id=SELECTED_SCHEME, n_components=N_COMPONENTS
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

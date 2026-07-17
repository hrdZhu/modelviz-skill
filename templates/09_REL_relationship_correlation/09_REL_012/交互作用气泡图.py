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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2. 颜色库设置 ==========================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["#fde0dd", "#fcc5c0", "#fa9fb5", "#f768a1", "#dd3497", "#ae017e", "#7a0177", "#49006a"],
    2: ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
    3: ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#99000d"],
    4: ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
    5: ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#993404", "#662506"],
    6: ["#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#4a1486"],
    7: ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"],
    8: ["#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
    9: ["#f7f4f9", "#e7e1ef", "#d4b9da", "#c994c7", "#df65b0", "#e7298a", "#ce1256", "#91003f"],
    10: ["#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525", "#000000"],
    11: ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679", "#41ab5d", "#238443", "#005a32"],
    12: ["#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#034e7b"],
    13: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58"],
    14: ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fdb42f"],
    15: ["#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60", "#cf4446", "#ed6925", "#fb9b06"],
    16: ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00"],
    17: ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5"],
    18: ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"],
    19: ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"],
    20: ["#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3"],
}
# =========================================================================================
# ====================================== 3. 形状标记库设置 ==========================
# =========================================================================================
SHAPE_SCHEMES = {
    1: {"bubble": "o", "high": "↑↑", "mid": "↑"},
    2: {"bubble": "s", "high": "↑↑", "mid": "↑"},
    3: {"bubble": "D", "high": "↑↑", "mid": "↑"},
    4: {"bubble": "^", "high": "↑↑", "mid": "↑"},
    5: {"bubble": "h", "high": "↑↑", "mid": "↑"},
    6: {"bubble": "o", "high": "**", "mid": "*"},
    7: {"bubble": "s", "high": "**", "mid": "*"},
    8: {"bubble": "8", "high": "##", "mid": "#"},
    9: {"bubble": "p", "high": "++", "mid": "+"},
    10: {"bubble": "*", "high": "!!", "mid": "!"},
    11: {"bubble": "d", "high": "AA", "mid": "A"},
    12: {"bubble": "v", "high": "VV", "mid": "V"},
    13: {"bubble": "H", "high": "ZZ", "mid": "Z"},
    14: {"bubble": "X", "high": "XX", "mid": "X"},
    15: {"bubble": ">", "high": ">>", "mid": ">"},
    16: {"bubble": "o", "high": "High", "mid": "Mid"},
    17: {"bubble": "s", "high": "●●", "mid": "●"},
    18: {"bubble": "D", "high": "▲▲", "mid": "▲"},
    19: {"bubble": "p", "high": "SSS", "mid": "S"},
    20: {"bubble": "o", "high": "max", "mid": "nl"},
}
selected_color_id = 14  # 设置当前选用的颜色
selected_shape_id = 20  # 设置当前选用的形状
# =========================================================================================
# ====================================== 4. 数据准备=========================================
# =========================================================================================
q_interaction_matrix = np.full((11, 11), np.nan)  # 创建一个空数组，初始值全部为NaN
# 定义交互作用数据，值是对应的q值
interaction_data = {
    (1, 0): 0.46,
    (2, 0): 0.47,
    (2, 1): 0.28,
    (3, 0): 0.42,
    (3, 1): 0.30,
    (3, 2): 0.25,
    (4, 0): 0.29,
    (4, 1): 0.37,
    (4, 2): 0.41,
    (4, 3): 0.37,
    (5, 0): 0.30,
    (5, 1): 0.35,
    (5, 2): 0.38,
    (5, 3): 0.37,
    (5, 4): 0.24,
    (6, 0): 0.36,
    (6, 1): 0.37,
    (6, 2): 0.37,
    (6, 3): 0.35,
    (6, 4): 0.28,
    (6, 5): 0.33,
    (7, 0): 0.35,
    (7, 1): 0.32,
    (7, 2): 0.35,
    (7, 3): 0.35,
    (7, 4): 0.19,
    (7, 5): 0.28,
    (7, 6): 0.34,
    (8, 0): 0.24,
    (8, 1): 0.40,
    (8, 2): 0.42,
    (8, 3): 0.37,
    (8, 4): 0.24,
    (8, 5): 0.30,
    (8, 6): 0.35,
    (8, 7): 0.32,
    (9, 0): 0.20,
    (9, 1): 0.20,
    (9, 2): 0.23,
    (9, 3): 0.22,
    (9, 4): 0.06,
    (9, 5): 0.18,
    (9, 6): 0.25,
    (9, 7): 0.18,
    (9, 8): 0.18,
    (10, 0): 0.24,
    (10, 1): 0.28,
    (10, 2): 0.29,
    (10, 3): 0.27,
    (10, 4): 0.16,
    (10, 5): 0.24,
    (10, 6): 0.26,
    (10, 7): 0.25,
    (10, 8): 0.21,
    (10, 9): 0.13,
}
# 将交互作用数据填充到数组中
for (r, c), val in interaction_data.items():  # 遍历字典中的每一项
    q_interaction_matrix[r, c] = val  # 将对应的值赋给矩阵的具体位置

# 定义单因子的q值，用于绘制对角线上的气泡
q_individual_factors = np.array([0.20, 0.15, 0.25, 0.10, 0.30, 0.12, 0.22, 0.18, 0.28, 0.13, 0.27])
# 定义X轴和Y轴的标签文本
labels = ["Tmp", "Pre", "Win", "Hum", "Dem", "Slp", "Asp", "Veg", "Pop", "Roa", "Riv"]


# =========================================================================================
# ====================================== 5. 绘图函数=========================================
# =========================================================================================
def plot_interaction_chart(q_matrix, q_factors, label_list, color_id=1, shape_id=1):
    selected_colors = COLOR_SCHEMES.get(color_id, COLOR_SCHEMES[1])  # 获取颜色方案
    selected_shapes = SHAPE_SCHEMES.get(shape_id, SHAPE_SCHEMES[1])  # 形状方案

    bubble_marker = selected_shapes["bubble"]  # 提取形状标记
    mark_high = selected_shapes["high"]  # 双因子增强符号
    mark_mid = selected_shapes["mid"]  # 非线性增强符号

    # 获取因子的数量
    n = len(label_list)

    fig, ax = plt.subplots(figsize=(10, 9))  # 创建画布和坐标轴

    # 从选定的颜色列表创建线性分段的颜色映射对象
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", selected_colors)
    # 设置颜色映射的归一化范围
    vmin = 0.0
    vmax = 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 创建归一化对象

    for i in range(n):  # 遍历行
        for j in range(n):  # 遍历列
            cell_center_x = j + 0.5  # 单元格中心的X坐标
            cell_center_y = i + 0.5  # 单元格中心的Y坐标

            # 左下角区域
            if i > j:  # 判断是否在矩阵的下三角区域
                q_interaction = q_matrix[i, j]  # 获取对应的交互作用值
                if not np.isnan(q_interaction):  # 如果该值不是NaN
                    q_factor_i = q_factors[i]  # 获取行因子的单因子q值
                    q_factor_j = q_factors[j]  # 获取列因子的单因子q值
                    arrow_text = ""  # 初始化标记文本

                    if q_interaction > (q_factor_i + q_factor_j):  # 如果交互值大于两因子之和
                        arrow_text = mark_high  # 双因子增强符号
                    elif q_interaction > max(
                        q_factor_i, q_factor_j
                    ):  # 如果交互值大于任意单个因子的最大值
                        arrow_text = mark_mid  # 非线性增强符号

                    number_color = cmap(norm(q_interaction))  # 根据数值获取对应的颜色用于文字显示

                    ax.text(
                        cell_center_x,  # X坐标
                        cell_center_y,  # Y坐标
                        f"{q_interaction:.2f}",  # 数值
                        ha="center",  # 水平居中
                        va="center",  # 垂直居中
                        fontsize=14,  # 字体大小
                        fontweight="bold",  # 字体粗细
                        color=number_color,  # 文本颜色
                        zorder=3,
                    )  # 图层顺序

                    if arrow_text:  # 如果存在需要绘制的标记文本
                        arrow_y_position = cell_center_y + 0.22  # Y坐标（
                        ax.text(
                            cell_center_x,  # X坐标
                            arrow_y_position,  # Y坐标
                            arrow_text,  # 标记符号
                            ha="center",  # 水平
                            va="top",  # 垂直
                            fontsize=14,  # 字体大小
                            color="black",  # 颜色
                            weight="bold",  # 字体粗细
                            zorder=4,
                        )  # 图层顺序

            # 右上角
            elif i < j:  # 判断是否在矩阵的上三角区域
                q_interaction_for_bubble = q_matrix[j, i]  # 获取对称位置的交互作用值
                if not np.isnan(q_interaction_for_bubble):  # 如果值有效
                    bubble_size = max(
                        60, 500 * q_interaction_for_bubble
                    )  # 计算气泡大小，确保不小于最小值
                    # 绘制散点
                    ax.scatter(
                        cell_center_x,  # X坐标
                        cell_center_y,  # Y坐标
                        s=bubble_size,  # 散点大小
                        c=q_interaction_for_bubble,  # 颜色值
                        cmap=cmap,  # 应用颜色映射
                        norm=norm,  # 应用归一化
                        marker=bubble_marker,  # 散点形状
                        alpha=0.85,  # 透明度
                        edgecolors="none",  # 不显示边缘颜色
                        zorder=2,
                    )  # 图层顺序

            # 对角线区域
            elif i == j:  # 判断是否在对角线上
                q_single = q_factors[i]  # 获取单因子数值
                if not np.isnan(q_single):  # 如果值有效
                    bubble_size = max(60, 500 * q_single)  # 计算气泡大小
                    # 对角线
                    ax.scatter(
                        cell_center_x,  # X坐标
                        cell_center_y,  # Y坐标
                        s=bubble_size,  # 散点大小
                        c=q_single,  # 颜色值
                        cmap=cmap,  # 应用颜色映射
                        norm=norm,  # 应用归一化
                        marker=bubble_marker,  # 散点形状
                        alpha=0.85,  # 透明度
                        edgecolors="none",  # 不显示边缘颜色
                        zorder=2,
                    )  # 图层顺序

    tick_label_positions = np.arange(n) + 0.5  # 刻度标签的位置
    ax.set_xticks(tick_label_positions)  # 设置X轴刻度位置

    ax.set_xticklabels(
        label_list,  # X轴刻度标签文本
        fontsize=14,  # 字体大小
        fontweight="bold",
    )  # 字体粗细
    ax.set_yticks(tick_label_positions)  # 设置Y轴刻度位置
    ax.set_yticklabels(
        label_list,  # Y轴刻度标签文本
        fontsize=14,  # 字体大小
        fontweight="bold",
    )  # 字体粗细
    ax.xaxis.tick_top()  # 将X轴刻度移动到顶部
    ax.xaxis.set_label_position("top")  # 将X轴标签位置设置在顶部

    ax.grid(False)  # 关闭默认网格

    grid_line_positions = np.arange(n + 1)  # 生成网格线的位置数组
    grid_line_style = {
        "color": "lightgrey",
        "linestyle": ":",
        "linewidth": 0.7,
        "zorder": 0,
    }  # 定义网格线样式
    for k_grid in grid_line_positions:  # 遍历网格位置
        ax.axvline(k_grid, **grid_line_style)  # 绘制垂直网格线
        ax.axhline(k_grid, **grid_line_style)  # 绘制水平网格线

    ax.set_xlim(0, n)  # X轴范围
    ax.set_ylim(n, 0)  # Y轴范围
    ax.set_aspect("equal", adjustable="box")  # 设置纵横比相等
    ax.tick_params(axis="both", which="both", length=0)  # 去掉刻度线
    # 去掉边框
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # 创建ScalarMappable对象用于生成颜色条
    sm.set_array([])  # 设置空数组
    cbar = fig.colorbar(
        sm,
        ax=ax,  # 在当前axes上添加颜色条
        shrink=1,  # 缩放比例
        aspect=30,  # 长宽比
        pad=0.01,  # 间距
        orientation="vertical",
    )  # 方向

    cbar.set_label(
        "q-statistic Value",  # 颜色条标题
        rotation=270,  # 角度度
        labelpad=20,  # 间距
        fontsize=14,  # 字体大小
        fontweight="bold",  # 字体粗细
    )  # 设置字体族
    cbar.ax.tick_params(
        labelsize=14,  # 刻度标签大小
        length=3,  # 刻度线长度
        width=2,  # 刻度线宽度
        colors="black",
    )  # 颜色

    # 设置图表标题
    ax.set_title(
        "Factor Interaction and Enhancement Analysis", fontsize=16, pad=25, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])  # 调整布局
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"chart_{selected_color_id}_{selected_shape_id}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(OUTPUT_DIR / f"chart_{selected_color_id}_{selected_shape_id}..pdf"),
        format="pdf",
        bbox_inches="tight",
    )


# =========================================================================================
# ====================================== 6.调用绘图函数=========================================
# =========================================================================================
plot_interaction_chart(
    q_interaction_matrix,  # 交互作用矩阵
    q_individual_factors,  # 单因子数组
    labels,  # 标签列表
    color_id=selected_color_id,  # 颜色
    shape_id=selected_shape_id,  # 形状
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

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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ======================================2.颜色库设置=========================================
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
        ],
        "bar_color": "#d3c0a3",
    },
    2: {
        "ring_colors": [
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
        ],
        "bar_color": "#f781bf",
    },
    7: {
        "ring_colors": [
            "#b2182b",
            "#d6604d",
            "#f4a582",
            "#fddbc7",
            "#d1e5f0",
            "#92c5de",
            "#4393c3",
        ],
        "bar_color": "#878787",
    },
    8: {
        "ring_colors": [
            "#7b3294",
            "#c2a5cf",
            "#e7d4e8",
            "#f7f7f7",
            "#d9f0d3",
            "#a6dba0",
            "#5aae61",
        ],
        "bar_color": "#bababa",
    },
    9: {
        "ring_colors": [
            "#54bebe",
            "#76c8c8",
            "#98d1d1",
            "#badbdb",
            "#dedad2",
            "#e4bcad",
            "#df979e",
        ],
        "bar_color": "#c7c1a8",
    },
    10: {
        "ring_colors": [
            "#d73027",
            "#fc8d59",
            "#fee090",
            "#ffffbf",
            "#e0f3f8",
            "#91bfdb",
            "#4575b4",
        ],
        "bar_color": "#a1a1a1",
    },
    11: {
        "ring_colors": [
            "#f3e79b",
            "#fac484",
            "#f8a07e",
            "#eb7f86",
            "#ce6693",
            "#a75aa4",
            "#7454a2",
        ],
        "bar_color": "#66545e",
    },
    12: {
        "ring_colors": [
            "#00429d",
            "#4771b2",
            "#73a2c6",
            "#a6d3d8",
            "#ffffe0",
            "#ffbcaf",
            "#f4777f",
        ],
        "bar_color": "#b0a8b9",
    },
    13: {
        "ring_colors": [
            "#2d5e3c",
            "#488055",
            "#66a36f",
            "#86c88b",
            "#a8eea0",
            "#cbf4b5",
            "#edfbd0",
        ],
        "bar_color": "#705d3b",
    },
    14: {
        "ring_colors": [
            "#4b0082",
            "#8a2be2",
            "#9932cc",
            "#ba55d3",
            "#dda0dd",
            "#e6e6fa",
            "#f8f8ff",
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
        ],
        "bar_color": "#8d8d8d",
    },
    16: {
        "ring_colors": [
            "#440154",
            "#414487",
            "#2a788e",
            "#22a884",
            "#7ad151",
            "#fde725",
            "#c4e029",
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
        ],
        "bar_color": "#a9a9a9",
    },
    20: {
        "ring_colors": [
            "#d53e4f",
            "#f46d43",
            "#fdae61",
            "#fee08b",
            "#e6f598",
            "#abdda4",
            "#66c2a5",
        ],
        "bar_color": "#3288bd",
    },
}


# =========================================================================================
# ======================================3.绘图函数=========================================
# =========================================================================================
def plot_multi_omics_circos(
    gene_names,  # 基因名
    heatmap_data,  # 热图数据
    color_scheme_id=1,
):  # 配色方案

    # 获取图层数和基因数
    num_layers, num_genes = heatmap_data.shape
    print(num_layers, num_genes)

    # 创建画布
    fig, ax = plt.subplots(figsize=(16, 16), subplot_kw=dict(projection="polar"))

    selected_scheme = COLOR_LIBRARY[color_scheme_id]  # 获取选定的配色方案
    ring_colors = selected_scheme["ring_colors"]  # 提取环的颜色列表
    bar_color = selected_scheme["bar_color"]  # 提取外部条的颜色

    ring_thickness = 1.2  # 定义单个环的厚度
    radial_gap = 0.2  # 环与环之间的径向间隔
    # 中心空白区域的半径
    central_hole_radius = 7.0

    # 计算每个环的半径，环厚度+间隔
    step_with_gap = ring_thickness + radial_gap
    # 每个环的起始半径位置
    ring_positions = np.arange(
        central_hole_radius, central_hole_radius + num_layers * step_with_gap, step_with_gap
    )

    # 定义顶部开口的大小，使其相当于3个基因特征所占的宽度
    num_features_in_gap = 3
    # 计算圆环需要划分成多少个小区域
    total_positions = num_genes + num_features_in_gap
    # 每个小区域平均占用的角度（弧度）
    angular_width_per_position = 2 * np.pi / total_positions
    # 计算开口的总角度
    gap_size_rad = num_features_in_gap * angular_width_per_position
    # 环形图的起始角度
    start_angle = np.pi / 2 + gap_size_rad / 2
    # 环形图的结束角度
    end_angle = start_angle + (2 * np.pi - gap_size_rad - 0.02)
    # 生成每个小区域的角度坐标
    theta = np.linspace(start_angle, end_angle, num_genes)
    width = angular_width_per_position * 0.8  # 计算每个小区域的角度宽度

    # 绘制7层环状热图
    for layer_idx in range(num_layers):  # 遍历所有的数据层
        for gene_idx in range(num_genes):  # 遍历该层中的所有基因
            # 根据数值1或0确定色块的透明度
            alpha_value = 1.0 if heatmap_data[layer_idx, gene_idx] == 1 else 0.2
            # 绘制色块
            ax.bar(
                x=theta[gene_idx],  # 角度位置
                height=ring_thickness,  # 高度/厚度
                width=width,  # 宽度
                bottom=ring_positions[layer_idx],  # 起始半径
                color=ring_colors[layer_idx],  # 颜色
                alpha=alpha_value,  # 透明度
                align="edge",  # 从指定的theta角度开始绘制
            )

    # 绘制最外层的径向堆叠方块图
    block_height = 0.5  # 每个小方块的高度
    block_spacing = 0.05  # 小方块之间的垂直间隙

    outer_bar_bottom_radius = (
        ring_positions.max() + ring_thickness + 4
    )  # 计算最外圈堆叠图的起始半径
    # 遍历每一个基因
    for gene_idx in range(num_genes):  # 循环遍历每一个基因/特征
        # 计算该基因/特征在7个热图层中值为1（有影响）的总数
        num_blocks = int(
            np.sum(heatmap_data[:, gene_idx])
        )  # 对该基因/特征所在列的数据求和，得到值为1的总数
        # 如果总数为0，跳过，不绘制任何方块
        if num_blocks == 0:
            continue
        # 设置第一个方块的起始半径
        current_bottom = outer_bar_bottom_radius
        # 根据总数，循环绘制相应数量的堆叠方块
        for _ in range(num_blocks):
            # 绘制单个方块
            ax.bar(
                x=theta[gene_idx],  # 方块的角度
                height=block_height,  # 高度
                width=width * 0.7,  # 角度宽度
                bottom=current_bottom,  # 起始半径
                color=bar_color,  # 颜色
                align="edge",  # 从指定的theta角度开始绘制
            )
            # 更新下一个方块的起始半径位置，这样就可以实现堆叠的效果
            current_bottom += block_height + block_spacing

    # 外围基因/特征标注的半径
    label_radius = ring_positions.max() + ring_thickness + 0.5
    # 遍历每一个基因/特征添加标签
    for i in range(num_genes):
        angle_deg = np.rad2deg(theta[i]) % 360  # 将基因/特征的角度从弧度转换为度数
        if 90 < angle_deg < 270:
            rotation = angle_deg  # 文本的旋转角度
            ha = "left"  # 水平对齐方式
        else:
            rotation = angle_deg  # 文本的旋转角度
            ha = "left"  # 水平对齐方式
        # 添加基因/特征的名称标注
        ax.text(
            theta[i] + width / 2,  # 角度位置
            label_radius,  # 半径
            gene_names[i],  # 内容，基因/特征的名称
            rotation=rotation,  # 文本的旋转角度
            ha=ha,  # 水平对齐方式
            va="center",  # 垂直对齐方式
            fontsize=8,  # 大小
            rotation_mode="anchor",
        )

    # 绘制虚线圈，分隔开主图与外围的堆叠图的区域
    # 虚线圈的半径
    dashed_circle_radius = ring_positions.max() + ring_thickness + 3.8
    # 绘制带开口的虚线圈
    ax.plot(
        np.linspace(start_angle, end_angle, 200),
        [dashed_circle_radius] * 200,
        color=bar_color,
        linestyle="--",
        lw=1,
        zorder=2,
    )

    # 定义开口处灰色背景条的中心角度
    label_angle = np.pi / 2
    # 灰色背景的颜色
    gap_gray_color = "#f0f0f0"
    # 灰色背景的高度
    gray_bar_height = (ring_positions.max() + ring_thickness) - ring_positions.min()
    # 灰色背景的宽度
    gray_bar_width = 1.5 * angular_width_per_position

    # 绘制灰色背景条
    ax.bar(
        x=label_angle,  # 背景条的中心角度
        height=gray_bar_height,  # 高度
        width=gray_bar_width,  # 宽度
        bottom=ring_positions[0],  # 起始半径
        color=gap_gray_color,  # 颜色
        align="center",  # 居中对齐
        zorder=0,
        edgecolor="white",  # 边缘线颜色
        linewidth=1,  # 边缘线宽度
    )

    # 在灰色背景上添加数字标签
    for i in range(num_layers):  # 遍历每一层
        # 计算每个数字的半径位置
        r = ring_positions[i] + ring_thickness / 2
        # 添加数字
        ax.text(
            label_angle,
            r,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#555555",
            zorder=3,
        )

    # 图面清理
    ax.grid(False)  # 去掉自带网格线
    ax.set_yticklabels([])  # 去掉自带半径的刻度标签
    ax.set_xticklabels([])  # 去掉自带角度的刻度标签
    ax.spines["polar"].set_visible(False)  # 去掉自带外围黑圈
    max_stack_height = num_layers * (block_height + block_spacing)  # 计算最外层堆叠的最大高度
    max_radius = outer_bar_bottom_radius + max_stack_height  # 计算图表所需的最大半径
    ax.set_ylim(0, max_radius + 2)  # 设置半径的显示范围

    # 添加图例
    # 定义与图片匹配的图例标签
    layer_labels = [
        "1:Drug target",
        "2:DEG by microarray",
        "3:DEG by RNA-seq",
        "4:DEG in DAM",
        "5:DEG in DAA",
        "6:DEG by proteome",
        "7:Literature evidence",
    ]  # 列表定义结束

    # 创建图例用的色块
    ring_patches = [
        mpatches.Patch(color=ring_colors[i], label=layer_labels[i]) for i in range(num_layers)
    ]

    # 为最外圈的堆叠图创建图例
    bar_patch = mpatches.Patch(color=bar_color, label="Multi-omics evidence")

    # 将所有图例元素合并
    all_patches = ring_patches + [bar_patch]

    # 添加图例
    ax.legend(handles=all_patches, loc="center", frameon=False, fontsize=12)

    # 保存
    png_filename = str(OUTPUT_DIR / f"{select_color}.png")
    pdf_filename = str(OUTPUT_DIR / f"{select_color}.pdf")
    plt.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(pdf_filename, format="pdf", dpi=300, bbox_inches="tight")


# =========================================================================================
# ======================================4.程序执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    select_color = 20  # 要使用的配色方案编号
    # 文件路径
    excel_filename = str(DATA_DIR / "data.xlsx")

    # 读取
    df = pd.read_excel(excel_filename, index_col=0)

    # 提取基因/特征名列表
    gene_names_from_excel = df.index.tolist()

    # 提取热图数据，注意：绘图函数需要的数据形状是 (层数, 基因数)，而Pandas读取的DataFrame是 (基因数, 层数)，因此需要进行转置 (.T)
    heatmap_data_from_excel = df.values.T

    # 调用函数进行绘图
    plot_multi_omics_circos(
        gene_names=gene_names_from_excel,  # 传入基因名数据
        heatmap_data=heatmap_data_from_excel,  # 传入热图数据
        color_scheme_id=select_color,  # 配色方案
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

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
# 1、导入所需库
# =============================================================================
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =============================================================================
# 2.定义颜色库
# =============================================================================
color_schemes = {
    1: ["#440154", "#3b528b", "#21908d", "#5dc863"],
    2: ["#0d0887", "#7201a8", "#bd3786", "#f0744e"],
    3: ["#67001f", "#b2182b", "#4393c3", "#2166ac"],
    4: ["#8c510a", "#bf812d", "#80cdc1", "#35978f"],
    5: ["#2d004b", "#8073ac", "#fee0b6", "#fdb863"],
    6: ["#7f3b08", "#e08214", "#b2abd2", "#8073ac"],
    7: ["#004d40", "#00796b", "#4db6ac", "#b2dfdb"],
    8: ["#313695", "#74add1", "#fee090", "#fdae61"],
    9: ["#c51b7d", "#f1b6da", "#b8e186", "#7fbc41"],
    10: ["#5e4fa2", "#66c2a5", "#fee08b", "#f46d43"],
    11: ["#1a1a1a", "#666666", "#b3b3b3", "#f0f0f0"],
    12: ["#9e0142", "#d53e4f", "#fdae61", "#abdda4"],
    13: ["#8e0152", "#c51b7d", "#7fbc41", "#4d9221"],
    14: ["#003c30", "#35978f", "#c7eae5", "#f5f5f5"],
    15: ["#a6611a", "#dfc27d", "#80cdc1", "#018571"],
    16: ["#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"],
    17: ["#543005", "#bf812d", "#a1d76a", "#4d9221"],
    18: ["#b35806", "#fdb863", "#d8daeb", "#542788"],
    19: ["#00876c", "#aed987", "#fec981", "#d45d31"],
    20: ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"],
    21: ["#581845", "#008080", "#90EE90", "#FFD700"],
}


# =============================================================================
# 3.数据加载函数
# =============================================================================
def load_data_from_excel(filepath):
    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(filepath)
    # 提取第一列数据作为x 轴数据
    wavelength = df.iloc[:, 0].values
    # 循环提取从第二列开始的所有数据列，作为 y 轴数据集
    all_y_data = [df.iloc[:, i].values for i in range(1, len(df.columns))]
    # 提取从第二列开始的列名，作为每条曲线的标签
    labels = df.columns[1:].tolist()
    print(f"成功从 {filepath} 加载数据。")
    # 返回加载和解析好的数据
    return wavelength, all_y_data, labels


# =============================================================================
# 4.瀑布图绘图函数
# =============================================================================
def plot_3d_waterfall(wavelength, all_y_data, labels, colors, group_colors_for_legend):
    fig = plt.figure(figsize=(14, 10))
    # 在图形窗口中添加一个子图，并指定其为 3D 投影
    ax = fig.add_subplot(111, projection="3d")
    # 获取曲线的总数
    num_datasets = len(all_y_data)
    # 遍历所有的数据集，i 是索引，y_data 是对应的光谱数据
    for i, y_data in enumerate(all_y_data):
        # 为了创建封闭的填充区域，在波长数据的首尾各添加一个点
        padded_wl = np.r_[wavelength[0], wavelength, wavelength[-1]]
        # 对应地，在 y 轴数据的首尾各添加一个 0，使填充区域的底部落在基线上
        padded_yd = np.r_[0, y_data, 0]
        # 将 x (padded_wl) 和 y (padded_yd) 坐标配对，形成多边形的顶点
        verts = [list(zip(padded_wl, padded_yd))]
        # 创建一个多边形集合（即填充区域），设置其顶点、填充颜色和透明度
        poly = PolyCollection(verts, facecolors=colors[i], alpha=0.4)
        # 将创建的多边形添加到3D坐标系中。zs=i 指定其在x轴方向的深度位置，zdir='x' 表示深度方向是x轴
        ax.add_collection3d(poly, zs=i, zdir="x")
        # 在同样的位置绘制曲线的轮廓线，使其颜色与填充区域一致，并设置线宽
        ax.plot(wavelength, y_data, zs=i, zdir="x", color=colors[i], linewidth=2.0)

    # 设置 Y 轴标签、字体大小和与坐标轴的间距
    ax.set_ylabel("Wavelength (nm)", fontsize=14, labelpad=15)
    # 设置 Z 轴标签、字体大小和与坐标轴的间距
    ax.set_zlabel("Amplitude (mV)", fontsize=14, labelpad=10)
    # 不设置 X 轴标签
    ax.set_xlabel("")
    # 设置 Y 轴的范围
    ax.set_ylim(1700, 800)
    # 设置 X 轴的范围，从-1到数据集总数，留出一些边距
    ax.set_xlim(-1, num_datasets)
    # 设置 Z 轴的范围
    ax.set_zlim(0, 2600)

    # 设置 X 轴的刻度位置，每个整数位置对应一条曲线
    ax.set_xticks(range(num_datasets))
    # 将 X 轴的刻度标签设置为数据集的名称，并调整字体大小和旋转角度
    ax.set_xticklabels(labels, fontsize=12, rotation=0, ha="left")
    # 调整 X 轴刻度标签与轴线的距离
    ax.tick_params(axis="x", pad=-5)
    # 设置 Y 轴和 Z 轴刻度标签的字体大小
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)

    # 设置 3D 视图的仰角和方位角
    ax.view_init(elev=10, azim=-160)
    # 将 X, Y, Z 三个坐标平面的背景色设置为完全透明
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # 设置 X 坐标平面的边框线为黑色
    ax.xaxis.pane.set_edgecolor("k")
    # 将 Y 和 Z 坐标平面的边框线设置为透明
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # 调整 3D 图形在 X, Y, Z 方向上的视觉比例
    ax.set_box_aspect((35, 15, 12))
    # 关闭 matplotlib 默认的背景网格线
    ax.grid(False)

    # 隐藏默认的坐标轴刻度短线
    ax.tick_params(axis="x", length=0, color=(0, 0, 0, 0))
    ax.tick_params(axis="y", length=0, color=(0, 0, 0, 0))
    # 获取 Z, X, Y 轴的当前范围
    zmin, zmax = ax.get_zlim()
    xmin, xmax = ax.get_xlim()
    ymin_back, ymax_front = ax.get_ylim()
    # 获取 X, Y 轴的刻度位置
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    # 计算用于手动绘制的刻度线的高度
    tick_height = (zmax - zmin) * 0.025
    # 在 XZ 平面 (底部后方) 为每个 X 轴刻度绘制一条短的垂直刻度线
    for x_pos in xticks:
        ax.plot(
            [x_pos, x_pos],
            [ymin_back, ymin_back],
            [zmin, zmin + tick_height],
            color="k",
            linewidth=1,
        )
    # 在 YZ 平面 (底部左侧) 为每个 Y 轴刻度绘制一条短的垂直刻度线
    for y_pos in yticks:
        ax.plot([xmin, xmin], [y_pos, y_pos], [zmin, zmin + tick_height], color="k", linewidth=1)
    # 获取 Y 轴和 X 轴的范围，用于绘制网格线
    y_lim_bottom, y_lim_top = ax.get_ylim()
    x_lim_left, x_lim_right = ax.get_xlim()
    # 在 XY 平面 (底部) 绘制与 Y 轴平行的浅灰色网格线
    for x in xticks:
        if abs(x - x_lim_left) < 1e-6:
            continue
        ax.plot([x, x], ax.get_ylim(), zmin, color="lightgray", linestyle="--", linewidth=0.5)
    # 在 XY 平面 (底部) 绘制与 X 轴平行的浅灰色网格线
    for y in yticks:
        if abs(y - y_lim_bottom) < 1e-6:
            continue
        ax.plot(ax.get_xlim(), [y, y], zmin, color="lightgray", linestyle="--", linewidth=0.5)

    # 为每个分组创建一个颜色块 (Patch) 和对应的标签，用于图例显示
    legend_patches = [
        mpatches.Patch(color=color, label=group) for group, color in group_colors_for_legend.items()
    ]
    # 在图上显示图例，并设置其位置、字体大小等属性
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(0.83, 0.72),
        fontsize=12,
        borderaxespad=0.0,
    )


# =============================================================================
# 6、主程序入口
# =============================================================================

if __name__ == "__main__":
    # 数据的路径
    excel_file_path = str(DATA_DIR / "瀑布图/simulated_gaussian_data.xlsx")
    # 选择要使用的配色
    color_scheme_choice = 21
    # 定义四个分组的名称
    group_names = ["groupA", "groupB", "groupC", "groupD"]
    # 定义一个映射关系，指定7条曲线中的每一条分别属于哪个分组，例如，第一条曲线('series-1')属于'groupD'，第二条('series-2')属于'groupC'，以此类推
    series_to_group_map = ["groupD", "groupC", "groupD", "groupC", "groupA", "groupB", "groupA"]
    # 调用函数从指定的 Excel 文件加载数据
    wavelength_data, y_datasets, data_labels = load_data_from_excel(excel_file_path)
    # 检查数据是否成功加载
    if wavelength_data is not None and y_datasets is not None:
        # 从颜色库中获取选定的配色方案，如果选择的编号不存在，则使用默认的21号方案
        selected_palette = color_schemes.get(color_scheme_choice, color_schemes[21])
        print(f"--- 已选择配色方案: {color_scheme_choice} ---")
        # 检查所选配色方案的颜色数量是否足够分配给所有分组
        if len(selected_palette) < len(group_names):
            raise ValueError(f"配色方案 {color_scheme_choice} 的颜色数量不足4种。")
        # 创建一个字典，将分组名称与所选方案的颜色一一对应
        group_colors = dict(zip(group_names, selected_palette))
        # 根据映射关系，为7条曲线生成一个对应的颜色列表
        plot_colors = [group_colors[group] for group in series_to_group_map]
        # 使用加载的数据和生成的颜色列表来调用主绘图函数
        plot_3d_waterfall(wavelength_data, y_datasets, data_labels, plot_colors, group_colors)
        # 保存
        base_filename = str(OUTPUT_DIR / "three_dimensional_series_waterfall")
        png_filename = f"{base_filename}_{color_scheme_choice}.png"
        pdf_filename = f"{base_filename}_{color_scheme_choice}.pdf"
        plt.savefig(png_filename, dpi=300, bbox_inches="tight")
        plt.savefig(pdf_filename, bbox_inches="tight")
        print(f"图形已成功保存为: {png_filename} 和 {pdf_filename}")

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

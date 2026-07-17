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

import os

import matplotlib

# =========================================================================================
# ======================================1.库的导入=========================================
# =========================================================================================
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt  #
import pandas as pd
import seaborn as sns
from PIL import Image

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# =====================================2.颜色库设置=========================================
# =========================================================================================
COLOR_PALETTES = {
    1: "bwr",
    2: "coolwarm",
    3: "viridis",
    4: "plasma",
    5: "inferno",
    6: "magma",
    7: "cividis",
    8: "RdBu_r",
    9: "PiYG",
    10: "PRGn",
    11: "BrBG",
    12: "PuOr",
    13: "Spectral_r",
    14: "gist_earth",
    15: "terrain",
    16: "ocean",
    17: "gnuplot",
    18: "jet",
    19: "turbo",
    20: "seismic",
}


# =========================================================================================
# =====================================3.绘图函数=========================================
# =========================================================================================
def create_and_save_violin_plot(df_data, output_basename, title_text, palette_name):
    title_fontsize, label_fontsize, tick_fontsize = (
        20,
        20,
        20,
    )  # 定义标题、坐标轴标签和刻度标签的字体大小
    spine_width, tick_major_width, tick_major_length, tick_direction = (
        1.5,
        1.5,
        8,
        "in",
    )  # 定义图框线宽、主刻度线宽度、长度和方向

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(
        x="Category",
        y="NO conversion (%)",
        data=df_data,
        hue="Average",  # 设置x轴、y轴、数据源和颜色映射变量
        palette=palette_name,  # 选择的配色
        inner="box",
        ax=ax,
        dodge=False,  # 在小提琴内部显示箱线图
    )
    ax.set_title(title_text, fontsize=title_fontsize)  # 设置图表标题和字体大小
    ax.set_ylabel("NO conversion (%)", fontsize=label_fontsize)  # 设置y轴标签和字体大小
    ax.set_xlabel("")  # 设置x轴标签为空
    for spine in ax.spines.values():  # 遍历图表图框
        spine.set_linewidth(spine_width)  # 设置每个图框的线宽
    ax.tick_params(
        axis="x", rotation=90, labelsize=tick_fontsize
    )  # 设置x轴刻度标签旋转90度，并设置字体大小
    ax.tick_params(
        axis="both",
        direction=tick_direction,
        which="major",  # 应用于两个轴，刻度线朝内，只修改主刻度线
        width=tick_major_width,
        length=tick_major_length,
        labelsize=tick_fontsize,  # 设置刻度线的宽度、长度和刻度标签的字体大小
    )
    if ax.get_legend() is not None:  # 检查图表上是否存在图例
        ax.get_legend().remove()  # 如果存在，则移除

    norm = mcolors.Normalize(
        vmin=df_data["Average"].min(), vmax=df_data["Average"].max()
    )  # 创建一个归一化对象，将颜色映射到数据的最小值和最大值之间
    cmap = plt.get_cmap(palette_name)  # 从matplotlib获取与seaborn中使用的同名颜色映射
    sm = cm.ScalarMappable(
        cmap=cmap, norm=norm
    )  # 创建一个可映射标量的对象，它结合了颜色映射和归一化规则
    sm.set_array([])  # 设置一个空数组，这是创建独立颜色条的常规做法
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)  # 在图表旁边创建一个颜色条，并设置其与图表的间距
    cbar.set_label(
        "Average NO conversion rate of different combinations (%)",
        fontsize=label_fontsize,
        rotation=270,  # 设置颜色条的标签
        labelpad=25,
    )  # 设置标签的字体大小、旋转角度和与颜色条的距离
    cbar.ax.tick_params(labelsize=tick_fontsize)  # 设置颜色条刻度标签的字体大小
    cbar.outline.set_linewidth(spine_width)  # 设置颜色条边框的线宽
    plt.tight_layout()  # 自动调整图表布局，防止标签重叠
    # 保存结果
    png_filename = f"{output_basename}.png"
    pdf_filename = f"{output_basename}.pdf"
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# =====================================4.主程序：输入文件、选择配色、绘图=========================================
# =========================================================================================
if __name__ == "__main__":
    selected_palette_key = 3  # 选择配色
    selected_palette_name = COLOR_PALETTES.get(selected_palette_key, "viridis")
    output_folder = str(OUTPUT_DIR)  # 设置所有输出图片将要保存的文件夹路径
    os.makedirs(output_folder, exist_ok=True)  # 如果没有就创建
    # 需要进行绘图的所有数据的路径
    input_files = [
        str(DATA_DIR / "violin_plot_group_1_data.xlsx"),
        str(DATA_DIR / "violin_plot_group_2_data.xlsx"),
        str(DATA_DIR / "violin_plot_group_3_data.xlsx"),
        str(DATA_DIR / "violin_plot_group_4_data.xlsx"),
        str(DATA_DIR / "violin_plot_group_3_data.xlsx"),
        str(DATA_DIR / "violin_plot_group_4_data.xlsx"),
    ]
    saved_plot_paths = []  # 初始化一个空列表存储子图用于后面组合图使用

    # 循环处理每个Excel文件
    for filepath in input_files:  # 遍历文件路径列表中的每一个文件
        filename = os.path.basename(filepath)  # 从完整路径中提取文件名
        df_raw = pd.read_excel(filepath)  # 读取数据
        df_plot = df_raw.copy()  # 创建一个原始数据的副本进行操作，以防修改原始数据
        df_plot["Average"] = df_plot.groupby("Category")["NO conversion (%)"].transform(
            "mean"
        )  # 计算平均值
        base_name = os.path.splitext(filename)[0]  # 获取文件名
        output_file_base = os.path.join(
            output_folder, f"{base_name}_{selected_palette_key}plot"
        )  # 构建输出路径
        plot_title = f"Weighted - {base_name}"  # 标题
        create_and_save_violin_plot(
            df_plot, output_file_base, plot_title, selected_palette_name
        )  # 调用绘图函数，并传入数据、输出名、标题和选定的颜色方案
        saved_plot_paths.append(
            f"{output_file_base}.png"
        )  # 将刚刚保存的PNG图片的完整路径添加到列表中，以备后续拼接
    # =========================================================================================
    # =====================================5.拼接组合图=========================================
    # =========================================================================================
    if saved_plot_paths:  # 检查是否绘制了子图
        print(f"\n{'=' * 20} 所有单图处理完成，开始拼接组合图 {'=' * 20}")
        composite_rows = 3  # 行数
        composite_cols = 2  # 列数
        num_images = len(saved_plot_paths)  # 获取要拼接的图片总数
        if num_images > composite_rows * composite_cols:  # 检查预设的网格是否足够容纳所有图片
            print(
                f" > 警告：图片数量 ({num_images}) 大于网格容量 ({composite_rows}x{composite_cols}={composite_rows * composite_cols})。"
            )
        else:  # 如果网格大小足够
            images_to_composite = [
                Image.open(path) for path in saved_plot_paths
            ]  # 打开所有已保存的PNG图片，并创建Image对象列表
            width, height = images_to_composite[
                0
            ].size  # 获取第一张图片的宽度和高度，假设所有图片尺寸相同
            composite_image = Image.new(
                "RGB", (width * composite_cols, height * composite_rows), color="white"
            )  # 创建一张新的白色背景大图，尺寸足以容纳所有小图
            for i, img in enumerate(images_to_composite):  # 遍历每一张要粘贴的小图
                row = i // composite_cols  # 计算当前图片应该在的行号（从0开始）
                col = i % composite_cols  # 计算当前图片应该在的列号（从0开始）
                x_offset = col * width  # 计算粘贴位置的x坐标（左上角）
                y_offset = row * height  # 计算粘贴位置的y坐标（左上角）
                composite_image.paste(img, (x_offset, y_offset))  # 将小图粘贴到大图的指定位置
                img.close()
            composite_filename_base = "violin_plots_composite"  # 组合图的文件名
            png_path_composite = os.path.join(output_folder, f"{composite_filename_base}.png")
            composite_image.save(png_path_composite)
            print(f"组合图已保存为: '{png_path_composite}'")

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

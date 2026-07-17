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

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


# =======================================================================
# 1.图像拼接函数
# =======================================================================
def stitch_images_grid(image_paths, n_cols, output_dir, output_filename_base):
    images = [Image.open(path) for path in image_paths]  # 使用列表推导式打开所有指定路径的图片
    img_width, img_height = images[0].size  # 获取第一张图片的宽度和高度
    n_images = len(images)  # 获取图片的总数量
    n_rows = math.ceil(n_images / n_cols)  # 计算拼接后网格图需要的行数，使用math.ceil向上取整
    total_width = n_cols * img_width  # 计算拼接后总图像的宽度
    total_height = n_rows * img_height  # 计算拼接后总图像的高度
    composite_image = Image.new(
        "RGB", (total_width, total_height), color="white"
    )  # 创建一个新的空白RGB图像，背景为白色，用于粘贴所有小图
    for i, img in enumerate(images):  # 遍历所有已打开的图片及其索引
        row = i // n_cols  # 计算当前图片应该被放置在哪一行
        col = i % n_cols  # 计算当前图片应该被放置在哪一列
        paste_x = col * img_width  # 计算粘贴位置的x坐标
        paste_y = row * img_height  # 计算粘贴位置的y坐标
        composite_image.paste(img, (paste_x, paste_y))  # 将当前图片粘贴到新创建的空白图像的指定位置
        img.close()
    png_path_composite = os.path.join(output_dir, f"{output_filename_base}.png")
    composite_image.save(png_path_composite)
    pdf_path_composite = os.path.join(output_dir, f"{output_filename_base}.pdf")
    if composite_image.mode != "RGB":
        composite_image = composite_image.convert("RGB")  # 如果图像模式不是RGB，则将其转换为RGB模式
    composite_image.save(pdf_path_composite, "PDF", resolution=300.0)


# =======================================================================
# 2.箱型图绘图函数
# =======================================================================
# 定义一个函数，用于创建一个带特定主题风格的箱型图
def create_themed_boxplot(df: pd.DataFrame, title: str, title_color: str, sub_label: str):
    from matplotlib import rcParams

    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#EAEAF2")  # 设置背景颜色
    # 定义每个类别指定颜色
    custom_palette = {
        "A1": "#8f9a27",
        "A2": "#d4ca69",
        "A3": "#462d63",
        "A4": "#8c4b8f",
        "A5": "#75b3d0",
        "A6": "#c89cc4",
        "A7": "#808080",
    }
    # 使用seaborn绘制箱型图
    sns.boxplot(
        x="Category",
        y="Value",
        data=df,
        ax=ax,
        palette=custom_palette,
        showmeans=True,  # x轴数据列，y轴数据列，数据源，绘图坐标轴，调色板，显示均值
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "10",
        },  # 设置均值点的样式
        width=0.6,
        linewidth=1.5,
        hue="Category",
        legend=False,  # 箱体宽度，线条宽度，按'Category'分类着色，不显示图例
    )
    ax.axhline(0, color="black", linewidth=1.5, zorder=3)  # 在y=0的位置绘制一条黑色水平线
    ax.grid(
        True, which="both", color="white", linestyle="-", linewidth=1.2
    )  # 在图表中显示网格线，设置颜色、样式和线宽
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(
        axis="y", which="major", length=7, width=1.5, labelsize=12
    )  # 设置y轴主刻度线的样式
    ax.tick_params(
        axis="x", which="major", length=7, width=1.5, labelsize=16
    )  # 设置x轴主刻度线的样式
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax_pos = ax.get_position()  # 获取坐标轴在图窗中的位置和大小
    # 在图窗的指定位置添加一个矩形块，作为自定义标题的背景
    fig.patches.extend(
        [
            plt.Rectangle(
                (ax_pos.x0, 0.92),
                ax_pos.width,
                0.08,
                facecolor="#f5eecf",
                alpha=1,
                zorder=-1,
                transform=fig.transFigure,
                figure=fig,
            )
        ]
    )
    # 添加标题
    fig.text(
        0.5,
        0.96,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        fontname="Times New Roman",
        color=title_color,
    )
    # 添加子标题
    ax.text(
        0.07,
        1.06,
        sub_label,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="right",
        fontname="Times New Roman",
    )
    # 调整子图布局，为顶部的自定义标题留出空间
    fig.subplots_adjust(top=0.92)
    return fig


# =======================================================================
# 主程序入口
# =======================================================================
if __name__ == "__main__":
    # 定义一个列表，其中包含要处理的Excel文件路径和对应的图表标题
    input_data_and_titles = [
        (str(DATA_DIR / "dataset_1.xlsx"), "Soil Moisture Content"),
        (str(DATA_DIR / "dataset_2.xlsx"), "River Discharge Rate"),
        (str(DATA_DIR / "dataset_3.xlsx"), "Atmospheric CO2 Levels"),
        (str(DATA_DIR / "dataset_4.xlsx"), "Glacier Melting Speed"),
        (str(DATA_DIR / "dataset_5.xlsx"), "Urban Heat Island Effect"),
        (str(DATA_DIR / "dataset_6.xlsx"), "Ocean Acidity Index"),
        (str(DATA_DIR / "dataset_7.xlsx"), "Soil Moisture Content"),
        (str(DATA_DIR / "dataset_8.xlsx"), "River Discharge Rate"),
        (str(DATA_DIR / "dataset_9.xlsx"), "Atmospheric CO2 Levels"),
        (str(DATA_DIR / "dataset_10.xlsx"), "Glacier Melting Speed"),
        (str(DATA_DIR / "dataset_11.xlsx"), "Urban Heat Island Effect"),
        (str(DATA_DIR / "dataset_12.xlsx"), "Ocean Acidity Index"),
    ]
    # 指定所有输出文件保存的目录
    output_dir = str(OUTPUT_DIR)
    # 设置最终拼接图的列数
    N = 3
    # 设置最终拼接图的基础文件名
    COMPOSITE_FILENAME_BASE = "Composite_Analysis"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 定义一个标题颜色的列表，用于循环给每个图表标题分配颜色
    title_colors = [
        "#d62728",
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#5254a3",
        "#6b6ecf",
    ]
    # 初始化一个空列表，用于存储生成的所有单个PNG图片的路径
    generated_png_paths = []
    # 遍历输入数据列表，同时获取索引和元组中的内容（文件路径，图表标题）
    for i, (file_path, plot_title) in enumerate(input_data_and_titles):
        # 读取数据
        dataframe = pd.read_excel(file_path)
        # 从文件路径中提取不带扩展名的基本文件名
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        # 根据当前循环的索引从颜色列表中选择一个颜色
        plot_color = title_colors[i % len(title_colors)]
        # 生成子标题
        plot_label = f"({chr(ord('a') + i)})"
        print(f"\n({i + 1}/{len(input_data_and_titles)}) 正在处理: {file_path}")
        # 调用箱型图绘图函数
        figure = create_themed_boxplot(dataframe, plot_title, plot_color, plot_label)
        png_filename = os.path.join(output_dir, f"{base_filename}.png")
        pdf_filename = os.path.join(output_dir, f"{base_filename}.pdf")
        figure.savefig(png_filename, dpi=300, bbox_inches="tight")
        figure.savefig(pdf_filename, bbox_inches="tight")
        # 将生成的PNG文件路径添加到列表中，以备后续拼接使用
        generated_png_paths.append(png_filename)
        plt.close(figure)
    # 调用图像拼接函数
    stitch_images_grid(generated_png_paths, N, output_dir, COMPOSITE_FILENAME_BASE)
    print(f"\n全部处理完成！所有文件已在 '{output_dir}' 文件夹中。")

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

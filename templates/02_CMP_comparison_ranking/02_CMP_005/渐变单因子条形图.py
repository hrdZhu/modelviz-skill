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
# ======================================1.库的导入=========================================
# =========================================================================================
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
# =========================================================================================
# ======================================2.颜色库设置=========================================
# =========================================================================================
SELECTED_SCHEME_KEY = "1"  # 选择配色
# 颜色库
COLOR_SCHEMES = {
    "1": {"colors": ["#66CDAA", "#20B2AA", "#008B8B", "#4682B4", "#000080", "#191970"]},
    "2": {"colors": ["#FFD700", "#FFA500", "#FF8C00", "#FF7F50", "#FF6347", "#FF4500"]},
    "3": {"colors": ["#ADFF2F", "#7FFF00", "#7CFC00", "#32CD32", "#228B22", "#006400"]},
    "4": {"colors": ["#E6E6FA", "#D8BFD8", "#DDA0DD", "#DA70D6", "#BA55D3", "#9932CC"]},
    "5": {"colors": ["#FF7F50", "#FF6B6B", "#FF4757", "#F7418F", "#E63946", "#D63031"]},
    "6": {"colors": ["#00BFFF", "#1E90FF", "#4169E1", "#0000FF", "#0000CD", "#00008B"]},
    "7": {"colors": ["#F4A460", "#D2B48C", "#BC8F8F", "#CD853F", "#A0522D", "#8B4513"]},
    "8": {"colors": ["#DCDCDC", "#C0C0C0", "#A9A9A9", "#808080", "#696969", "#2F4F4F"]},
    "9": {"colors": ["#F5F5DC", "#FFEBCD", "#F5DEB3", "#DEB887", "#D2B48C", "#8B4513"]},
    "10": {"colors": ["#FFB6C1", "#FFC0CB", "#FFDAB9", "#E6E6FA", "#B0E0E6", "#ADD8E6"]},
    "11": {"colors": ["#FFFF00", "#FFFACD", "#FAFAD2", "#FFEFD5", "#FFE4B5", "#FFDAB9"]},
    "12": {"colors": ["#2E8B57", "#556B2F", "#6B8E23", "#8FBC8F", "#20B2AA", "#008080"]},
    "13": {"colors": ["#FFC0CB", "#FFB6C1", "#DB7093", "#C71585", "#D8BFD8", "#E6E6FA"]},
    "14": {"colors": ["#FF4500", "#FF6347", "#FF7F50", "#FF8C00", "#FFA500", "#FFD700"]},
    "15": {"colors": ["#B0C4DE", "#778899", "#708090", "#4682B4", "#5F9EA0", "#6495ED"]},
    "16": {"colors": ["#DC143C", "#B22222", "#A52A2A", "#8B0000", "#CD5C5C", "#E9967A"]},
    "17": {"colors": ["#FFFACD", "#F0E68C", "#FFD700", "#DAA520", "#B8860B", "#F4A460"]},
    "18": {"colors": ["#F5F5F5", "#E0FFFF", "#AFEEEE", "#7FFFD4", "#40E0D0", "#48D1CC"]},
    "19": {"colors": ["#2C3E50", "#34495E", "#7F8C8D", "#95A5A6", "#BDC3C7", "#ECF0F1"]},
    "20": {"colors": ["#DC143C", "#C71585", "#9932CC", "#8A2BE2", "#BA55D3", "#DA70D6"]},
}


# =========================================================================================
# ======================================3.单因子条形图绘制函数=========================================
# =========================================================================================
def create_and_save_plot(data, plot_title, output_filename_base, scheme):
    print(f"--- 正在绘制图表: {plot_title} ---")
    df = pd.DataFrame(data)  # 将传入的字典格式数据转换为dataFrame格式
    df = df.sort_values(by="q Value", ascending=False).reset_index(
        drop=True
    )  # 按照指定列的值进行降序排序，并重置索引
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_gradient", scheme["colors"]
    )  # 根据传入的颜色列表创建一个自定义的渐变颜色映射
    bar_colors = custom_cmap(
        np.linspace(0, 1, len(df))
    )  # 使用颜色映射生成一组渐变颜色，数量与数据行数相同
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("#F5F5F5")  # 设置坐标轴区域的背景颜色

    bars = ax.barh(
        df["Variables"], df["q Value"], color=bar_colors, edgecolor="black", linewidth=1.5, zorder=2
    )  # 绘制水平条形图，并设置颜色、边框等样式
    text_outline = [
        path_effects.withStroke(linewidth=1.5, foreground="black")
    ]  # 创建一个文本描边效果
    for bar in bars:  # 遍历图中的每一个条形对象
        width = bar.get_width()  # 获取当前条形的宽度
        if width < 0.01:  # 判断条形宽度是否非常小
            label_x_pos = width + 0.002  # 如果条形很短，则将数值标签放在条形外部右侧
            ha = "left"  # 设置文本水平对齐方式为左对齐
        else:
            label_x_pos = width - 0.001  # 将数值标签放在条形内部靠近右端的位置
            ha = "right"  # 设置文本水平对齐方式为右对齐
        # 添加条形图的文字标注
        ax.text(
            label_x_pos,
            bar.get_y() + bar.get_height() / 2,  # 位置
            f"{width:.2f}",  # 两位小数
            ha=ha,
            va="center",  # 水平和垂直对齐方式
            color="white",
            fontsize=12,
            fontweight="bold",  # 颜色、字号和粗细
            path_effects=text_outline,
        )  # 应用描边效果
    # 设置显著性标记
    for i, row in df.iterrows():  # 遍历DataFrame的每一行，i为索引，row为该行数据
        if row["p_significant"]:  # 检查该行的'p_significant'值是否为True
            star_x_pos = row["q Value"] + 0.005  # 计算星号标记的x轴位置，放在条形右侧
            ax.text(
                star_x_pos,
                i,
                r"$\star$",  # 在计算好的位置添加一个*
                ha="center",
                va="center",  # 设置*的对齐方式
                color="black",
                fontsize=20,
                fontweight="bold",
            )  # 设置*的颜色、大小和粗细
    ax.invert_yaxis()  # 反转y轴，使得q值最大的条形显示在最上方
    ax.set_title(plot_title, fontsize=20, pad=15)  # 设置标题
    ax.set_xlabel("q Value", fontsize=16)  # 设置x轴的标签
    ax.set_ylabel("Variables", fontsize=16)  # 设置y轴的标签
    ax.tick_params(axis="y", labelsize=14)  # 设置y轴刻度标签的字体大小
    ax.tick_params(axis="x", labelsize=12)  # 设置x轴刻度标签的字体大小
    ax.set_xticks([0.0, 0.1, 0.2])  # 手动设置x轴上要显示的刻度位置
    ax.set_xlim(right=0.23)  # 设置x轴的显示范围上限
    ax.grid(
        axis="both", linestyle="-", color="white", linewidth=2, zorder=1
    )  # 在图表上绘制白色的网格线，并将其置于条形图之下
    ax.tick_params(
        axis="both", which="major", length=0
    )  # 将x轴和y轴的主要刻度线长度设置为0，即隐藏刻度线
    # 隐藏图框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    star_patch = Line2D(
        [0],
        [0],
        marker=r"$\star$",
        color="none",
        label="p < 0.05",  # 创建一个自定义的图例项，用于*
        markerfacecolor="black",
        markersize=15,
        linestyle="None",
    )  # 设置图例项的样式
    legend = ax.legend(
        handles=[star_patch],  # 在图表上显示图例
        loc="lower left",  # 设置图例的位置在左下角
        bbox_to_anchor=(-0.02, -0.08),  # 图例的精确位置
        fontsize=14,  # 字体大小
        frameon=True,  # 边框
        facecolor="#F0F0F8",  # 背景颜色
        edgecolor="none",  # 边框颜色
    )
    for text in legend.get_texts():  # 遍历图例中的所有文本对象
        text.set_fontweight("bold")  # 将图例文本设置为粗体
    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )  # 自动调整子图参数，使其填充整个图像区域，并为标题留出空间
    # 保存
    output_folder = str(OUTPUT_DIR)
    os.makedirs(output_folder, exist_ok=True)
    png_path = os.path.join(output_folder, f"{output_filename_base}_{SELECTED_SCHEME_KEY}.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.join(output_folder, f"{output_filename_base}_{SELECTED_SCHEME_KEY}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


# =========================================================================================
# ======================================4.拼接函数=========================================
# =========================================================================================
def stitch_images_grid(image_paths, n_cols, output_filename_base):
    images = [
        Image.open(path) for path in image_paths
    ]  # 使用列表推导式，依次打开所有路径对应的图片文件
    img_width, img_height = images[0].size  # 获取第一张图片的宽度和高度，作为所有图片拼接的基准尺寸
    n_images = len(images)  # 获取图片的总数量
    n_rows = (n_images + n_cols - 1) // n_cols  # 行数
    total_width = n_cols * img_width  # 计算拼接后画布的总宽度
    total_height = n_rows * img_height  # 计算拼接后画布的总高度
    composite_image = Image.new(
        "RGB", (total_width, total_height), color="white"
    )  # 创建一个新的白色背景的RGB图像作为画布
    for i, img in enumerate(images):  # 遍历所有图片及其索引
        row = i // n_cols  # 计算当前图片应该在的行号
        col = i % n_cols  # 计算当前图片应该在的列号
        paste_x = col * img_width  # 计算粘贴位置的左上角x坐标
        paste_y = row * img_height  # 计算粘贴位置的左上角y坐标
        composite_image.paste(img, (paste_x, paste_y))  # 将当前图片粘贴到画布的计算好的位置
        img.close()
    # 定义保存路径和文件名
    output_folder = os.path.dirname(image_paths[0])  # 使用第一张图片的目录作为拼接后图片的保存目录
    # 保存
    png_path_composite = os.path.join(output_folder, f"{output_filename_base}.png")
    composite_image.save(png_path_composite)
    pdf_path_composite = os.path.join(output_folder, f"{output_filename_base}.pdf")
    composite_image.save(pdf_path_composite)


# =========================================================================================
# ======================================5.库的导入=========================================
# =========================================================================================
if __name__ == "__main__":
    # 绘图数据，变量、数值、显著性
    all_plot_data = [
        {
            "Variables": [
                "TEMP",
                "PR",
                "AH",
                "ISO2",
                "AWS",
                "IOAD",
                "ISD",
                "IEC",
                "INOX",
                "SIVA",
                "GRP",
                "UHEC",
                "PD",
                "TIVA",
                "PC",
                "PIVA",
                "GCR",
                "PCGRP",
            ],
            "q Value": [
                0.18,
                0.09,
                0.09,
                0.08,
                0.07,
                0.06,
                0.06,
                0.05,
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.02,
                0.002,
                0.002,
                0.001,
                0.001,
            ],
            "p_significant": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
        },
        {
            "Variables": [
                "TEMP",
                "PR",
                "AH",
                "ISO2",
                "AWS",
                "IOAD",
                "ISD",
                "IEC",
                "INOX",
                "SIVA",
                "GRP",
                "UHEC",
                "PD",
                "TIVA",
                "PC",
                "PIVA",
                "GCR",
                "PCGRP",
            ],
            "q Value": [
                0.22,
                0.15,
                0.12,
                0.11,
                0.09,
                0.08,
                0.07,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.01,
                0.01,
                0.005,
                0.003,
                0.001,
            ],
            "p_significant": [
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
                False,
            ],
        },
        {
            "Variables": [
                "TEMP",
                "PR",
                "AH",
                "ISO2",
                "AWS",
                "IOAD",
                "ISD",
                "IEC",
                "INOX",
                "SIVA",
                "GRP",
                "UHEC",
                "PD",
                "TIVA",
                "PC",
                "PIVA",
                "GCR",
                "PCGRP",
            ],
            "q Value": [
                0.12,
                0.11,
                0.10,
                0.09,
                0.09,
                0.08,
                0.08,
                0.07,
                0.06,
                0.05,
                0.05,
                0.04,
                0.03,
                0.03,
                0.02,
                0.01,
                0.01,
                0.008,
            ],
            "p_significant": [
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
            ],
        },
        {
            "Variables": [
                "TEMP",
                "PR",
                "AH",
                "ISO2",
                "AWS",
                "IOAD",
                "ISD",
                "IEC",
                "INOX",
                "SIVA",
                "GRP",
                "UHEC",
                "PD",
                "TIVA",
                "PC",
                "PIVA",
                "GCR",
                "PCGRP",
            ],
            "q Value": [
                0.08,
                0.07,
                0.06,
                0.05,
                0.05,
                0.04,
                0.04,
                0.04,
                0.03,
                0.03,
                0.02,
                0.02,
                0.02,
                0.01,
                0.01,
                0.01,
                0.005,
                0.002,
            ],
            "p_significant": [
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
            ],
        },
        {
            "Variables": [
                "TEMP",
                "PR",
                "AH",
                "ISO2",
                "AWS",
                "IOAD",
                "ISD",
                "IEC",
                "INOX",
                "SIVA",
                "GRP",
                "UHEC",
                "PD",
                "TIVA",
                "PC",
                "PIVA",
                "GCR",
                "PCGRP",
            ],
            "q Value": [
                0.19,
                0.18,
                0.15,
                0.10,
                0.08,
                0.07,
                0.07,
                0.06,
                0.05,
                0.03,
                0.03,
                0.03,
                0.02,
                0.02,
                0.01,
                0.01,
                0.009,
                0.004,
            ],
            "p_significant": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
        },
    ]
    # 标题、文件名
    plot_configs = [
        {"title": "(a) National", "filename": "national"},
        {"title": "(b) Regional Group A", "filename": "regional_a"},
        {"title": "(c) Regional Group B", "filename": "regional_b"},
        {"title": "(d) Urban Analysis", "filename": "urban"},
        {"title": "(e) Rural Analysis", "filename": "rural"},
    ]

    selected_scheme = COLOR_SCHEMES.get(SELECTED_SCHEME_KEY, COLOR_SCHEMES["1"])  # 选择配色

    # 循环绘图
    saved_png_paths = []  # 初始化一个空列表，用于存储每个成功生成的图片的路径
    for i in range(5):
        config = plot_configs[i]  # 获取当前循环的图表配置
        current_data = all_plot_data[i]  # 获取当前循环的图表数据
        png_path = create_and_save_plot(
            data=current_data,  # 调用绘图函数
            plot_title=config["title"],  # 传入标题
            output_filename_base=config["filename"],  # 传入文件名基础部分
            scheme=selected_scheme,
        )  # 传入选择的配色方案
        saved_png_paths.append(png_path)  # 将保存的图片路径追加到列表中

    STITCH_COLS = 5  # 组合图的列数
    stitch_images_grid(
        image_paths=saved_png_paths,  # 调用拼接函数
        n_cols=STITCH_COLS,  # 列
        output_filename_base="final_composite_plot",
    )  # 文件名

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

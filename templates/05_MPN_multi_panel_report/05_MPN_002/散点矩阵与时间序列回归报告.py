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
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy import stats

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# =============================================================================
# ===== 2.颜色库 ==============================================================
# =============================================================================
color_schemes = {
    1: {
        "AAC": "#FF0000",
        "PC": "#0000FF",
        "LC": "#00FF00",
        "TV": "#FFFF00",
        "scatter": "#483D8B",
        "regline": "#FF0000",
        "sig_bg": "#FFDDDD",
        "ns_bg": "#F0F8FF",
    },
    2: {
        "AAC": "#FF1493",
        "PC": "#00BFFF",
        "LC": "#32CD32",
        "TV": "#FFD700",
        "scatter": "#2F4F4F",
        "regline": "#FF1493",
        "sig_bg": "#FFDFEE",
        "ns_bg": "#FFF0F5",
    },
    3: {
        "AAC": "#FF4500",
        "PC": "#4682B4",
        "LC": "#2E8B57",
        "TV": "#DAA520",
        "scatter": "#556B2F",
        "regline": "#FF4500",
        "sig_bg": "#FFD3C4",
        "ns_bg": "#F5F5DC",
    },
    4: {
        "AAC": "#FF00FF",
        "PC": "#00FFFF",
        "LC": "#F0E68C",
        "TV": "#8A2BE2",
        "scatter": "#4B0082",
        "regline": "#FF00FF",
        "sig_bg": "#FFDFFF",
        "ns_bg": "#F5FFFA",
    },
    5: {
        "AAC": "#DC143C",
        "PC": "#1E90FF",
        "LC": "#3CB371",
        "TV": "#FFD700",
        "scatter": "#2F4F4F",
        "regline": "#DC143C",
        "sig_bg": "#FFC9D0",
        "ns_bg": "#F0FFFF",
    },
    6: {
        "AAC": "#FF6347",
        "PC": "#4169E1",
        "LC": "#32CD32",
        "TV": "#FF8C00",
        "scatter": "#483D8B",
        "regline": "#FF6347",
        "sig_bg": "#FFDBD3",
        "ns_bg": "#F0F8FF",
    },
    7: {
        "AAC": "#FF007F",
        "PC": "#00A5E3",
        "LC": "#8DD54F",
        "TV": "#FF6F00",
        "scatter": "#008080",
        "regline": "#FF007F",
        "sig_bg": "#FFCCE8",
        "ns_bg": "#E0FFFF",
    },
    8: {
        "AAC": "#B22222",
        "PC": "#1E90FF",
        "LC": "#228B22",
        "TV": "#FFD700",
        "scatter": "#000080",
        "regline": "#B22222",
        "sig_bg": "#F5CCCC",
        "ns_bg": "#F0FFF0",
    },
    9: {
        "AAC": "#FF8C00",
        "PC": "#00CED1",
        "LC": "#ADFF2F",
        "TV": "#BA55D3",
        "scatter": "#800080",
        "regline": "#FF8C00",
        "sig_bg": "#FFE6CC",
        "ns_bg": "#F0FFFF",
    },
    10: {
        "AAC": "#F03232",
        "PC": "#325AF0",
        "LC": "#32F05A",
        "TV": "#F0E732",
        "scatter": "#00008B",
        "regline": "#F03232",
        "sig_bg": "#FADADD",
        "ns_bg": "#F5F5F5",
    },
    11: {
        "AAC": "#FF4081",
        "PC": "#2196F3",
        "LC": "#8BC34A",
        "TV": "#FFC107",
        "scatter": "#0D47A1",
        "regline": "#FF4081",
        "sig_bg": "#FFD1E6",
        "ns_bg": "#E3F2FD",
    },
    12: {
        "AAC": "#E91E63",
        "PC": "#00BCD4",
        "LC": "#4CAF50",
        "TV": "#FFEB3B",
        "scatter": "#006064",
        "regline": "#E91E63",
        "sig_bg": "#FBD3DF",
        "ns_bg": "#E0F7FA",
    },
    13: {
        "AAC": "#FF5722",
        "PC": "#3F51B5",
        "LC": "#8BC34A",
        "TV": "#FFC107",
        "scatter": "#1A237E",
        "regline": "#FF5722",
        "sig_bg": "#FFDBCA",
        "ns_bg": "#E8EAF6",
    },
    14: {
        "AAC": "#FF9800",
        "PC": "#03A9F4",
        "LC": "#8BC34A",
        "TV": "#FFEB3B",
        "scatter": "#01579B",
        "regline": "#FF9800",
        "sig_bg": "#FFE8CC",
        "ns_bg": "#E1F5FE",
    },
    15: {
        "AAC": "#E040FB",
        "PC": "#18FFFF",
        "LC": "#00E676",
        "TV": "#FFEA00",
        "scatter": "#004D40",
        "regline": "#E040FB",
        "sig_bg": "#F8D1FF",
        "ns_bg": "#E0F2F1",
    },
    16: {
        "AAC": "#6200EA",
        "PC": "#00BFA5",
        "LC": "#AEEA00",
        "TV": "#FFC400",
        "scatter": "#004D40",
        "regline": "#6200EA",
        "sig_bg": "#DDD1FA",
        "ns_bg": "#E6FFFA",
    },
    17: {
        "AAC": "#FF5252",
        "PC": "#448AFF",
        "LC": "#69F0AE",
        "TV": "#FFEB3B",
        "scatter": "#1565C0",
        "regline": "#FF5252",
        "sig_bg": "#FFDADA",
        "ns_bg": "#E3F2FD",
    },
    18: {
        "AAC": "#FFAB40",
        "PC": "#26C6DA",
        "LC": "#7C4DFF",
        "TV": "#CDDC39",
        "scatter": "#311B92",
        "regline": "#FFAB40",
        "sig_bg": "#FFEECC",
        "ns_bg": "#EDE7F6",
    },
    19: {
        "AAC": "#D500F9",
        "PC": "#00B8D4",
        "LC": "#AEEA00",
        "TV": "#FFEA00",
        "scatter": "#006064",
        "regline": "#D500F9",
        "sig_bg": "#F5C4FF",
        "ns_bg": "#E0F7FA",
    },
    20: {
        "AAC": "#FF1744",
        "PC": "#2962FF",
        "LC": "#00E676",
        "TV": "#FFEA00",
        "scatter": "#0D47A1",
        "regline": "#FF1744",
        "sig_bg": "#FFCCD3",
        "ns_bg": "#E3F2FD",
    },
}


# =============================================================================
# ===== 3.相关性矩阵图绘制函数==============================================================
# =============================================================================
def generate_pair_plot(data, output_filename, colors):
    print("正在生成左侧相关性矩阵图")
    variables = ["AAC", "PC", "LC", "TV"]  # 要分析的变量
    sns.set_theme(style="ticks")  # 设置全局绘图风格为，有刻度线但默认无上、右边框
    g = sns.PairGrid(
        data,  # 指定要用于绘图的据集。
        vars=variables,  # 指定绘图要使用的变量
        corner=False,  # 是否只绘制矩阵的左下角。
        diag_sharey=False,  # 对角线上的图是否共享Y轴。True，所有对角线图的Y轴范围相同，False，每个对角线图根据自身数据范围独立调整Y轴。
        height=2.25,
    )  # 每个子图的高度。# 创建一个PairGrid网格对象，用于绘制矩阵图

    def plot_diag(x, **kwargs):  # 用于绘制对角线图的内部函数
        kwargs.pop("color", None)  # 从关键字参数中移除 color键，以避免与自定义颜色冲突
        var_name = x.name  # 获取当前传入数据系列的名称，即列名
        color = colors.get(var_name, "#2f6ba8")  # 获取该变量对应的颜色
        sns.histplot(
            x,  # 要绘制直方图的数据。
            kde=True,  # 是否同时绘制核密度估计曲线。
            color=color,  # 颜色。
            **kwargs,
        )  # 关键字参数。

    g.map_diag(plot_diag)  # 将上面定义的 plot_diag 函数应用到 PairGrid 的对角线位置

    def plot_lower(x, y, **kwargs):  # 用于绘制左下三角图的内部函数
        plt.scatter(x, y, color=colors["scatter"], alpha=1)  # 绘制散点图，颜色，透明度
        sns.regplot(
            x=x,
            y=y,
            scatter=False,
            color=colors["regline"],
            line_kws={"linestyle": "--", "linewidth": 1.5},
        )  # 叠加一条红色的虚线回归线

    g.map_lower(plot_lower)  # 将上面定义的 plot_lower 函数应用到 PairGrid 的左下三角位置

    def plot_upper(x, y, **kwargs):  # 用于绘制右上三角图的内部函数
        ax = plt.gca()  # 获取当前的坐标轴对象
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, y
        )  # 使用 scipy.stats 进行线性回归分析，获取斜率、截距、R值、P值等
        r_squared = r_value**2  # 计算R2
        # 显著性标记设置
        if p_value < 0.0001:
            significance, facecolor = "****", colors["sig_bg"]  # 设置显著性标记，背景色
        elif p_value < 0.05:
            significance, facecolor = "*", colors["sig_bg"]
        else:
            significance = "ns"
            facecolor = colors["ns_bg"]  # 设置不显著背景色
        ax.set_facecolor(facecolor)  # 将计算出的背景色应用到当前坐标轴
        text = f"$R^2 = {r_squared:.4f}$\n$P = {p_value:.4f}$\n{significance}"  # 使用 f-string 格式化要显示的文本
        ax.text(
            0.5,
            0.5,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,  # 在子图中心位置添加文本
            fontsize=12,
            fontweight="bold" if significance != "ns" else "normal",
        )  # 设置字体大小和字重

    g.map_upper(plot_upper)  # 将上面定义的 plot_upper 函数应用到 PairGrid 的右上三角位置

    for ax in g.axes.flatten():  # 遍历相关性矩阵中的每一个子图坐标轴对象
        ax.spines["top"].set_visible(True)  # 设置子图的上边框
        ax.spines["right"].set_visible(True)  # 设置子图的右边框
        ax.tick_params(axis="x", length=0)  # 设置x轴的刻度线
        ax.tick_params(axis="y", length=0)  # 设置y轴的刻度线

    g.fig.suptitle("(a)", x=0.03, y=0.98, fontsize=16, fontweight="bold", ha="left")  # 小标题
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 自动调整子图参数
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"相关性矩阵绘制完成 {output_filename}")


# =============================================================================
# ===== 4.回归分析图绘制函数==============================================================
# =============================================================================
def generate_regression_plots(data, output_filename, colors):
    print("正在生成右侧时间序列回归图")
    variables = ["AAC", "PC", "LC", "TV"]  # 要分析的特征
    sns.set_theme(style="ticks")  # 设置 seaborn 的全局绘图风格为 "ticks"
    fig, axes = plt.subplots(4, 1, figsize=(4, 9), sharex=True)
    for i, var in enumerate(variables):  # 遍历变量列表，同时获取索引 i 和变量名 var
        ax = axes[i]  # 获取当前循环对应的子图坐标轴对象
        sns.regplot(
            x="Year",
            y=var,
            data=data,
            ax=ax,
            color=colors[var],  # 在当前子图上绘制回归图
            scatter_kws={"alpha": 1, "edgecolor": "w", "linewidths": 0.5},  # 设置散点的样式
            line_kws={"color": colors["regline"], "linestyle": "--", "linewidth": 1.5},
        )  # 设置回归线的样式
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            data["Year"], data[var]
        )  # 对当前变量和'Year'进行线性回归分析
        r_squared = r_value**2  # 计算R2
        # 设置显著性标注
        significance = "****" if p_value < 0.0001 else "ns"
        equation = f"$y = {slope:.4f}x {'-' if intercept < 0 else '+'} {abs(intercept):.2f}$\n$R^2 = {r_squared:.4f}$    {significance}"  # 文本标注信息设置
        ax.text(
            0.03, 0.05, equation, transform=ax.transAxes, fontsize=11, va="bottom"
        )  # 在子图的左下角添加文本
        ax.set_ylabel(f"{var} (%)", fontsize=12)  # 设置当前子图的y 标签
        ax.set_xlabel("")  # 清空当前子图的 x 轴标签（因为共享x轴，只在最下方显示）
    axes[-1].set_xlabel("Year", fontsize=12)  # 为最下面的子图设置x轴标签
    fig.suptitle(
        "(b)", x=0.1, y=0.98, fontsize=16, fontweight="bold", ha="left"
    )  # 为整个图形添加一个小标题
    sns.despine()  # 移除图形的上边框和右边框
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# ===== 5.结果图拼接函数==============================================================
# =============================================================================
def stitch_images(left_image_path, right_image_path, output_filename_base="combined_plot"):
    print("\n--- 正在拼接图片... ---")
    img_left = Image.open(left_image_path)  # 加载矩阵图
    img_right = Image.open(right_image_path)  # 加载回归图
    total_width = img_left.width + img_right.width  # 计算拼接后图像的总宽度
    max_height = img_left.height  # 以左图的高度作为基准高度
    if img_right.height != max_height:  # 如果右图的高度与左图不一致
        img_right = img_right.resize(
            (img_right.width, max_height), Image.Resampling.LANCZOS
        )  # 调整右图的高度以匹配左图
    composite_image = Image.new(
        "RGB", (total_width, max_height), color="white"
    )  # 创建一个新的白色背景的空白图像，用于存放拼接结果
    composite_image.paste(img_left, (0, 0))  # 将左图粘贴到新图像的左上角 (0, 0) 位置
    composite_image.paste(img_right, (img_left.width, 0))  # 将右图粘贴到左图的右侧
    # 保存
    output_png = str(OUTPUT_DIR / f"{output_filename_base}_{color_choice}.png")
    composite_image.save(output_png)
    output_pdf = str(OUTPUT_DIR / f"{output_filename_base}_{color_choice}.pdf")
    composite_image.save(output_pdf, "PDF", resolution=100.0)


# =============================================================================
# ===== 6.程序执行部分==============================================================
# =============================================================================
if __name__ == "__main__":
    color_choice = 5  # 选择配色
    active_colors = color_schemes.get(color_choice, color_schemes[1])
    excel_filepath = str(DATA_DIR / "simulated_data.xlsx")  # 输入数据
    dataframe = pd.read_excel(excel_filepath)  # 读取数据
    output_folder = str(OUTPUT_DIR)  # 结果输出位置
    left_plot_file = os.path.join(output_folder, str(OUTPUT_DIR / "plot_part_a.png"))  # 矩阵图保存
    right_plot_file = os.path.join(output_folder, str(OUTPUT_DIR / "plot_part_b.png"))  # 回归图保存
    # 调用矩阵图函数
    generate_pair_plot(dataframe, left_plot_file, colors=active_colors)
    # 调用回归图函数
    generate_regression_plots(dataframe, right_plot_file, colors=active_colors)
    # 凭借函数
    stitch_images(left_plot_file, right_plot_file, "final_combined_figure")

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

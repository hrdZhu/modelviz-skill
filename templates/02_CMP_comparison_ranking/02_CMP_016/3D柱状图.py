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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2. 颜色库设置=======================================
# =========================================================================================
COLORMAP_OPTIONS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "coolwarm",
    "RdYlBu_r",
    "Blues",
    "jet",
    "terrain",
    "Spectral_r",
    "PRGn",
    "PiYG",
    "RdBu_r",
    "hot",
    "gist_earth",
    "ocean",
    "spring",
    "summer",
    "autumn",
]
selected_scheme = 0  # 选择的配色方案


# =========================================================================================
# ======================================3. 绘图函数====================================
# =========================================================================================
def plot_3d_bar_chart(df, selected_scheme_index):
    # 创建一个图形(figure)对象，并设置其大小
    fig = plt.figure(figsize=(12, 9))
    # 在图形中添加一个三维坐标系(axes)
    ax = fig.add_subplot(111, projection="3d")
    xpos = df["X"].values  # 获取'X'列的所有值作为每个柱子的x坐标
    ypos = df["Y"].values  # 获取'Y'列的所有值作为每个柱子的y坐标
    zpos = 0  # 设置所有柱子都从 z=0 的高度开始绘制
    dx = np.ones_like(xpos) * 0.8  # 创建一个和xpos形状相同、元素全为0.8的数组，作为柱子的宽度
    dy = np.ones_like(ypos) * 0.8  # 创建一个和ypos形状相同、元素全为0.8的数组，作为柱子的深度
    dz = df["Z"].values  # 获取'Z'列的所有值作为每个柱子的高度

    selected_cmap_name = COLORMAP_OPTIONS[selected_scheme_index]  # 从列表中获取配色方案的名称字符串
    selected_cmap = plt.get_cmap(selected_cmap_name)  # 使用matplotlib获取对应的颜色映射对象

    norm = plt.Normalize(
        dz.min(), dz.max()
    )  # 创建一个归一化对象，将数据范围(dz的最小值到最大值)映射到0-1之间
    colors = selected_cmap(norm(dz))  # 将每个柱子的高度(dz)通过归一化和颜色映射表转换为具体的颜色
    # 绘制三维柱状图
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort="average")
    ax.set_xlabel("X", labelpad=4, fontsize=16)  # 设置X轴标题
    ax.set_ylabel("Y", labelpad=4, fontsize=16)  # 设置Y轴标题
    ax.set_zlabel("Z", labelpad=4, fontsize=16)  # 设置Z轴标题
    ax.xaxis._axinfo["axisline"]["linewidth"] = 2  # 使用底层方法加粗X轴主线
    ax.yaxis._axinfo["axisline"]["linewidth"] = 2  # 使用底层方法加粗Y轴主线
    ax.zaxis._axinfo["axisline"]["linewidth"] = 2  # 使用底层方法加粗Z轴主线
    ax.tick_params(
        axis="x", direction="out", pad=2, labelsize=16
    )  # 设置X轴刻度线朝外，并调整其与数字的间距
    ax.tick_params(
        axis="y", direction="out", pad=2, labelsize=16
    )  # 设置Y轴刻度线朝外，并调整其与数字的间距
    ax.tick_params(
        axis="z", direction="out", pad=2, labelsize=16
    )  # 设置Z轴刻度线朝外，并调整其与数字的间距
    pane_color = "white"  # 定义背景板颜色为白色
    ax.view_init(elev=15, azim=-135)  # 设置观察视角：仰角30度，方位角-135度
    # 让坐标轴范围根据导入的数据自动调整，并增加一点边距
    ax.set_xlim(df["X"].min() - 0.1, df["X"].max() + 1)  # 设置X轴范围
    ax.set_ylim(df["Y"].min() - 0.1, df["Y"].max() + 1)  # 设置Y轴范围
    ax.set_zlim(0, df["Z"].max() * 1.1)  # 设置Z轴范围，上限比数据最大值高10%
    ax.xaxis.pane.set_facecolor(pane_color)  # 设置X轴背景板的填充颜色
    ax.yaxis.pane.set_facecolor(pane_color)  # 设置Y轴背景板的填充颜色
    ax.zaxis.pane.set_facecolor(pane_color)  # 设置Z轴背景板的填充颜色
    # 显示背景网格线
    ax.grid(True)
    # 获取当前坐标轴的最终范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    # 定义封顶线的样式
    line_style = {"color": "gray", "linestyle": "-", "linewidth": 1, "alpha": 0.8}
    # 绘制顶部平行于Y轴的封顶线
    ax.plot([xlim[1], xlim[1]], ylim, [zlim[1], zlim[1]], **line_style)
    # 绘制顶部平行于X轴的封顶线
    ax.plot(xlim, [ylim[1], ylim[1]], [zlim[1], zlim[1]], **line_style)
    # 绘制前景的垂直封顶线
    ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[0]], zlim, **line_style)
    # 添加颜色条
    mappable = ScalarMappable(
        cmap=selected_cmap, norm=norm
    )  # 创建一个与柱子颜色映射规则完全相同的“可映射”对象
    cbar = fig.colorbar(
        mappable, ax=ax, shrink=0.4, aspect=20, pad=0.015
    )  # 基于该对象创建颜色条，并调整其大小、形状和间距
    cbar.set_label("Height (Z Value)", fontsize=16)  # 设置颜色条的标题和字体
    # 设置颜色条刻度数字的字体和大小
    for t in cbar.ax.get_yticklabels():  # 遍历颜色条上的每一个刻度标签
        t.set_fontsize(14)  # 设置刻度数字的大小
    plt.tight_layout()
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"3d_bar_{selected_scheme}.png"), dpi=300)
    plt.savefig(str(OUTPUT_DIR / f"3d_bar_{selected_scheme}.pdf"))
    return fig, ax, cbar


# =========================================================================================
# ======================================4.数据读取及绘图====================================
# =========================================================================================
input_excel_path = str(DATA_DIR / "data.xlsx")  # 源数据路径
df = pd.read_excel(input_excel_path)  # 读取数据
# 调用函数来创建绘图
fig, ax, cbar = plot_3d_bar_chart(df, selected_scheme)

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

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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
matplotlib.use("Agg")

excel_file_path = str(DATA_DIR / "3D堆叠图/simulated_data.xlsx")
excel_data = pd.read_excel(
    excel_file_path, sheet_name=None, header=None
)  # sheet_name=None 表示读取所有工作表，header=None 表示数据没有表头
all_sheets = [
    df.to_numpy() for df in excel_data.values()
]  # 遍历所有读取的工作表（DataFrame），将它们转换为 NumPy 数组，并存入一个列表中
data = np.stack(
    all_sheets, axis=0
)  # 将列表中的所有二维数组沿着新的轴（axis=0）堆叠起来，形成一个三维数组
z_levels, y_size, x_size = (
    data.shape
)  # 获取三维数组的形状（维度），分别对应 Z 轴的层数、Y 轴的大小、X 轴的大小

fig = plt.figure(figsize=(10, 8))  # 创建一个新的图形对象（画布），并设置其大小为 10x8 英寸
ax = fig.add_subplot(
    111, projection="3d"
)  # 在图形对象上添加一个子图，111 表示 1x1 网格的第 1 个子图，并指定为 3D 投影


x_new = (
    np.arange(y_size + 1) - 0.5
)  # 创建 X 轴的网格坐标点，+1 是因为 plot_surface 需要网格的边界，-0.5 是为了让数据点位于网格中心
y_new = np.arange(x_size + 1) - 0.5  # 创建 Y 轴的网格坐标点，同上
X, Y = np.meshgrid(x_new, y_new)  # 根据 x 和 y 坐标点生成二维网格坐标矩阵
color_schemes = [  # 定义一个包含多种可选颜色映射方案名称的列表
    "RdYlBu",
    "viridis",
    "plasma",
    "inferno",
    "coolwarm",
    "cividis",
    "Spectral_r",
    "PuOr",
    "YlGnBu",
    "hot_r",
]
selected_scheme_index = 9  # 选择列表中的第 10 个颜色映射方案 ('hot_r')
cmap = plt.get_cmap(
    color_schemes[selected_scheme_index], 20
)  # 获取指定的颜色映射对象，并将其分为 20 个离散的颜色级别
norm = plt.Normalize(
    vmin=data.min(), vmax=data.max()
)  # 创建一个归一化对象，将数据值线性映射到 [0, 1] 区间，用于颜色映射


for z_idx in range(z_levels):  # 循环遍历 Z 轴的每一层
    z_val = z_idx  # 获取当前层的 Z 坐标值
    data_slice = data[z_idx]  # 提取当前 Z 层的二维数据切片
    data_slice_transposed = data_slice.T  # 转置数据切片，因为 plot_surface 的 X, Y 对应数据的列和行
    colors = cmap(
        norm(data_slice_transposed)
    )  # 根据数据值，通过归一化和颜色映射计算出每个网格面的颜色
    Z = np.full(X.shape, z_val)  # 创建一个与 X, Y 形状相同且所有值都等于当前 Z 坐标的矩阵
    ax.plot_surface(
        X,
        Y,
        Z,  # 绘制 3D 曲面（这里是一个平面层）
        facecolors=colors,  # 设置每个小面的填充颜色
        rstride=1,  # 设置行的步长为 1，即绘制所有行
        cstride=1,  # 设置列的步长为 1，即绘制所有列
        shade=False,  # 关闭阴影效果，使颜色纯粹地反映数据值
        alpha=1,  # 设置透明度为 1（完全不透明）
        edgecolor="white",  # 设置网格线的颜色为白色
        linewidth=0,
    )  # 设置网格线的宽度为 0（不显示网格线）


mappable = cm.ScalarMappable(cmap=cmap, norm=norm)  # 创建一个可用于生成颜色条的 ScalarMappable 对象
mappable.set_array(data)  # 将原始数据关联到该对象，以确定颜色条的范围
cbar = fig.colorbar(
    mappable, ax=ax, shrink=0.65, aspect=20, pad=-0.02
)  # 在图上添加颜色条，并设置其大小、长宽比和与图像的间距
cbar.set_label("WD3", fontsize=14, weight="bold")  # 设置颜色条的标签文字及其字体大小和粗细
cbar.ax.tick_params(
    labelsize=12, length=5, width=1
)  # 设置颜色条刻度标签的字体大小、刻度线长度和宽度


ax.set_xlabel("Gene", fontsize=14, weight="bold")  # 设置 X 轴的标签文字及其字体大小和粗细
ax.set_ylabel("X Title", fontsize=14, weight="bold")  # 设置 Y 轴的标签文字及其字体大小和粗细
ax.set_zlabel("Z Value", fontsize=14, weight="bold")  # 设置 Z 轴的标签文字及其字体大小和粗细
ax.tick_params(axis="x", labelsize=12, length=6, width=1.2, pad=1)  # 设置 X 轴刻度的样式
ax.tick_params(axis="y", labelsize=12, length=6, width=1.2, pad=1)  # 设置 Y 轴刻度的样式
ax.tick_params(axis="z", labelsize=12, length=6, width=1.2, pad=1)  # 设置 Z 轴刻度的样式
ax.set_zticks(np.arange(z_levels))  # 设置 Z 轴的刻度位置为 0, 1, 2, ...
ax.set_zlim(0, z_levels - 1)  # 设置 Z 轴的显示范围
ax.set_xticks(np.arange(y_size))  # 设置 X 轴的刻度位置
ax.set_yticks(np.arange(x_size))  # 设置 Y 轴的刻度位置
ax.set_xlim(x_new.min(), x_new.max())  # 设置 X 轴的显示范围，以匹配网格边界
ax.set_ylim(y_new.min(), y_new.max())  # 设置 Y 轴的显示范围，以匹配网格边界


ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 设置 X 轴背景面板的颜色为透明 (R, G, B, Alpha)
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 设置 Y 轴背景面板的颜色为透明
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # 设置 Z 轴背景面板的颜色为透明
ax.grid(False)  # 关闭背景网格线


x_min, x_max = ax.get_xlim()  # 获取 X 轴的当前范围
y_min, y_max = ax.get_ylim()  # 获取 Y 轴的当前范围
z_min, z_max = ax.get_zlim()  # 获取 Z 轴的当前范围
pillar_style = dict(color="black", linewidth=1.2)  # 定义柱子的样式（颜色和线宽）
ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], **pillar_style)  # 绘制右前方的柱子
ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], **pillar_style)  # 绘制左后方的柱子
ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], **pillar_style)  # 绘制右后方的柱子

# --- 视图和布局调整 ---
ax.view_init(elev=13, azim=-135)  # 设置 3D 视图的观察角度：仰角(elev)和方位角(azim)
ax.set_box_aspect((10, 12, 12))  # 设置坐标轴的显示比例（X, Y, Z）
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，避免标签重叠

# --- 保存和显示图形 ---
pdf_save_path = str(OUTPUT_DIR / f"result_plot{selected_scheme_index}.png")
plt.savefig(pdf_save_path, dpi=300, bbox_inches="tight")
pdf_save_path = str(OUTPUT_DIR / f"result_plot{selected_scheme_index}.pdf")
plt.savefig(pdf_save_path, dpi=300, bbox_inches="tight")
plt.close("all")  # Interactive display removed; assets were exported above.

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

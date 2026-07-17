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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import griddata

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
output_folder = str(OUTPUT_DIR)
excel_path = os.path.join(output_folder, str(DATA_DIR / "3DZHdata.xlsx"))
df = pd.read_excel(excel_path)
x_flat = df["X_coordinate"].values
y_flat = df["Y_coordinate"].values
z_flat = df["Z_scaled_value"].values

# 根据输入数据的总点数自动计算网格大小
grid_size = int(np.sqrt(len(x_flat)))
# 创建一个新的、规则的网格坐标。我们将在这些点上进行插值。
grid_x, grid_y = np.mgrid[
    min(x_flat) : max(x_flat) : complex(grid_size), min(y_flat) : max(y_flat) : complex(grid_size)
]

# 使用 'cubic' 方法进行三次样条插值，从原始数据(x_flat, y_flat, z_flat)计算出新网格点(grid_x, grid_y)上的Z值。
grid_z = griddata((x_flat, y_flat), z_flat, (grid_x, grid_y), method="cubic")

# 将插值后的网格数据赋值给原来的变量名
X = grid_x
Y = grid_y
Z_scaled = grid_z

# 原始的网格对于清晰的可视化来说可能过于密集。我们每隔几个点取一个，创建一个更稀疏的网格用于三角剖分。
step = 5  # 定义步长，每隔3个点取一个
X_sparse = X[::step, ::step]  # 对X数据进行切片降采样
Y_sparse = Y[::step, ::step]  # 对Y数据进行切片降采样
Z_sparse = Z_scaled[::step, ::step]  # 对Z数据进行切片降采样

# 将稀疏数据展平以用于三角剖分
x_sparse_flat = X_sparse.flatten()  # 将稀疏X二维数组变回一维
y_sparse_flat = Y_sparse.flatten()  # 将稀疏Y二维数组变回一维
z_sparse_flat = Z_sparse.flatten()  # 将稀疏Z二维数组变回一维

fig = plt.figure(figsize=(12, 9))  # 创建一个新的图形窗口，并设置尺寸为12x9英寸
ax = fig.add_subplot(111, projection="3d")  # 在图形窗口中添加一个子图，并指定为3D投影

# 颜色库
color_schemes = {
    "1": "jet",
    "2": "viridis",
    "3": "plasma",
    "4": "inferno",
    "5": "magma",
    "6": "cividis",
    "7": "coolwarm",
    "8": "bwr",
    "9": "seismic",
    "10": "turbo",
    "11": "ocean",
    "12": "gist_earth",
    "13": "terrain",
    "14": "gist_stern",
    "15": "gnuplot",
    "16": "brg",
    "17": "rainbow",
    "18": "gist_rainbow",
    "19": "hsv",
    "20": "twilight",
}
# 选择不同的配色方案
selected_scheme_name = "20"
# 根据上面选择的编号，从字典中获取对应的颜色映射名称
cmap_name = color_schemes[selected_scheme_name]
# 根据获取到的名称，得到最终的颜色映射对象
cmap = plt.get_cmap(cmap_name)

tri = mtri.Triangulation(x_sparse_flat, y_sparse_flat)  # 基于稀疏的x, y坐标进行三角剖分
# 创建一个线段数组，每个线段连接三角剖分中的一条边
segments = np.array(
    [
        (
            (x_sparse_flat[i], y_sparse_flat[i], z_sparse_flat[i]),
            (x_sparse_flat[j], y_sparse_flat[j], z_sparse_flat[j]),
        )
        for i, j in tri.edges
    ]
)

# 定义颜色映射范围时使用完整数据的实际极值，以保证颜色条的准确性
z_min_target = z_flat.min()
z_max_target = z_flat.max()

# 计算每条线段中点的Z值，用于给线段上色
z_mid = np.array([(z_sparse_flat[i] + z_sparse_flat[j]) / 2.0 for i, j in tri.edges])
# 创建一个归一化对象，将Z值映射到[0, 1]
norm = Normalize(vmin=z_min_target, vmax=z_max_target)
# 根据中点的Z值和颜色映射方案，为每条线段生成颜色
colors = cmap(norm(z_mid))

# 创建3D线条集合对象，传入线段、颜色和线宽
line_collection = Line3DCollection(segments, colors=colors, linewidths=1.2)
# 将创建好的线条集合添加到3D坐标轴中
ax.add_collection3d(line_collection)

contour_offset = z_min_target + (z_max_target - z_min_target) * 0.03  # 计算悬空高度，设为Z范围的3%
# 绘制填充的等高线图，将其投影在z方向的指定offset位置
contour = ax.contourf(
    X, Y, Z_scaled, levels=20, zdir="z", offset=contour_offset, cmap=cmap, alpha=0.8
)

ax.set_xlabel("Time of uncertainty", fontsize=14, labelpad=15)  # 设置X轴标签、字体大小和与轴的距离
ax.set_ylabel("Upper and lower\nrange of fluctuation", fontsize=14, labelpad=15)  # 设置Y轴标签
ax.set_zlabel("Capacity(kW·h)", fontsize=14, labelpad=10)  # 设置Z轴标签

ax.set_xticks([0, 0.33, 0.66, 1.0])  # 设置X轴的刻度位置
ax.set_xticklabels(["6(h)", "4-8(h)", "2-10(h)", "0-12(h)"], fontsize=12)  # 设置X轴刻度对应的标签
ax.set_yticks(np.arange(0, 1.1, 0.2))  # 设置Y轴的刻度
ax.tick_params(axis="y", labelsize=12)  # 设置Y轴刻度标签的字体大小

ax.set_zlim(
    z_min_target, z_max_target + (z_max_target - z_min_target) * 0.05
)  # 设置Z轴的显示范围，上限增加5%的裕量
ax.tick_params(axis="z", labelsize=12)  # 设置Z轴刻度标签的字体大小
# 在Z轴最小值和最大值之间生成5个均匀分布的刻度
z_ticks = np.linspace(np.ceil(z_min_target / 1000) * 1000, np.floor(z_max_target / 1000) * 1000, 5)
ax.set_zticks(z_ticks)  # 应用手动设置的Z轴刻度

# 显式设置 X 和 Y 轴的范围，移除自动边距，确保边框线与轴对齐
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

mappable = cm.ScalarMappable(norm=norm, cmap=cmap)  # 创建一个可映射对象，用于颜色条
mappable.set_array(Z_scaled)  # 为可映射对象设置数据数组
# 在图形中添加颜色条，并设置其大小、长宽比和与图形的间距
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.ax.tick_params(labelsize=12)  # 设置颜色条刻度标签的字体大小

ax.xaxis.pane.set_facecolor("white")  # 设置X轴背景面为白色
ax.yaxis.pane.set_facecolor("white")  # 设置Y轴背景面为白色
ax.zaxis.pane.set_facecolor("white")  # 设置Z轴背景面为白色
ax.xaxis.pane.set_edgecolor("w")  # 设置X轴背景面的边框为白色
ax.yaxis.pane.set_edgecolor("w")  # 设置Y轴背景面的边框为白色
ax.zaxis.pane.set_edgecolor("w")  # 设置Z轴背景面的边框为白色

x_min, x_max = 0, 1  # 定义X轴范围
y_min, y_max = 0, 1  # 定义Y轴范围
z_min, z_max = ax.get_zlim()  # 获取Z轴的实际显示范围来画线
lw = 1.0  # 定义边框线的宽度

# 绘制四条黑色边框线以封闭后方和右侧的“开口”，增强立体感
ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], color="black", linewidth=lw)  # 右后方垂直线
ax.plot(
    [x_min, x_max], [y_max, y_max], [z_max, z_max], color="black", linewidth=lw
)  # 上方后侧水平线
ax.plot(
    [x_max, x_max], [y_min, y_max], [z_max, z_max], color="black", linewidth=lw
)  # 上方右侧水平线
ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], color="black", linewidth=lw)  # 右前方垂直线

ax.view_init(elev=20, azim=-145)  # 设置观察视角，elev是仰角，azim是方位角
ax.set_box_aspect((1, 1, 0.7))

# 使用subplots_adjust手动调整边距，替代tight_layout()，避免自动布局失败的警告
fig.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.95)

output_folder = str(OUTPUT_DIR)
png_path = os.path.join(output_folder, f"3dplot{selected_scheme_name}.png")
pdf_path = os.path.join(output_folder, f"3dplot{selected_scheme_name}.pdf")
plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
print(f"图像已保存到: {png_path} 和 {pdf_path}")

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

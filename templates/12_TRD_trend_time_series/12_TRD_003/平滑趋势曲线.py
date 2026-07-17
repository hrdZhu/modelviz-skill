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

import matplotlib  # 导入 matplotlib 主库

matplotlib.use(
    "Agg"
)  # 设置 matplotlib 的图形后端为 'TkAgg'，以确保在某些交互式环境中能正确显示窗口
import matplotlib.lines as mlines  # 导入 matplotlib.lines 模块，用于创建自定义的线对象，常用于图例
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于创建图表，通常简写为 plt
import numpy as np  # 导入 NumPy 库，用于高效的数值计算，通常简写为 np
from scipy.interpolate import (
    make_interp_spline,  # 从 scipy.interpolate 模块导入 make_interp_spline 函数，用于创建平滑曲线
)

plt.rcParams["font.family"] = "Times New Roman"  # 设置全局默认的字体为 'Times New Roman'
# 允许在坐标轴中正常显示负号
plt.rcParams["axes.unicode_minus"] = False  # 设置 matplotlib 参数，以确保坐标轴上的负号可以正常显示
# 创建数据
years = np.array([2000, 2001, 2002, 2003, 2004])  # 定义一个包含年份的 NumPy 数组 (5年数据)
data = np.array(
    [  # 创建一个 NumPy 数组，存放6条曲线在5年中的数据
        [0.1, 0.2, 0.1, 0.3, 0.2],  # 第1条曲线数据
        [0.2, 0.3, 0.4, 0.2, 0.5],  # 第2条曲线数据
        [0.3, 0.2, 0.4, 0.3, 0.5],  # 第3条曲线数据
        [0.4, 0.5, 0.3, 0.4, 0.5],  # 第4条曲线数据
        [0.5, 0.6, 0.5, 0.7, 0.6],  # 第5条曲线数据
        [0.6, 0.7, 0.6, 0.8, 0.7],  # 第6条曲线数据
    ]
)

# 插值
smooth_years = np.linspace(2000, 2004, 500)  # 创建一个包含500个点的密集年份数组，用于绘制平滑曲线
splines = [
    make_interp_spline(years, data[i], k=3) for i in range(6)
]  # 对6条曲线数据分别进行3次样条插值，生成平滑函数

# 绘制平滑的曲线并保留标记
plt.plot(
    smooth_years, splines[0](smooth_years), "b-", label="X1"
)  # 绘制第1条平滑曲线，颜色为蓝色(b)，样式为实线(-)
plt.plot(years, data[0], "b^", markersize=8)  # 在原始数据点上绘制蓝色(b)三角形(^)标记
plt.plot(
    smooth_years, splines[1](smooth_years), "r-", label="X2"
)  # 绘制第2条平滑曲线，颜色为红色(r)
plt.plot(years, data[1], "ro", markersize=8)  # 在原始数据点上绘制红色(r)圆圈(o)标记
plt.plot(
    smooth_years, splines[2](smooth_years), "g-", label="X3"
)  # 绘制第3条平滑曲线，颜色为绿色(g)
plt.plot(years, data[2], "gs", markersize=8)  # 在原始数据点上绘制绿色(g)方形(s)标记
plt.plot(
    smooth_years, splines[3](smooth_years), "m-", label="X4"
)  # 绘制第4条平滑曲线，颜色为品红色(m)
plt.plot(years, data[3], "mD", markersize=8)  # 在原始数据点上绘制品红色(m)菱形(D)标记
plt.plot(
    smooth_years, splines[4](smooth_years), "c-", label="X5"
)  # 绘制第5条平滑曲线，颜色为青色(c)
plt.plot(years, data[4], "c<", markersize=8)  # 在原始数据点上绘制青色(c)左三角(<)标记
plt.plot(
    smooth_years, splines[5](smooth_years), "y-", label="X6"
)  # 绘制第6条平滑曲线，颜色为黄色(y)
plt.plot(years, data[5], "yH", markersize=8)  # 在原始数据点上绘制黄色(y)六边形(H)标记

# 设置标题和标签
# 自定义 X 轴刻度
plt.xticks(years)  # 设置X轴的刻度只显示指定的年份
# 自定义图例
# 使用 HandlerLine 来处理特殊图例样式
legend_elements = [  # 创建一个列表，用于自定义图例中的每个元素
    mlines.Line2D(
        [0], [0], color="blue", lw=2, label="X1", marker="^", markersize=8
    ),  # 创建一个图例元素：蓝色实线带三角形标记
    mlines.Line2D(
        [0], [0], color="red", lw=2, label="X2", marker="o", markersize=8
    ),  # 创建一个图例元素：红色实线带圆形标记
    mlines.Line2D(
        [0], [0], color="green", lw=2, label="X3", marker="s", markersize=8
    ),  # 创建一个图例元素：绿色实线带方形标记
    mlines.Line2D(
        [0], [0], color="purple", lw=2, label="X4", marker="D", markersize=8
    ),  # 创建一个图例元素：紫色实线带菱形标记
    mlines.Line2D(
        [0], [0], color="cyan", lw=2, label="X5", marker="<", markersize=8
    ),  # 创建一个图例元素：青色实线带左三角标记
    mlines.Line2D(
        [0], [0], color="yellow", lw=2, label="X6", marker="H", markersize=8
    ),  # 创建一个图例元素：黄色实线带六边形标记
]

plt.legend(
    handles=legend_elements, loc="center", bbox_to_anchor=(0.5, -0.1), ncol=6
)  # 显示图例，使用自定义的元素，位置在图下方居中，分6列显示

# 显示图形
plt.savefig(str(OUTPUT_DIR / "smoothed_trajectory_single_panel.png"), dpi=300, bbox_inches="tight")
plt.close("all")

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于创建图表，通常简写为 plt
import numpy as np  # 导入 NumPy 库，用于高效的数值计算，通常简写为 np
from matplotlib.lines import (
    Line2D,  # 从 matplotlib.lines 模块导入 Line2D 类，用于创建线对象，常用于自定义图例
)
from scipy.interpolate import (
    make_interp_spline,  # 从 scipy.interpolate 模块导入 make_interp_spline 函数，用于创建平滑曲线
)

# ----------------------
# 1. 手动设置年份和数据
# ----------------------

# 自定义年份（2000～2004）
years = np.array([2000, 2001, 2002, 2003, 2004])  # 定义一个包含年份的 NumPy 数组

# 自定义 6 个地区的 5 条曲线数据
data = np.array(
    [  # 定义第一个地区的数据，包含5条曲线
        [0.1, 0.2, 0.15, 0.05, 0.1],  # 地区 1 的曲线 1
        [0.2, 0.3, 0.35, 0.2, 0.45],  # 地区 1 的曲线 2
        [0.3, 0.4, 0.45, 0.3, 0.35],  # 地区 1 的曲线 3
        [0.4, 0.5, 0.35, 0.4, 0.25],  # 地区 1 的曲线 4
        [0.5, 0.6, 0.65, 0.2, 0.35],  # 地区 1 的曲线 5
    ]
)

# 其他地区的示例数据
data_2 = np.array(
    [  # 定义其他地区的数据，也包含5条曲线
        [0.15, 0.25, 0.2, 0.3, 0.35],
        [0.25, 0.35, 0.4, 0.35, 0.1],
        [0.35, 0.45, 0.5, 0.15, 0.2],
        [0.45, 0.55, 0.6, 0.25, 0.4],
        [0.55, 0.65, 0.5, 0.25, 0.3],
    ]
)
data_3 = np.array(
    [  # 定义第一个地区的数据，包含5条曲线
        [0.1, 0.2, 0.15, 0.05, 0.1],  # 地区 1 的曲线 1
        [0.2, 0.3, 0.35, 0.4, 0.45],  # 地区 1 的曲线 2
        [0.3, 0.4, 0.45, 0.2, 0.35],  # 地区 1 的曲线 3
        [0.4, 0.5, 0.35, 0.4, 0.25],  # 地区 1 的曲线 4
        [0.5, 0.6, 0.65, 0.2, 0.35],  # 地区 1 的曲线 5
    ]
)

# 其他地区的示例数据
data_4 = np.array(
    [  # 定义其他地区的数据，也包含5条曲线
        [0.15, 0.25, 0.2, 0.3, 0.35],
        [0.25, 0.35, 0.4, 0.35, 0.1],
        [0.35, 0.45, 0.5, 0.15, 0.2],
        [0.45, 0.55, 0.6, 0.25, 0.4],
        [0.55, 0.65, 0.5, 0.25, 0.3],
    ]
)
data_5 = np.array(
    [  # 定义第一个地区的数据，包含5条曲线
        [0.1, 0.2, 0.15, 0.05, 0.1],  # 地区 1 的曲线 1
        [0.2, 0.3, 0.35, 0.4, 0.45],  # 地区 1 的曲线 2
        [0.3, 0.4, 0.45, 0.2, 0.35],  # 地区 1 的曲线 3
        [0.4, 0.5, 0.35, 0.4, 0.25],  # 地区 1 的曲线 4
        [0.5, 0.6, 0.65, 0.2, 0.35],  # 地区 1 的曲线 5
    ]
)

# 其他地区的示例数据
data_6 = np.array(
    [  # 定义其他地区的数据，也包含5条曲线
        [0.15, 0.25, 0.2, 0.3, 0.35],
        [0.25, 0.35, 0.4, 0.35, 0.1],
        [0.35, 0.45, 0.5, 0.15, 0.2],
        [0.45, 0.55, 0.6, 0.25, 0.4],
        [0.55, 0.65, 0.5, 0.25, 0.3],
    ]
)
# 将其合并为所有地区的数据（6 个地区）
all_data = np.array(
    [data, data_2, data_3, data_4, data_5, data_6]
)  # 将6个地区的数据合并成一个三维数组

# ----------------------
# 2. 自定义标记与颜色
# ----------------------

# 手动指定一组颜色（所有5条曲线的颜色都相同）
colors = ["blue", "red", "green", "purple", "cyan"]  # 定义一个颜色列表，用于5条不同的曲线

# 手动指定标记形状
markers = ["^", "o", "s", "D", "<"]  # 定义一个标记列表，用于5条不同的曲线

# 3. 创建 2x3 的子图布局
fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 创建一个2行3列的子图网格，整个图窗大小为15x10英寸
axs = axs.flatten()  # 将2x3的子图数组扁平化为一维数组，方便后续循环遍历

# ----------------------
# 4. 绘制每个子图
# ----------------------

titles = ["Temperature", "Precipitation", "purple", "NDSI", "NIR", "AGB"]  # 定义6个子图的标题

for i in range(6):  # 循环6次，对应6个地区（子图）
    ax = axs[i]  # 获取当前循环的子图对象
    ax.text(
        0.15, 0.9, titles[i], ha="center", va="bottom", fontsize=10, transform=ax.transAxes
    )  # 在子图的相对位置(0.15, 0.9)处添加标题

    # ax.set_xlabel("Year", fontsize=10) # 设置X轴标签（此行被注释掉了）
    # ax.set_ylabel("q value", fontsize=10) # 设置Y轴标签（此行被注释掉了）

    # 调整 xlim 留出空白
    ax.set_xlim(years.min() - 0.2, years.max() + 0.2)  # 设置X轴的范围，在数据两端各留出0.2的空白
    ax.set_xticks(years)  # 设置X轴的刻度只显示指定的年份

    # 绘制每个地区的 5 条曲线
    for j in range(5):  # 循环5次，对应每个地区的5条曲线
        y = all_data[i, j, :]  # 获取当前地区(i)的第 j 条曲线的数据
        # 使用样条插值生成平滑曲线
        x_smooth = np.linspace(
            years.min(), years.max(), 300
        )  # 创建一个密集的X轴坐标点用于绘制平滑曲线
        spline = make_interp_spline(years, y, k=3)  # 对当前曲线的数据进行3次样条插值
        y_smooth = spline(x_smooth)  # 计算平滑曲线在密集坐标点上的Y值
        # 绘制平滑曲线
        ax.plot(
            x_smooth, y_smooth, color=colors[j], linestyle="-", lw=2
        )  # 绘制平滑曲线，使用预设的颜色和线宽
        # 绘制原始数据点标记
        ax.plot(
            years, y, color=colors[j], marker=markers[j], linestyle="", markersize=8
        )  # 在原始数据点上绘制标记，不画连接线

# ----------------------
# 5. 添加统一图例
# ----------------------

# 创建图例条目

legend_elements = [  # 创建一个列表，用于自定义图例中的每个元素
    mlines.Line2D(
        [0], [0], color="blue", lw=2, label="X1", marker="^", markersize=8
    ),  # 创建一个图例元素：蓝色实线带三角形标记
    mlines.Line2D(
        [0], [0], color="red", lw=2, label="X2", marker="o", markersize=8
    ),  # 创建一个图例元素：红色实线带圆形标记
    mlines.Line2D(
        [0], [0], color="green", lw=2, label="X3", marker="s", markersize=8
    ),  # 创建一个图例元素：绿色实线带方形标记
    mlines.Line2D(
        [0], [0], color="purple", lw=2, label="X4", marker="D", markersize=8
    ),  # 创建一个图例元素：紫色实线带菱形标记
    mlines.Line2D(
        [0], [0], color="cyan", lw=2, label="X5", marker="<", markersize=8
    ),  # 创建一个图例元素：青色实线带左三角标记
    # mlines.Line2D([0], [0], color='yellow', lw=2, label='X6', marker='H', markersize=8) # 第6个图例元素（此行被注释掉了）
]

# 添加统一图例到所有图的下方
fig.legend(  # 在整个图窗(fig)级别添加一个统一的图例
    handles=legend_elements,  # 使用上面自定义的图例元素列表
    loc="center",  # 图例的定位基准点设置为中心
    ncol=5,  # 图例分为5列显示
    bbox_to_anchor=(0.5, 0.05),  # 将图例的定位基准点放置在图窗相对坐标(0.5, 0.05)处，即底部居中
    frameon=False,  # 不显示图例的外边框
    fontsize=12,  # 设置图例的字体大小
)

# ----------------------
# 6. 调整布局并显示图形
# ----------------------


fig.savefig(str(OUTPUT_DIR / "smoothed_trajectory_multi_panel.png"), dpi=300, bbox_inches="tight")


plt.close(fig)

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

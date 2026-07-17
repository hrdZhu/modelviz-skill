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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.path import Path

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = 14
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# 颜色库
color_schemes = {  # 定义一个名为color_schemes的字典，存储多种配色方案
    "1": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c1", ["#FFDDC1", "#FF6347"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d1", ["#D6EAF8", "#8E44AD"]),
    },
    "2": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c2", ["#d4fc79", "#96e6a1"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d2", ["#f8cdda", "#994e63"]),
    },
    "3": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c3", ["#fff1eb", "#ace0f9"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d3", ["#d8b5ff", "#1eae98"]),
    },
    "4": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c4", ["#f093fb", "#f5576c"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d4", ["#4facfe", "#00f2fe"]),
    },
    "5": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c5", ["#fdfbfb", "#ebedee"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d5", ["#485461", "#28313b"]),
    },
    "6": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c6", ["#ffc3a0", "#ffafbd"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d6", ["#2193b0", "#6dd5ed"]),
    },
    "7": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c7", ["#f6d365", "#fda085"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d7", ["#a1c4fd", "#c2e9fb"]),
    },
    "8": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c8", ["#d4fc79", "#96e6a1"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d8", ["#a18cd1", "#fbc2eb"]),
    },
    "9": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c9", ["#fbc2eb", "#a6c1ee"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d9", ["#a8edea", "#fed6e3"]),
    },
    "10": {
        "charge": mcolors.LinearSegmentedColormap.from_list("c10", ["#84fab0", "#8fd3f4"]),
        "discharge": mcolors.LinearSegmentedColormap.from_list("d10", ["#ff9a9e", "#fecfef"]),
    },
}
selected_scheme_name = "9"  # 设置当前选择的配色方案
selected_scheme = color_schemes[selected_scheme_name]  # 从字典中根据名称获取选定的配色方案


class HandlerHalfCircleLine(HandlerLine2D):  # 定义一个自定义图例处理器类，继承自HandlerLine2D
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):  # 重写create_artists方法来创建自定义的图例图标
        x_center = xdescent + 0.5 * width  # 计算图例中标记的中心x坐标
        y_center = height / 2  # 计算图例中标记的中心y坐标
        line_artists = super().create_artists(
            legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        )  # 调用父类的方法，创建原始的线条图例
        p1 = Line2D(
            [x_center],
            [y_center],
            marker="o",  # 创建一个Line2D对象作为空心圆（底层）
            markersize=orig_handle.get_markersize(),  # 设置标记大小与原始句柄一致
            markerfacecolor="white",  # 设置标记的填充颜色为白色
            markeredgecolor="red",  # 设置标记的边缘颜色为红色
            markeredgewidth=1.5,  # 设置标记的边缘线宽为1.5
            linestyle="None",
            transform=trans,
        )  # 不显示线条，并应用变换
        arc = Path.arc(180, 360)  # 创建一个从180度到360度的圆弧路径
        verts = np.concatenate(
            [arc.vertices, [[0, 0]]]
        )  # 将圆弧的顶点与一个闭合点(0,0)连接起来，形成半圆形
        codes = np.concatenate([arc.codes, [Path.CLOSEPOLY]])  # 将圆弧的路径代码与一个闭合代码连接
        half_circle_path = Path(verts, codes)  # 使用顶点和代码创建一个完整的半圆形路径对象
        p2 = Line2D(
            [x_center],
            [y_center],
            marker=half_circle_path,  # 创建一个Line2D对象作为实心半圆（顶层）
            markersize=orig_handle.get_markersize(),  # 设置标记大小与原始句柄一致
            markerfacecolor="red",  # 设置标记的填充颜色为红色
            markeredgecolor="none",  # 不显示标记的边缘线
            linestyle="None",
            transform=trans,
        )  # 不显示线条，并应用变换
        return line_artists + [p1, p2]  # 返回原始线条图例、空心圆和实心半圆的组合


time = np.arange(1, 25)  # 创建一个从1到24的整数数组，代表时间(h)
charging_power = np.array(
    [
        0,
        3000,
        2900,
        3000,
        3000,
        2950,
        2500,
        2600,
        0,
        0,
        0,
        0,
        1100,
        900,
        150,
        0,
        0,
        0,
        0,
        0,
        0,
        150,
        0,
        0,
    ]
)
discharging_power = np.array(
    [
        -2000,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        -3000,
        -2500,
        -2800,
        -2400,
        0,
        0,
        0,
        0,
        -3000,
        -2000,
        -3000,
        -2700,
        -2500,
        0,
        0,
        0,
    ]
)
soc = np.array(
    [  # 创建一个numpy数组，存储每个小时对应的电池荷电状态(SOC, kW·h)
        2500,
        6000,
        9000,
        12000,
        15000,
        18000,
        20500,
        21500,
        18500,
        16000,
        13500,
        12000,
        13000,
        14000,
        14200,
        14500,
        14400,
        11500,
        8500,
        5500,
        2800,
        2600,
        2700,
        2800,
    ]
)


def gradient_bar(
    ax, x, y, width=0.7, bottom=0, cmap=plt.cm.viridis
):  # 定义一个函数，用于绘制带有渐变色的单个柱子
    X = np.array(
        [[x - width / 2, x + width / 2], [x - width / 2, x + width / 2]]
    )  # 定义柱子的X坐标网格（左边界和右边界）
    Y = np.array(
        [[bottom, bottom], [bottom + y, bottom + y]]
    )  # 定义柱子的Y坐标网格（下边界和上边界）
    Z = np.array([[0, 0], [1, 1]])  # 定义用于颜色映射的Z值，从下到上为0到1
    if bottom < 0:  # 判断柱子是否在x轴下方
        Z = 1 - Z  # 如果是，则反转Z值，使渐变方向相反
    ax.pcolormesh(
        X, Y, Z, cmap=cmap, shading="gouraud", vmin=0, vmax=1
    )  # 使用pcolormesh绘制一个渐变填充的矩形（柱子）


fig, ax1 = plt.subplots(
    figsize=(12, 10)
)  # 创建一个图形(fig)和一个主坐标轴(ax1)，设置图形大小为12x12英寸
for t, charge, discharge in zip(time, charging_power, discharging_power):  # 遍历数据
    if charge > 0:
        gradient_bar(ax1, t, charge, cmap=selected_scheme["charge"])  # 调用函数绘制charge柱状图
    if discharge < 0:  # 如果当前小时的放电功率小于0
        gradient_bar(
            ax1, t, abs(discharge), bottom=discharge, cmap=selected_scheme["discharge"]
        )  # 调用函数绘制discharge柱状图，注意bottom参数设为负值
bar_charge = plt.Rectangle(
    (0, 0), 1, 1, fc=selected_scheme["charge"](0.5), label="Charging Power"
)  # 创建一个矩形对象，作为charge图例的代理
bar_discharge = plt.Rectangle(
    (0, 0), 1, 1, fc=selected_scheme["discharge"](0.5), label="Discharging Power"
)  # 创建一个矩形对象，作为discharge图例的代理
ax1.set_xlabel("Time(h)", fontsize=22, labelpad=10)  # 设置主坐标轴的X轴标签
ax1.set_ylabel("Power(kW)", fontsize=22)  # 设置主坐标轴（左侧）的Y轴标签
ax1.set_ylim(-3500, 3500)  # 设置主坐标轴（左侧）的Y轴范围
ax1.set_xticks(np.arange(0, 26, 5))  # 设置X轴的刻度位置（从0到25，步长为5）
ax1.axhline(0, color="black", linewidth=1)  # 在y=0的位置绘制一条黑色的水平线
ax1.set_xlim(0.5, 24.5)  # 设置X轴的显示范围，使两边的柱子不贴边
ax1.minorticks_on()  # 开启主坐标轴的次刻度线
ax1.tick_params(
    axis="both", which="major", direction="in", labelsize=22, width=1.5, length=6
)  # 设置主刻度线的样式（方向朝内，标签大小，线宽，长度）
ax1.tick_params(axis="both", which="minor", direction="in", width=1, length=3)  # 设置次刻度线的样式
ax2 = ax1.twinx()  # 创建一个共享X轴的次坐标轴（右侧Y轴）
ax2.plot(
    time, soc, color="red", linewidth=2.5, zorder=1
)  # 在次坐标轴上绘制SOC折线图，设置颜色、线宽和绘图层次
ax2.plot(
    time,
    soc,  # 在折线图上添加标记点（底层：空心圆）
    marker="o",
    markersize=8,
    markerfacecolor="white",  # 设置标记形状为圆圈，大小为8，填充色为白色
    markeredgecolor="red",
    markeredgewidth=1.5,  # 设置标记边缘颜色为红色，边缘线宽为1.5
    linestyle="None",
    zorder=2,
)  # 不显示连接线，绘图层次高于折线
half_circle_path = Path.arc(180, 360).vertices  # 获取一个从180度到360度的圆弧路径的顶点
half_circle_path = np.concatenate([half_circle_path, [[0, 0]]])  # 将圆弧顶点与闭合点(0,0)连接
half_circle_path = Path(
    half_circle_path, np.concatenate([Path.arc(180, 360).codes, [Path.CLOSEPOLY]])
)  # 创建一个完整的半圆形路径对象
ax2.plot(
    time,
    soc,  # 在折线图上添加标记点（顶层：实心半圆）
    marker=half_circle_path,
    markersize=8,
    markerfacecolor="red",  # 设置标记形状为自定义的半圆形，填充色为红色
    markeredgecolor="none",
    linestyle="None",
    zorder=3,
)  # 不显示边缘线和连接线，绘图层次最高
ax2.set_ylabel("SOC(kW·h)", fontsize=22)  # 设置次坐标轴（右侧）的Y轴标签
ax2.set_ylim(0, 25000)  # 设置次坐标轴（右侧）的Y轴范围
ax2.minorticks_on()  # 开启次坐标轴的次刻度线
ax2.tick_params(
    axis="y", which="major", direction="in", labelsize=22, width=1.5, length=6
)  # 设置次坐标轴Y轴主刻度线的样式
ax2.tick_params(
    axis="y", which="minor", direction="in", width=1, length=3
)  # 设置次坐标轴Y轴次刻度线的样式
for spine in ax1.spines.values():  # 遍历主坐标轴的所有边框（上下左右）
    spine.set_linewidth(1.5)  # 将边框的线宽设置为1.5

line_proxy_handle = Line2D(
    [0], [0], linestyle="-", color="red", markersize=8, label="Capacity of SESS"
)  # 为SOC折线创建一个代理图例句柄，以便使用自定义处理器
bar_handles = [bar_charge, bar_discharge]  # 将矩形句柄放入一个列表

legend1 = ax1.legend(
    handles=[line_proxy_handle],  # 使用折线的代理句柄创建图例
    loc="upper right",  # 设置图例的大致位置在右上角
    bbox_to_anchor=(0.58, 0.99),  # 使用bbox_to_anchor精确定位图例的位置
    frameon=False,  # 不显示图例边框
    fontsize=22,  # 设置图例的字体大小
    handler_map={line_proxy_handle: HandlerHalfCircleLine()},
)  # 指定折线句柄使用我们自定义的处理器
legend2 = ax1.legend(
    handles=bar_handles,  # 使用柱状图的代理句柄列表创建图例
    loc="upper right",  # 设置图例的大致位置在右上角
    bbox_to_anchor=(1.0, 0.99),  # 使用bbox_to_anchor精确定位图例的位置
    frameon=False,  # 不显示图例边框
    fontsize=22,
)  # 设置图例的字体大小
ax1.add_artist(legend1)

output_folder = str(OUTPUT_DIR)  # 定义输出文件夹的路径
file_name = f"chart_scheme_{selected_scheme_name}.pdf"
output_path = os.path.join(output_folder, file_name)
plt.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"图片已成功保存到: {output_path}")
file_name = f"chart_scheme_{selected_scheme_name}.png"
output_path = os.path.join(output_folder, file_name)
plt.savefig(output_path, bbox_inches="tight")
print(f"图片已成功保存到: {output_path}")
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

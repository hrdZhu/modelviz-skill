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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
color_schemes = {
    0: [
        "#80B1DA",
        "#F69A8F",
        "#7FC9B0",
        "#AB87C4",
        "#FFD700",
        "#ff764a",
        "#ffa600",
        "#58508d",
        "#ff6361",
        "#f95d6a",
    ],
    1: [
        "#003f5c",
        "#374c80",
        "#7a5195",
        "#bc5090",
        "#ef5675",
        "#ff764a",
        "#ffa600",
        "#58508d",
        "#ff6361",
        "#f95d6a",
    ],
    2: [
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
        "#003f5c",
        "#f95d6a",
        "#ff7c43",
    ],
    3: [
        "#335c67",
        "#fff3b0",
        "#e09f3e",
        "#9e2a2b",
        "#540b0e",
        "#6a994e",
        "#a7c957",
        "#f2e8cf",
        "#bc4749",
        "#386641",
    ],
    4: [
        "#f8b195",
        "#f67280",
        "#c06c84",
        "#6c5b7b",
        "#355c7d",
        "#99b898",
        "#feceab",
        "#ff847c",
        "#e84a5f",
        "#2a363b",
    ],
    5: [
        "#4a4e69",
        "#9a8c98",
        "#c9ada7",
        "#f2e9e4",
        "#22223b",
        "#4a4e69",
        "#f2e9e4",
        "#9a8c98",
        "#c9ada7",
        "#22223b",
    ],
    6: [
        "#8ecae6",
        "#219ebc",
        "#126782",
        "#023047",
        "#ffb703",
        "#fd9e02",
        "#fb8500",
        "#8ecae6",
        "#219ebc",
        "#023047",
    ],
    7: [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#264653",
    ],
    8: [
        "#a7c957",
        "#f2e8cf",
        "#bc4749",
        "#6a994e",
        "#386641",
        "#f2e8cf",
        "#bc4749",
        "#6a994e",
        "#386641",
        "#a7c957",
    ],
    9: [
        "#5f0f40",
        "#9a031e",
        "#fb8b24",
        "#e36414",
        "#0f4c5c",
        "#9a031e",
        "#fb8b24",
        "#e36414",
        "#0f4c5c",
        "#5f0f40",
    ],
    10: [
        "#588157",
        "#3a5a40",
        "#344e41",
        "#a3b18a",
        "#dad7cd",
        "#3a5a40",
        "#344e41",
        "#a3b18a",
        "#dad7cd",
        "#588157",
    ],
    11: [
        "#fec5bb",
        "#fcd5ce",
        "#fae1dd",
        "#f8edeb",
        "#e8e8e4",
        "#d8e2dc",
        "#ece4db",
        "#ffe5d9",
        "#ffd7ba",
        "#fec89a",
    ],
    12: [
        "#ffc09f",
        "#ffee93",
        "#fcf5c7",
        "#a0ced9",
        "#adf7b6",
        "#ffee93",
        "#fcf5c7",
        "#a0ced9",
        "#adf7b6",
        "#ffc09f",
    ],
    13: [
        "#cfbaf0",
        "#a3c4f3",
        "#90dbf4",
        "#8eecf5",
        "#98f5e1",
        "#b9fbc0",
        "#a3c4f3",
        "#90dbf4",
        "#8eecf5",
        "#98f5e1",
    ],
    14: [
        "#d4e09b",
        "#f6f4d2",
        "#cbdfbd",
        "#f19c79",
        "#a44a3f",
        "#f6f4d2",
        "#cbdfbd",
        "#f19c79",
        "#a44a3f",
        "#d4e09b",
    ],
    15: [
        "#f08080",
        "#f4978e",
        "#f8ad9d",
        "#fbc4ab",
        "#ffdab9",
        "#f4978e",
        "#f8ad9d",
        "#fbc4ab",
        "#ffdab9",
        "#f08080",
    ],
    16: [
        "#03045e",
        "#023e8a",
        "#0077b6",
        "#0096c7",
        "#00b4d8",
        "#48cae4",
        "#90e0ef",
        "#ade8f4",
        "#caf0f8",
        "#03045e",
    ],
    17: [
        "#4cc9f0",
        "#4361ee",
        "#3f37c9",
        "#3a0ca3",
        "#480ca8",
        "#560bad",
        "#7209b7",
        "#b5179e",
        "#f72585",
        "#4361ee",
    ],
    18: [
        "#f94144",
        "#f3722c",
        "#f8961e",
        "#f9844a",
        "#f9c74f",
        "#90be6d",
        "#43aa8b",
        "#4d908e",
        "#577590",
        "#277da1",
    ],
    19: [
        "#f94144",
        "#f3722c",
        "#f8961e",
        "#f9c74f",
        "#90be6d",
        "#43aa8b",
        "#577590",
        "#f94144",
        "#f3722c",
        "#f8961e",
    ],
    20: [
        "#ef476f",
        "#ffd166",
        "#06d6a0",
        "#118ab2",
        "#073b4c",
        "#ffd166",
        "#06d6a0",
        "#118ab2",
        "#073b4c",
        "#ef476f",
    ],
}


# =========================================================================================
# ====================================== 3. 绘图函数定义 ========================================
# =========================================================================================
def create_circular_barplot(
    data, color_scheme_id=1, output_filename="circular_bar_chart", **kwargs
):

    start_radius = kwargs.get("start_radius", 1.0)  # 条形的中心半径，默认为1.0
    thickness = kwargs.get("thickness", 0.18)  # 条形的厚度，默认为0.18
    max_scale_value = kwargs.get("max_scale_value", 0.8)  # 整个圆周的最大值，默认为0.8
    axis_min = kwargs.get("axis_min", 0.0)  # 外圈刻度的最小值，默认为0.0
    axis_max = kwargs.get("axis_max", 0.7)  # 外圈刻度的最大值，默认为0.7
    num_ticks = kwargs.get("num_ticks", 8)  # 外圈刻度的数量，默认为8

    categories = list(data.keys())  # 从数据字典中提取所有键，也就是类别，并转换为一个列表
    chosen_colors = color_schemes.get(color_scheme_id, color_schemes[1])  # 获取配色方案

    radii = {}  # 用于存储每个类别的中心半径
    colors = {}  # 用于存储每个类别的颜色
    for i, cat in enumerate(categories):  # 遍历每个类别及其索引
        radii[cat] = start_radius + i * (
            thickness + 0.07
        )  # 计算每个类别条形的中心半径，确保每一个条形之间不重叠
        colors[cat] = chosen_colors[
            i % len(chosen_colors)
        ]  # 为每个类别分配颜色，但是要注意如果超出了我预设的颜色，就需要往里面加新的颜色了

    fig, ax = plt.subplots(
        figsize=(10, 10), subplot_kw={"projection": "polar"}
    )  # 创建画布，使用极坐标投影

    # 循环绘图
    for cat in categories:  # 遍历每一个类别来进行绘制
        value = data[cat]  # 获取当前类别对应的数值
        radius = radii[cat]  # 获取当前类别的中心半径
        color = colors[cat]  # 获取当前类别的颜色
        # 根据数值计算其在圆周上对应的弧长，并生成200个点来平滑绘制圆弧
        theta = np.linspace(0, (value / max_scale_value) * 2 * np.pi, 200)
        # 生成一个等差数列，用于表示角度
        theta = np.linspace(
            0,  # 起始角度
            (value / max_scale_value) * 2 * np.pi,  # 结束角度
            200,
        )
        theta = np.pi / 2 - theta  # 将坐标系逆时针旋转90度，使得0点在正上方，条形顺时针增长
        r_inner = radius - thickness / 2  # 计算当前条形的内径
        r_outer = radius + thickness / 2  # 计算当前条形的外径

        # 填充条形区域
        ax.fill_between(
            theta,  # 指定填充区域的角度坐标（x轴）
            r_inner,  # 指定填充区域的内边界的径向坐标
            r_outer,  # 指定填充区域的外边界的径向坐标
            color=color,  # 填充颜色
            edgecolor="grey",  # 外框的颜色
            linewidth=0.7,
        )  # 外框的宽度

        # 计算条形尾部的角度
        end_angle_rad = np.pi / 2 - (value / max_scale_value) * 2 * np.pi

        angle_offset = 0.15  # 角度偏移量，用于将文本从最末端向内移动一点
        label_angle_rad = end_angle_rad + angle_offset  # 计算文本标注最终所在位置的角度

        label_rotation_deg = np.rad2deg(
            label_angle_rad - np.pi / 2
        )  # 将文本位置的弧度转换为角度，减去90度以使其与圆弧的切线平行

        # 添加文本
        ax.text(
            label_angle_rad,  # x，在极坐标中代表角度
            radius,  # y，在极坐标中代表半径
            f"{value:.0f}",  # 文本
            ha="center",  # 水平对齐方式
            va="center",  # 垂直对齐方式
            rotation=label_rotation_deg,  # 自身的旋转角度
            fontsize=18,  # 字体大小
            color="black",  # 颜色
        )

    axis_radius = max(radii.values()) + thickness / 2 + 0.2  # 计算外圈坐标轴的半径
    arc_start_rad = (
        np.pi / 2 - (axis_min / max_scale_value) * 2 * np.pi
    )  # 计算外圈坐标轴弧线的起始角度
    arc_end_rad = (
        np.pi / 2 - (axis_max / max_scale_value) * 2 * np.pi
    )  # 计算外圈坐标轴弧线的结束角度

    axis_theta = np.linspace(arc_end_rad, arc_start_rad, 200)  # 生成一系列角度点来绘制坐标轴弧线

    ax.plot(
        axis_theta, np.full_like(axis_theta, axis_radius), color="grey", linewidth=1.2
    )  # 在指定半径处绘制灰色的坐标轴弧线

    tick_values = np.linspace(axis_min, axis_max, num_ticks)  # 根据设定的范围和数量，均匀生成刻度值
    angle_rad_min = np.pi / 2 - (axis_min / max_scale_value) * 2 * np.pi  # 计算起始刻度值的角度
    # 其实刻度
    ax.text(
        angle_rad_min,
        axis_radius + 0.08,
        f"{axis_min:.0f}",
        ha="center",
        va="center",
        rotation=np.rad2deg(angle_rad_min) - 90,
        fontsize=28,
    )
    angle_rad_max = np.pi / 2 - (axis_max / max_scale_value) * 2 * np.pi  # 计算结束刻度值的角度
    # 结束刻度
    ax.text(
        angle_rad_max,
        axis_radius + 0.08,
        f"{axis_max:.0f}",
        ha="center",
        va="center",
        rotation=np.rad2deg(angle_rad_max) - 90,
        fontsize=28,
    )
    for val in tick_values:  # 遍历所有刻度值
        if val > axis_min and val < axis_max:  # 判断是否为中间的刻度值
            angle_rad = np.pi / 2 - (val / max_scale_value) * 2 * np.pi  # 计算当前刻度值对应的角度
            angle_deg = np.rad2deg(angle_rad)  # 将弧度转换为角度
            ax.text(
                angle_rad,
                axis_radius + 0.08,
                f"{val:.0f}",
                ha="center",
                va="center",
                rotation=angle_deg - 90,
                fontsize=28,
            )

    for cat in categories:  # 遍历每个类别
        anchor_point = (np.pi / 2, radii[cat])  # 定义锚点位置（正上方，对应类别的半径处）
        horizontal_offset_points = -20  # 水平偏移量

        ax.annotate(
            cat,  # 文本内容
            xy=anchor_point,  # 目标点的坐标
            xytext=(horizontal_offset_points, 0),  # 文本坐标
            textcoords="offset points",  # 坐标系
            ha="right",
            va="center",
            fontsize=24,
        )  # 字体大小

    # 在图表的左上角添加标签
    ax.text(0.01, 0.98, "(A)", transform=ax.transAxes, fontsize=28, va="top", ha="left")

    # ax.grid(False)
    ax.spines["polar"].set_visible(False)  # 移除最外层的圆形边框
    ax.set_xticks([])  # 移除默认的角度刻度
    ax.set_yticks([])  # 移除默认的半径刻度
    ax.set_ylim(0, axis_radius + 0.2)  # 设置半径的显示范围

    # 保存
    plt.savefig(f"{output_filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_filename}.pdf", bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# ====================================== 4. 定义数据与绘图========================================
# =========================================================================================
if __name__ == "__main__":
    # 定义数据
    data = {
        "HF": 215,
        "GG": 244,
        "VV": 196,
        "CC": 178,
        "NN": 152,
        "KK": 243,
        "LOI": 222,
        "PHK": 236,
    }
    color_scheme = 21  # 设置要使用的配色方案
    filename = str(OUTPUT_DIR / f"category_comparison_{color_scheme}")
    # 调用函数生成并保存图表
    create_circular_barplot(
        data=data,  # 传入数据
        color_scheme_id=color_scheme,  # 选择的配色方案
        output_filename=filename,  # 输出文件名
        max_scale_value=300,  # 整个圆周代表的最大值
        axis_min=0.0,  # 外圈刻度的最小值
        axis_max=270,  # 设置外圈刻度的最大值
        num_ticks=6,
    )  # 设置外圈刻度的数量

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

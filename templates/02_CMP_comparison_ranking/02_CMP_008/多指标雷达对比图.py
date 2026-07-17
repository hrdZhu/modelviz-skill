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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

mpl.rcParams["font.family"] = "Times New Roman"
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ======================================2.颜色库设置=========================================
# =========================================================================================
COLOR_THEMES = {
    1: {
        "series1": "#582a7f",
        "series2": "#b14156",
        "series3": "#adab2d",
        "center_bg": "#f0ebe2",
        "labels": [
            "#d95f02",
            "#2ca02c",
            "#d62728",
            "#17becf",
            "#17becf",
            "#17becf",
            "#17becf",
            "#17becf",
            "#17becf",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#ff7f0e",
            "#ff7f0e",
            "#9467bd",
            "#d62728",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
        ],
    },
    2: {
        "series1": "#005f73",
        "series2": "#0a9396",
        "series3": "#94d2bd",
        "center_bg": "#e9f5f5",
        "labels": [
            "#001219",
            "#005f73",
            "#0a9396",
            "#94d2bd",
            "#e9d8a6",
            "#ee9b00",
            "#ca6702",
            "#bb3e03",
            "#ae2012",
            "#9b2226",
        ]
        * 3,
    },
    3: {
        "series1": "#f94144",
        "series2": "#f8961e",
        "series3": "#f9c74f",
        "center_bg": "#fff6e6",
        "labels": [
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
        ]
        * 3,
    },
    4: {
        "series1": "#2d6a4f",
        "series2": "#52b788",
        "series3": "#b7e4c7",
        "center_bg": "#f3faf5",
        "labels": [
            "#081c15",
            "#1b4332",
            "#2d6a4f",
            "#40916c",
            "#52b788",
            "#74c69d",
            "#95d5b2",
            "#b7e4c7",
            "#d8f3e5",
            "#004b23",
        ]
        * 3,
    },
    5: {
        "series1": "#e63946",
        "series2": "#457b9d",
        "series3": "#a8dadc",
        "center_bg": "#f1faee",
        "labels": [
            "#e63946",
            "#f1faee",
            "#a8dadc",
            "#457b9d",
            "#1d3557",
            "#e76f51",
            "#f4a261",
            "#e9c46a",
            "#2a9d8f",
            "#264653",
        ]
        * 3,
    },
    6: {
        "series1": "#012a4a",
        "series2": "#013a63",
        "series3": "#2a6f97",
        "center_bg": "#f0f4f8",
        "labels": [
            "#011627",
            "#012a4a",
            "#013a63",
            "#01497c",
            "#2c7da0",
            "#468faf",
            "#61a5c2",
            "#89c2d9",
            "#a9d6e5",
            "#415a77",
        ]
        * 3,
    },
    7: {
        "series1": "#7209b7",
        "series2": "#f72585",
        "series3": "#b5179e",
        "center_bg": "#fceef7",
        "labels": [
            "#560bad",
            "#7209b7",
            "#b5179e",
            "#f72585",
            "#480ca8",
            "#3a0ca3",
            "#3f37c9",
            "#4361ee",
            "#4895ef",
            "#4cc9f0",
        ]
        * 3,
    },
    8: {
        "series1": "#a44200",
        "series2": "#d58936",
        "series3": "#f2c12e",
        "center_bg": "#fff8e8",
        "labels": [
            "#693004",
            "#a44200",
            "#d58936",
            "#f2c12e",
            "#462502",
            "#86572a",
            "#b78953",
            "#e8ba7e",
            "#3c1d01",
            "#c06c16",
        ]
        * 3,
    },
    9: {
        "series1": "#ffadad",
        "series2": "#ffd6a5",
        "series3": "#caffbf",
        "center_bg": "#fdfffc",
        "labels": [
            "#ffadad",
            "#ffd6a5",
            "#fdffb6",
            "#caffbf",
            "#9bf6ff",
            "#a0c4ff",
            "#bdb2ff",
            "#ffc6ff",
            "#ffddd2",
            "#e4c1f9",
        ]
        * 3,
    },
    10: {
        "series1": "#ef476f",
        "series2": "#ffd166",
        "series3": "#06d6a0",
        "center_bg": "#f7f7f7",
        "labels": [
            "#ef476f",
            "#ffd166",
            "#06d6a0",
            "#118ab2",
            "#073b4c",
            "#f94144",
            "#f3722c",
            "#f8961e",
            "#f9c74f",
            "#90be6d",
        ]
        * 3,
    },
    11: {
        "series1": "#212529",
        "series2": "#6c757d",
        "series3": "#adb5bd",
        "center_bg": "#f8f9fa",
        "labels": [
            "#212529",
            "#343a40",
            "#495057",
            "#6c757d",
            "#adb5bd",
            "#ced4da",
            "#dee2e6",
            "#e9ecef",
            "#f8f9fa",
            "#000000",
        ]
        * 3,
    },
    12: {
        "series1": "#6a4c93",
        "series2": "#a06cd5",
        "series3": "#c19ee0",
        "center_bg": "#f5f0fa",
        "labels": [
            "#4d2d79",
            "#6a4c93",
            "#8d6bbd",
            "#a06cd5",
            "#b68fe0",
            "#c19ee0",
            "#d4b7e8",
            "#e9d6f1",
            "#3b225e",
            "#583c87",
        ]
        * 3,
    },
    13: {
        "series1": "#b5838d",
        "series2": "#e5989b",
        "series3": "#ffb4a2",
        "center_bg": "#fff6f4",
        "labels": [
            "#6d6875",
            "#b5838d",
            "#e5989b",
            "#ffb4a2",
            "#ffcda3",
            "#7f5539",
            "#9c6644",
            "#b08968",
            "#ddb892",
            "#e6ccb2",
        ]
        * 3,
    },
    14: {
        "series1": "#ff7b00",
        "series2": "#ffaa00",
        "series3": "#ffdd00",
        "center_bg": "#fff9e6",
        "labels": [
            "#d95f02",
            "#e6780e",
            "#f2911a",
            "#ffaa00",
            "#ffbe33",
            "#ffd266",
            "#ffe599",
            "#fca311",
            "#e85d04",
            "#f48c06",
        ]
        * 3,
    },
    15: {
        "series1": "#344e41",
        "series2": "#588157",
        "series3": "#a3b18a",
        "center_bg": "#f0f4f0",
        "labels": [
            "#283618",
            "#3a5a40",
            "#344e41",
            "#588157",
            "#606c38",
            "#a3b18a",
            "#bc6c25",
            "#dda15e",
            "#fefae0",
            "#2a2a2a",
        ]
        * 3,
    },
    16: {
        "series1": "#ff70a6",
        "series2": "#ff9770",
        "series3": "#ffd670",
        "center_bg": "#fff5f5",
        "labels": [
            "#ff70a6",
            "#ff9770",
            "#ffd670",
            "#e9ff70",
            "#70e0ff",
            "#70a1ff",
            "#8f70ff",
            "#ef70ff",
            "#ff708d",
            "#ffab70",
        ]
        * 3,
    },
    17: {
        "series1": "#00b4d8",
        "series2": "#0077b6",
        "series3": "#03045e",
        "center_bg": "#f0f9ff",
        "labels": [
            "#03045e",
            "#0077b6",
            "#00b4d8",
            "#90e0ef",
            "#caf0f8",
            "#023e8a",
            "#0096c7",
            "#48cae4",
            "#ade8f4",
            "#ffffff",
        ]
        * 3,
    },
    18: {
        "series1": "#800f2f",
        "series2": "#a4133c",
        "series3": "#c9184a",
        "center_bg": "#fde8ef",
        "labels": [
            "#590d22",
            "#800f2f",
            "#a4133c",
            "#c9184a",
            "#ff4d6d",
            "#ff758f",
            "#ff8fa3",
            "#ffb3c1",
            "#ffccd5",
            "#fff0f3",
        ]
        * 3,
    },
    19: {
        "series1": "#55a630",
        "series2": "#80b918",
        "series3": "#aacc00",
        "center_bg": "#f5faef",
        "labels": [
            "#2b9348",
            "#55a630",
            "#80b918",
            "#aacc00",
            "#bfd200",
            "#d4ee00",
            "#dddf00",
            "#eeef20",
            "#ffff3f",
            "#004400",
        ]
        * 3,
    },
    20: {
        "series1": "#6f4e37",
        "series2": "#a07e63",
        "series3": "#c8a88a",
        "center_bg": "#f7f3ef",
        "labels": [
            "#4a3728",
            "#6f4e37",
            "#87674e",
            "#a07e63",
            "#b8967b",
            "#c8a88a",
            "#dcc8b1",
            "#eedfce",
            "#3d2b1f",
            "#5e4532",
        ]
        * 3,
    },
}


# =========================================================================================
# ======================================3.绘图函数=========================================
# =========================================================================================
def create_and_save_radar_chart(categories, data_detected, data_different, data_ratio, palette):
    # 数据的颜色
    series1_color = palette["series1"]
    series2_color = palette["series2"]
    series3_color = palette["series3"]
    center_bg_color = palette["center_bg"]  # 中心背景色
    label_colors = palette["labels"]  # 外圈标注的颜色

    # 获取分类标签的数量
    num_vars = len(categories)
    data_detected_closed = (
        data_detected + data_detected[:1]
    )  # 将第一个数据列的末尾连接到开头，使其闭合
    data_different_closed = (
        data_different + data_different[:1]
    )  # 将第二个数据列的末尾连接到开头，使其闭合
    data_ratio_closed = data_ratio + data_ratio[:1]  # 将第三个数据列的末尾连接到开头，使其闭合
    angles = np.linspace(
        0, 2 * np.pi, num_vars, endpoint=False
    ).tolist()  # 创建一个包含每个分类角度的列表（弧度制）
    angles += angles[:1]  # 将角度列表的末尾也连接到开头，与数据闭合相对应

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    ax.set_theta_offset(np.pi / 2)  # 角度的起始位置
    ax.set_theta_direction(-1)  # 角度的增长方向，顺时针

    ax.grid(True)  # 显示网格线
    yticks = [0.2, 0.4, 0.6, 0.8, 1.0]  # 定义要在哪些半径位置显示刻度
    ax.set_yticks(yticks)  # 设置半径刻度位置
    ax.set_yticklabels([])  # 隐藏半径刻度的数字标签

    gridlines = ax.yaxis.get_gridlines()  # 获取所有的环形网格线对象
    for i, line in enumerate(gridlines):  # 遍历每一条环形网格线
        tick_value = yticks[i]  # 获取当前网格线对应的刻度值
        if tick_value in [0.4, 0.6, 0.8]:
            line.set_visible(False)  # 隐藏这条线

        elif tick_value == 0.2:
            line.set_linestyle("--")  # 线型
            line.set_color("black")  # 颜色
            line.set_linewidth(1.5)  # 线宽为

        elif tick_value == 1.0:
            line.set_linestyle("-")  # 线型
            line.set_color("black")  # 颜色
            line.set_linewidth(2)  # 线宽

    # 绘制雷达图的线
    ax.plot(
        angles,  # 角度
        data_detected_closed,  # 每个角度对应的数据值
        color=series1_color,  # 线的颜色
        linewidth=2,  # 线的宽度
        marker="o",  # 节点的形状
        markersize=8,  # 节点的大小
        zorder=30,
    )  # 绘图顺序

    # 填充雷达图的区域
    ax.fill(
        angles,  # 角度
        data_detected_closed,  # 每个角度对应的数据值
        color=series1_color,  # 填充区域的颜色
        alpha=0.1,  # 填充区域的透明度
        zorder=20,
    )  #
    # 第二列数据
    ax.plot(
        angles,
        data_different_closed,
        color=series2_color,
        linewidth=2,
        marker="o",
        markersize=8,
        zorder=30,
    )
    ax.fill(angles, data_different_closed, color=series2_color, alpha=0.2, zorder=20)
    # 第三列数据
    ax.plot(
        angles,
        data_ratio_closed,
        color=series3_color,
        linewidth=2,
        marker="o",
        markersize=8,
        zorder=30,
    )
    ax.fill(angles, data_ratio_closed, color=series3_color, alpha=0.5, zorder=20)

    # 绘制一个圆形压盖到中心的位置上，制造出空心的效果
    inner_radius = 0.09  # 圆的半径
    # 创建圆形的Patch对象
    center_circle = Circle(
        (0, 0),  # 圆心坐标
        inner_radius,  # 半径
        transform=ax.transData._b,  # 指定坐标变换，确保圆形在数据坐标系中正确显示
        color=center_bg_color,  # 背景颜色
        zorder=2,  # 绘图顺序
        clip_on=False,
    )  # False即使圆超出坐标轴范围也完整绘制
    ax.add_patch(center_circle)  # 将这个圆添加到图表中
    # 创建一个空心虚线圆作为边框
    # 创建另一个圆形 Patch 对象，作为中心圆的虚线边框
    center_circle_border = Circle(
        (0, 0),  # 圆心坐标
        inner_radius,  # 半径
        transform=ax.transData._b,  # 指定坐标变换
        color="black",  # 边框颜色
        linestyle="--",  # 线样式
        linewidth=1,  # 线宽
        fill=False,  # 不填充颜色
        zorder=4.1,
        clip_on=False,
    )
    # 加到图上
    ax.add_patch(center_circle_border)

    ax.set_xticks(angles[:-1])  # 设置外圈标注的角度位置
    ax.set_xticklabels([])  # 隐藏默认的角度刻度标签
    # 遍历外圈标注及其索引
    for i, category in enumerate(categories):
        angle_rad = angles[i]  # 获取当前标注的角度

        visual_angle_deg = np.rad2deg(np.pi / 2 - angle_rad) % 360  # 旋转角度
        if 90 < visual_angle_deg < 270:  # 如果标签在图的左半边
            rotation = visual_angle_deg - 90  # 旋转角度
            horizontal_alignment = "center"  # 水平对齐方式
        else:  # 如果标签在图的右半边
            rotation = visual_angle_deg - 90  # 旋转角度
            horizontal_alignment = "center"  # 水平对齐方式
        # 添加外圈标注
        ax.text(
            angle_rad,  # x坐标
            1.05,  # y坐标
            category,  # 文本内容
            size=14,  # 大小
            color=label_colors[i],  # 颜色
            rotation=rotation,  # 旋转角度
            ha=horizontal_alignment,  # 水平对齐方式
            va="center",  # 垂直对齐方式
            zorder=5,
        )

    ax.set_ylim(0, 1.0)  # 设置半径的范围
    angle_20_percent = (angles[3] + angles[4]) / 2  # 20%标注的角度
    ax.text(angle_20_percent, 0.22, "20%", fontsize=16, ha="center", va="center", zorder=5)

    ax.spines["polar"].set_visible(False)  # 隐藏外圈的边框线
    # 添加小标题
    fig.text(0.1, 0.9, "(A)", fontsize=30, ha="center")

    # 创建一个图例元素的列表
    legend_elements = [
        plt.Line2D(
            [0],  # x坐标，作为占位符
            [0],  # y坐标
            linestyle="-",  # 线条样式
            linewidth=2,  # 线条宽度
            color=series1_color,  # 线条的颜色
            marker="o",  # 标记样式
            markerfacecolor=series1_color,  # 标记的填充颜色
            markersize=12,  # 标记的大小
            label="detected features in ratio",
        ),  # 图例文本
        plt.Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=2,
            color=series2_color,
            marker="o",
            markerfacecolor=series2_color,
            markersize=12,
            label="different features in ratio",
        ),
        plt.Line2D(
            [0],
            [0],
            linestyle="-",
            linewidth=2,
            color=series3_color,
            marker="o",
            markerfacecolor=series3_color,
            markersize=12,
            label="different/detected features in ratio",
        ),
    ]
    # 添加图例
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        fontsize=14,
        handlelength=2.5,
    )

    plt.tight_layout(pad=2)  # 自动调整图表布局
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"radar_{selected_scheme}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"radar_{selected_scheme}.pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# ======================================4.执行部分========================================
# =========================================================================================
if __name__ == "__main__":
    # 更改配色方案
    selected_scheme = 20
    # 绘图数据的路径
    excel_data_path = str(DATA_DIR / "radar_chart_data.xlsx")

    # 读取数据
    df_from_excel = pd.read_excel(excel_data_path)

    # 将DataFrame的每一列转换为列表，以供后续绘图函数使用
    categories_main = df_from_excel["Category"].tolist()
    data_detected_main = df_from_excel["Detected Features"].tolist()
    data_different_main = df_from_excel["Different Features"].tolist()
    data_ratio_main = df_from_excel["Ratio (Different/Detected)"].tolist()

    # 提从颜色库取配色方案
    select_color = COLOR_THEMES.get(selected_scheme, 1)

    # 调用绘图函数
    create_and_save_radar_chart(
        categories=categories_main,  # 外圈标注数据
        data_detected=data_detected_main,  # 第一列数据
        data_different=data_different_main,  # 第二例数据
        data_ratio=data_ratio_main,  # 第三列数据
        palette=select_color,  # 配色方案
    )

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

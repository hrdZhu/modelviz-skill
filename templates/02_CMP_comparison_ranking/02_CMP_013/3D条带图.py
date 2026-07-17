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
import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库=========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: [
        "#A6C1E3",
        "#00579C",
        "#B5D68B",
        "#008026",
        "#FB9A99",
        "#E31A1C",
        "#FDBF6F",
        "#FF7F00",
        "#CAB2D6",
    ],
    2: [
        "#2c3e50",
        "#e74c3c",
        "#ecf0f1",
        "#3498db",
        "#2980b9",
        "#8e44ad",
        "#f1c40f",
        "#d35400",
        "#16a085",
    ],
    3: [
        "#a8aabc",
        "#8f8f8f",
        "#d6cfcb",
        "#c4a69d",
        "#8ba1a3",
        "#6d6875",
        "#b5838d",
        "#e5989b",
        "#ffb4a2",
    ],
    4: [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#6a040f",
        "#9d0208",
        "#d00000",
        "#dc2f02",
    ],
    5: [
        "#ffbe0b",
        "#fb5607",
        "#ff006e",
        "#8338ec",
        "#3a86ff",
        "#06d6a0",
        "#ef476f",
        "#118ab2",
        "#073b4c",
    ],
    6: [
        "#03045e",
        "#023e8a",
        "#0077b6",
        "#0096c7",
        "#00b4d8",
        "#48cae4",
        "#90e0ef",
        "#ade8f4",
        "#caf0f8",
    ],
    7: [
        "#004b23",
        "#006400",
        "#007200",
        "#008000",
        "#38b000",
        "#70e000",
        "#9ef01a",
        "#ccff33",
        "#d9ed92",
    ],
    8: [
        "#ff00ff",
        "#00ffff",
        "#ffff00",
        "#00ff00",
        "#7a04eb",
        "#f72585",
        "#4cc9f0",
        "#4361ee",
        "#3f37c9",
    ],
    9: [
        "#000000",
        "#252525",
        "#525252",
        "#737373",
        "#969696",
        "#bdbdbd",
        "#d9d9d9",
        "#f0f0f0",
        "#ffffff",
    ],
    10: [
        "#ffadad",
        "#ffd6a5",
        "#fdffb6",
        "#caffbf",
        "#9bf6ff",
        "#a0c4ff",
        "#bdb2ff",
        "#ffc6ff",
        "#fffffc",
    ],
    11: [
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
    ],
    12: [
        "#606c38",
        "#283618",
        "#fefae0",
        "#dda15e",
        "#bc6c25",
        "#582f0e",
        "#7f5539",
        "#9c6644",
        "#b08968",
    ],
    13: [
        "#10002b",
        "#240046",
        "#3c096c",
        "#5a189a",
        "#7b2cbf",
        "#9d4edd",
        "#c77dff",
        "#e0aaff",
        "#dec9e9",
    ],
    14: [
        "#d9ed92",
        "#b5e48c",
        "#99d98c",
        "#76c893",
        "#52b69a",
        "#34a0a4",
        "#168aad",
        "#1a759f",
        "#1e6091",
    ],
    15: [
        "#4a4e69",
        "#9a8c98",
        "#c9ada7",
        "#f2e9e4",
        "#22223b",
        "#495057",
        "#ced4da",
        "#e9ecef",
        "#f8f9fa",
    ],
    16: [
        "#0081C8",
        "#FCB131",
        "#000000",
        "#00A651",
        "#EE334E",
        "#808080",
        "#DAA520",
        "#C0C0C0",
        "#B87333",
    ],
    17: [
        "#003049",
        "#d62828",
        "#f77f00",
        "#fcbf49",
        "#eae2b7",
        "#f4a261",
        "#e76f51",
        "#f08080",
        "#ffdab9",
    ],
    18: [
        "#2b2d42",
        "#8d99ae",
        "#edf2f4",
        "#ef233c",
        "#d90429",
        "#003566",
        "#ffc300",
        "#000814",
        "#ffd60a",
    ],
    19: [
        "#cdb4db",
        "#ffc8dd",
        "#ffafcc",
        "#bde0fe",
        "#a2d2ff",
        "#e0c3fc",
        "#dabbfc",
        "#bbd0ff",
        "#b8c0ff",
    ],
    20: [
        "#001219",
        "#005f73",
        "#0a9396",
        "#94d2bd",
        "#e9d8a6",
        "#ee9b00",
        "#ca6702",
        "#bb3e03",
        "#ae2012",
    ],
}


# =========================================================================================
# ======================================3.绘图函数=========================================
# =========================================================================================
def draw_3d_chart(data_dict, years_arr, county_list, color_list):
    num_c = len(county_list)  # 计算数据类别的数量
    fig = plt.figure(figsize=(10, 8), dpi=150)  # 创建画布
    ax = fig.add_subplot(111, projection="3d")  # 添加一个3D子图
    yticks = np.arange(num_c)  # 成Y轴的刻度位置数组
    half_width = 0.3  # 设置3D条带的一半宽度
    for i, county in enumerate(county_list):  # 遍历每一个区域
        zs = data_dict[county]  # 获取当前区域对应的数据值列表
        c = color_list[i % len(color_list)]  # 根据索引循环选择颜色
        for j in range(len(years_arr)):  # 遍历每一个年份点
            # 绘制垂线
            ax.plot(
                [
                    years_arr[j],  # 垂线的X坐标，起点
                    years_arr[j],
                ],  # 设置垂线的X坐标，终点
                [i, i],  # 设置垂线的Y坐标
                [0, zs[j]],  # 设置垂线的Z坐标
                color="black",  # 颜色
                linestyle="-",  # 线型
                linewidth=1,  # 垂线宽度
                alpha=0.9,  # 透明度
                zorder=1,
            )
        for j in range(len(years_arr) - 1):  # 遍历年份区间
            X_segment = np.array(
                [
                    [
                        years_arr[j],  # 定义面的X坐标网格，当前年份
                        years_arr[j + 1],
                    ],  # 定义面的X坐标网格，下一年份
                    [
                        years_arr[j],  # 定义面的X坐标网格
                        years_arr[j + 1],
                    ],
                ]
            )  # 定义面的X坐标网格
            Y_segment = np.array(
                [
                    [
                        i - half_width,  # Y坐标网格，宽度下界
                        i - half_width,
                    ],  # Y坐标网格，宽度下界
                    [
                        i + half_width,  # Y坐标网格，宽度上界
                        i + half_width,
                    ],
                ]
            )  # Y坐标网格，宽度上界

            Z_segment = np.array(
                [
                    [
                        zs[j],  # Z坐标网格，当前数据值
                        zs[j + 1],
                    ],  # Z坐标网格，下一数据值
                    [
                        zs[j],  # Z坐标网格
                        zs[j + 1],
                    ],
                ]
            )  # Z坐标网格
            # 绘制条带
            ax.plot_surface(
                X_segment,  # X坐标
                Y_segment,  # Y坐标
                Z_segment,  # Z坐标
                color=c,  # 颜色
                alpha=0.9,  # 透明度
                shade=False,  # 是否开启阴影效果
                edgecolor="black",  # 边缘线颜色
                zorder=2,
            )
    ax.set_xticks(years_arr)  # 设置X轴的刻度位置为年份数组
    # 设置X轴的刻度标签文本
    ax.set_xticklabels(
        years_arr,  # 数据
        rotation=-20,  # 标签旋转角度
        va="center",  # 垂直对齐方式
        ha="center",  # 水平对齐方式
        fontsize=18,
    )  # 字体大小
    ax.set_xlim(years_arr[0] - 2, years_arr[-1] + 1)  # X轴的显示范围
    ax.set_yticks(yticks)  # 设置Y轴的刻度位置
    # 设置Y轴的刻度标签
    ax.set_yticklabels(
        county_list,  # 数据
        rotation=-90,  # 旋转角度
        va="center",  # 垂直对齐方式
        ha="center",  # 水平对齐方式
        fontsize=18,
    )  # 字体大小
    ax.set_ylim(-1, num_c)  # Y轴的显示范围
    ax.set_zlim(0, 1.0)  # Z轴的显示范围
    # 设置Z轴的标题
    ax.set_zlabel(
        "Social Urbanization (a)",
        fontsize=18,  # 字体大小
        labelpad=12,  # Z轴标题与轴的距离
        rotation=90,
    )  # 标题旋转角度
    ax.tick_params(
        axis="z",  # Z轴刻度参数
        labelsize=18,
    )  # Z轴刻度标签大小
    x_min, x_max = years_arr[0] - 2, years_arr[-1] + 1  # 定义边框的X轴范围
    y_min, y_max = -1, num_c  # 定义边框的Y轴范围
    z_min, z_max = 0, 1.0  # 定义边框的Z轴范围
    frame_color = "black"  # 设置边框颜色
    frame_width = 1.2  # 设置边框宽度
    # 绘制边框线
    ax.plot(
        [x_min, x_min],  # X坐标
        [y_min, y_min],  # Y坐标
        [z_min, z_max],  # Z坐标
        color=frame_color,  # 颜色
        linewidth=frame_width,  # 线宽
        zorder=0,
    )
    ax.plot(
        [x_min, x_min],
        [y_max, y_max],
        [z_min, z_max],
        color=frame_color,
        linewidth=frame_width,
        zorder=0,
    )
    ax.plot(
        [x_min, x_min],
        [y_min, y_max],
        [z_max, z_max],
        color=frame_color,
        linewidth=frame_width,
        zorder=0,
    )
    ax.plot(
        [x_min, x_max],
        [y_max, y_max],
        [z_max, z_max],
        color=frame_color,
        linewidth=frame_width,
        zorder=0,
    )
    ax.plot(
        [x_min, x_min],
        [y_min, y_max],
        [z_min, z_min],
        color=frame_color,
        linewidth=frame_width,
        zorder=0,
    )
    ax.plot(
        [x_min, x_max],
        [y_max, y_max],
        [z_min, z_min],
        color=frame_color,
        linewidth=frame_width,
        zorder=0,
    )

    pane_color = (1.0, 1.0, 1.0, 0.0)  # 设置面板背景颜色，完全透明
    grid_color = "#D0D0D0"  # 设置网格线颜色

    ax.xaxis.pane.set_facecolor(pane_color)  # X轴面板背景色
    ax.xaxis.pane.set_edgecolor(frame_color)  # X轴面板边缘颜色
    ax.xaxis.pane.set_alpha(1)  # X轴面板透明度
    ax.yaxis.pane.set_facecolor(pane_color)  # Y轴面板背景色
    ax.yaxis.pane.set_edgecolor(frame_color)  # Y轴面板边缘颜色
    ax.yaxis.pane.set_alpha(1)  # Y轴面板透明度
    ax.zaxis.pane.set_facecolor(pane_color)  # Z轴面板背景色
    ax.zaxis.pane.set_edgecolor(frame_color)  # Z轴面板边缘颜色
    ax.zaxis.pane.set_alpha(1)  # Z轴面板透明度

    ax.xaxis._axinfo["grid"]["linestyle"] = "-"  # X轴网格线型
    ax.yaxis._axinfo["grid"]["linestyle"] = "-"  # Y轴网格线型
    ax.zaxis._axinfo["grid"]["linestyle"] = "-"  # Z轴网格线型
    ax.xaxis._axinfo["grid"]["color"] = grid_color  # X轴网格颜色
    ax.yaxis._axinfo["grid"]["color"] = grid_color  # Y轴网格颜色
    ax.zaxis._axinfo["grid"]["color"] = grid_color  # Z轴网格颜色

    ax.view_init(elev=30, azim=-60)  # 仰角和方位角
    # 创建图例的矩形色块列表
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color_list[i % len(color_list)], alpha=0.9)
        for i in range(num_c)
    ]
    legend = plt.legend(  # 创建图例对象
        legend_patches,  # 色块句柄
        county_list,  # 标签文本
        loc="upper left",  # 图例的位置
        bbox_to_anchor=(0.1, 0.72),  # 具体位置
        ncol=1,  # 列数
        fontsize=12,
        frameon=True,  # 显示图例边框
        edgecolor="black",  # 图例边框颜色
        fancybox=False,
    )
    legend.get_frame().set_linewidth(1.0)  # 图例边框的线宽
    plt.subplots_adjust(left=-0.05, right=1, top=1.05, bottom=0.001)
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"scheme_{scheme_index}.png"), dpi=300)
    plt.savefig(str(OUTPUT_DIR / f"scheme_{scheme_index}.pdf"), dpi=300)


# =========================================================================================
# ======================================4.执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    excel_path = str(DATA_DIR / "data.xlsx")  # Excel数据文件的路径
    scheme_index = 20  # 要使用的颜色方案
    df = pd.read_excel(excel_path, index_col=0)  # 读取Excel文件
    years = df.index.to_numpy()  # 将年份转换为NumPy数组
    counties = df.columns.tolist()  # 将列转换为列表
    data = df.to_dict(orient="list")  # 将数据框转换为字典格式，列名为键，列表为值
    current_colors = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])  # 颜色方案
    # 调用函数进行绘图
    draw_3d_chart(data, years, counties, current_colors)

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

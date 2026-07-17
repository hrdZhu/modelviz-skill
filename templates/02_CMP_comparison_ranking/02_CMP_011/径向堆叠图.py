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
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Times New Roman"
# =========================================================================================
# ======================================2.颜色库设置=========================================
# =========================================================================================
selected_scheme = 20  # 选择要使用配色方案
COLOR_THEMES = {
    1: [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
        "#e6194b",
    ],
    2: [
        "#0d2b45",
        "#14466b",
        "#1d6292",
        "#277fb8",
        "#319cd4",
        "#3ba8e5",
        "#55b6eb",
        "#70c4f1",
        "#8acff8",
        "#a5dbfe",
        "#0077b6",
        "#0089cc",
        "#009ad1",
        "#00a9d6",
        "#00b8d9",
        "#00c6d9",
        "#00d3d6",
        "#00e0d1",
        "#00eccc",
        "#00f7c4",
        "#6fffd2",
    ],
    3: [
        "#540d0d",
        "#721c12",
        "#902b17",
        "#ae3a1c",
        "#cc4921",
        "#e25e2e",
        "#f07b49",
        "#f99a66",
        "#ffb884",
        "#ffd5a2",
        "#c26a00",
        "#d48000",
        "#e59400",
        "#f5a800",
        "#ffba08",
        "#5c4500",
        "#7a5e00",
        "#997800",
        "#b89200",
        "#d6ac00",
        "#f5c600",
    ],
    4: [
        "#4e79a7",
        "#f28e2c",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc949",
        "#af7aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ab",
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
    ],
    5: [
        "#0d0887",
        "#3a049a",
        "#5c01a6",
        "#7e03a8",
        "#9f12a1",
        "#be2595",
        "#d83c87",
        "#eb566d",
        "#f8765c",
        "#fca65c",
        "#fdc570",
        "#fde386",
        "#f0f921",
        "#e0ea25",
        "#d0db2a",
        "#c0cc2e",
        "#b0be33",
        "#a0b037",
        "#90a23c",
        "#809440",
        "#708645",
    ],
    6: [
        "#ffadad",
        "#ffd6a5",
        "#fdffb6",
        "#caffbf",
        "#9bf6ff",
        "#a0c4ff",
        "#bdb2ff",
        "#ffc6ff",
        "#ffb3ba",
        "#ffdfba",
        "#ffffba",
        "#e6ffba",
        "#bafff0",
        "#bae1ff",
        "#d1baff",
        "#ffbaf2",
        "#ff9e9e",
        "#ffcf9e",
        "#ffff9e",
        "#d6ff9e",
        "#9effe5",
    ],
    7: [
        "#58508d",
        "#716e9b",
        "#898ca9",
        "#a2abb8",
        "#bbc7c6",
        "#d5e2d4",
        "#bc4749",
        "#d16a6c",
        "#e58d8f",
        "#f7b0b1",
        "#f6c5af",
        "#a7c957",
        "#b5d36e",
        "#c3dd85",
        "#d1e79c",
        "#e0f1b3",
        "#003049",
        "#2c536e",
        "#597590",
        "#8598b3",
        "#b2bad5",
    ],
    8: [
        "#002b1f",
        "#003d2d",
        "#004f3b",
        "#006149",
        "#007357",
        "#008565",
        "#009773",
        "#00a981",
        "#00bb8f",
        "#00cd9d",
        "#00dfab",
        "#00f1b9",
        "#33f4c2",
        "#66f6cb",
        "#99f9d5",
        "#ccfbe0",
        "#dffcec",
        "#a5d6a7",
        "#81c784",
        "#66bb6a",
        "#4caf50",
    ],
    9: [
        "#990024",
        "#b5002b",
        "#d10032",
        "#ed0039",
        "#001a99",
        "#0021b5",
        "#0028d1",
        "#002fed",
        "#00994d",
        "#00b55b",
        "#00d169",
        "#00ed77",
        "#995400",
        "#b56400",
        "#d17400",
        "#ed8400",
        "#4d0099",
        "#5b00b5",
        "#6900d1",
        "#7700ed",
        "#009999",
    ],
    10: [
        "#ff00ff",
        "#e600e6",
        "#cc00cc",
        "#b300b3",
        "#990099",
        "#00ffff",
        "#00e6e6",
        "#00cccc",
        "#00b3b3",
        "#009999",
        "#ff0055",
        "#cc0044",
        "#990033",
        "#00ff55",
        "#00cc44",
        "#009933",
        "#5500ff",
        "#4400cc",
        "#330099",
        "#ff5500",
        "#cc4400",
    ],
    11: [
        "#4e342e",
        "#5d4037",
        "#6d4c41",
        "#795548",
        "#8d6e63",
        "#a1887f",
        "#bcaaa4",
        "#d7ccc8",
        "#efebe9",
        "#b99976",
        "#a2886a",
        "#8b775e",
        "#746651",
        "#5d5545",
        "#464438",
        "#30322c",
        "#a47c48",
        "#b58f5c",
        "#c6a270",
        "#d7b584",
        "#e8c898",
    ],
    12: [
        "#d73027",
        "#e65141",
        "#f46d59",
        "#fc8d59",
        "#fdae61",
        "#fee090",
        "#ffffbf",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
        "#fde0ef",
        "#f768a1",
        "#dd3497",
        "#ae017e",
        "#7a0177",
    ],
    13: [
        "#a9b7c0",
        "#b6c2c9",
        "#c4ced3",
        "#d1d9dd",
        "#dfe5e7",
        "#e2d8d5",
        "#d4c8c4",
        "#c7b8b2",
        "#baaa9f",
        "#aca08d",
        "#b8c4bb",
        "#a2ada6",
        "#8c9691",
        "#767f7c",
        "#606867",
        "#c0b8d0",
        "#aca2c2",
        "#988cb4",
        "#8476a6",
        "#706098",
        "#8c7b74",
    ],
    14: [
        "#5f0f40",
        "#721b50",
        "#852860",
        "#993471",
        "#ac4081",
        "#bf4d92",
        "#d259a2",
        "#e565b3",
        "#f871c3",
        "#ff7dd4",
        "#780000",
        "#910000",
        "#ab0000",
        "#c40000",
        "#de0000",
        "#3d348b",
        "#4e43a3",
        "#5e53ba",
        "#6f62d1",
        "#8071e8",
        "#9181ff",
    ],
    15: [
        "#2c0735",
        "#480a47",
        "#650d5a",
        "#81116c",
        "#9e157f",
        "#ba1a93",
        "#d61ea6",
        "#f322ba",
        "#ff40c1",
        "#ff5ec8",
        "#ff7bcf",
        "#ff98d6",
        "#ffb4dc",
        "#ff0054",
        "#ff3377",
        "#ff6699",
        "#ff99bb",
        "#ffccdd",
        "#ffc300",
        "#ffd633",
        "#ffe866",
    ],
    16: [
        "#ef476f",
        "#ffd166",
        "#06d6a0",
        "#118ab2",
        "#073b4c",
        "#ff6b6b",
        "#f0a6ca",
        "#56cfe1",
        "#6d6875",
        "#ff9f1c",
        "#f7d794",
        "#f6b93b",
        "#e55039",
        "#f6b93b",
        "#20bf6b",
        "#1e90ff",
        "#ff4757",
        "#eccc68",
        "#ffa502",
        "#ff6348",
        "#7bed9f",
    ],
    17: [
        "#e0fbfc",
        "#c2dfe3",
        "#a3c2cb",
        "#85a6b2",
        "#67899a",
        "#fde2e4",
        "#fad2e1",
        "#f8c2de",
        "#f6b2db",
        "#f4a2d8",
        "#e2e2e2",
        "#d4d4d4",
        "#c6c6c6",
        "#b8b8b8",
        "#aaaaaa",
        "#d9ed92",
        "#b5e48c",
        "#99d98c",
        "#76c893",
        "#52b69a",
        "#34a0a4",
    ],
    18: [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#a9a9a9",
        "#ffffff",
        "#000000",
    ],
    19: [
        "#f2f2f2",
        "#e6e6e6",
        "#d9d9d9",
        "#cccccc",
        "#bfbfbf",
        "#b3b3b3",
        "#a6a6a6",
        "#999999",
        "#8c8c8c",
        "#808080",
        "#737373",
        "#666666",
        "#595959",
        "#4d4d4d",
        "#404040",
        "#333333",
        "#262626",
        "#1a1a1a",
        "#0d0d0d",
        "#000000",
        "#f8f8f8",
    ],
    20: [
        "#ff8c00",
        "#ffa500",
        "#ffb700",
        "#ffc900",
        "#ffdb00",
        "#ffed00",
        "#d90429",
        "#ef233c",
        "#fb5607",
        "#ffbe0b",
        "#8338ec",
        "#3a86ff",
        "#40916c",
        "#52b788",
        "#95d5b2",
        "#f94144",
        "#f3722c",
        "#f8961e",
        "#f9c74f",
        "#277da1",
        "#6d2e46",
    ],
}


# =========================================================================================
# ======================================4.绘图函数=========================================
# =========================================================================================
def plot_radial_chart_from_file(data_file_path, palette_choice=1):
    selected_colors = COLOR_THEMES.get(palette_choice, 1)  # 提取配色方案

    # 读取数据
    df_map = pd.read_excel(data_file_path, sheet_name="ProteinTissueMap")
    df_order = pd.read_excel(data_file_path, sheet_name="ProteinOrder")

    # b. 通用化地获取列名
    item_col = df_map.columns[0]  # 用于定义画多少个径向的柱子
    category_col = df_map.columns[1]  # 堆叠的层
    order_col = df_order.columns[0]  # 画的顺序

    # 每个径向对应多少层
    item_category_map = df_map.groupby(item_col)[category_col].apply(list).to_dict()
    # 将定义顺序的数据列转换为列表
    ordered_items = df_order[order_col].tolist()

    # 获取所有不重复的类别，并进行排序
    unique_categories = sorted(df_map[category_col].unique())

    # 为每个唯一类别分配一个颜色，如果颜色用完则从头开始循环
    category_colors = {
        category: selected_colors[i % len(selected_colors)]
        for i, category in enumerate(unique_categories)
    }

    num_items = len(ordered_items)  # 径向主的总数
    angle_range = 2 * np.pi * 0.98  # 整个图的总角度，设置一个开口
    angles = np.linspace(0, angle_range, num_items, endpoint=False).tolist()  # 每个径向柱子的角度
    bar_width = (angle_range / num_items) * 0.9  # 计算每个柱子的宽度

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    ax.grid(False)  # 移除网格线
    ax.set_xticks([])  # 移除角度刻度
    ax.set_yticks([])  # 半径刻度

    ax.spines["polar"].set_visible(False)  # 移除最外层的圆形边框

    ax.set_theta_offset(np.pi / 2 - bar_width / 2)  # 设置起始角度
    ax.set_theta_direction(-1)  # 设置方向为顺时针

    # 用于追踪每个径向柱子的最大堆叠高度，以便设置Rlim
    max_stacked_height = 0

    # 绘制堆叠的柱子
    for i, item in enumerate(ordered_items):
        angle = angles[i]  # 获取当前柱子的角度
        categories_for_item = item_category_map.get(item, [])  # 获取当前柱子对应的类别列表

        # 在最内圈绘制一个白色的占位层，形成空白内圈
        ax.bar(
            angle,
            height=0.98,
            width=bar_width,
            bottom=1,
            color="white",
            edgecolor="white",
            linewidth=0.98,
        )

        # 从第二层开始绘真实的数据
        bottom = 2  # 初始化真实数据的堆叠起始半径
        for category in categories_for_item:  # 遍历当前柱子的每一个类别
            color = category_colors.get(
                category, "#000000"
            )  # 获取该类别对应的颜色，如果找不到则默认为黑色
            ax.bar(
                angle,
                height=0.98,
                width=bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.98,
            )
            bottom += 1  # 增加起始半径，为下一个类别做准备，形成堆叠效果

        # 更新当前径向柱子的最大高度
        max_stacked_height = max(max_stacked_height, bottom)

    # 设置极坐标的径向范围
    ax.set_rlim(1, max_stacked_height + 1.5)

    # 添加外圈的标签
    for i, item in enumerate(ordered_items):  # 再次遍历每一个柱子及其索引
        angle_rad = angles[i]  # 获取当前柱子的角度

        bar_length = len(item_category_map.get(item, []))  # 计算当前柱子堆叠的层数
        label_radius = 1 + bar_length + 1.8  # 设置柱子标签的半径，使其在柱子外部
        visual_angle_deg = np.rad2deg(np.pi / 2 - angle_rad) % 360  # 将绘图角度转换为视觉角度

        if 90 < visual_angle_deg < 270:  # 左半侧
            rotation = visual_angle_deg + 180
            alignment = "right"  # 对齐方式
        else:  # 右半侧
            rotation = visual_angle_deg
            alignment = "left"  # 对齐方式

        ax.text(
            angle_rad, label_radius, item, ha=alignment, va="center", rotation=rotation, fontsize=12
        )

    # # 添加标题
    # ax.text(0.05,
    #         0.95,
    #         'Down-regulated DEPs\nin ≥ 8 tissue types',
    #         ha='left',
    #         va='top',
    #         fontsize=20,
    #         color='#008080',
    #         transform=ax.transAxes)

    # 创建图例 ---
    legend_patches = [
        mpatches.Patch(color=color, label=category) for category, color in category_colors.items()
    ]
    # 添加图例
    ax.legend(
        handles=legend_patches,
        title="Tissue",
        title_fontsize="16",
        fontsize="12",
        bbox_to_anchor=(0.9, 0.9),
        frameon=False,
    )

    # plt.tight_layout()  #自动调整子图参数，使之填充整个图像区域

    # 保存
    png_filename = str(OUTPUT_DIR / f"radial_chart_{selected_scheme}.png")
    pdf_filename = str(OUTPUT_DIR / f"radial_chart_{selected_scheme}.pdf")
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.close(fig)


# 主执行模块
if __name__ == "__main__":  # 当该脚本作为主程序直接运行时，执行以下代码
    # 指定绘图数据的路径
    excel_filename = str(DATA_DIR / "radial_chart_data.xlsx")
    # 调用绘图函数
    plot_radial_chart_from_file(excel_filename, palette_choice=selected_scheme)
    print("绘图完成")

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

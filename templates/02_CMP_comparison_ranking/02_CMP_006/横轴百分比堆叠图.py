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
# ====================================== 1. 环境设置 =======================================
# =========================================================================================
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ====================================== 2. 颜色库设置 ==========================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["#5D2126", "#A6302E", "#CD7D4E", "#EFE68A", "#DBEFF9"],
    2: ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
    3: ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],
    4: ["#335c67", "#fff3b0", "#e09f3e", "#9e2a2b", "#540b0e"],
    5: ["#d72631", "#a2d5c6", "#077b8a", "#5c3c92", "#e2d810"],
    6: ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"],
    7: ["#f94144", "#f3722c", "#f8961e", "#f9c74f", "#90be6d"],
    8: ["#54478c", "#2c699a", "#048ba8", "#0db39e", "#16db93"],
    9: ["#0d3b66", "#faf0ca", "#f4d35e", "#ee964b", "#f95738"],
    10: ["#5f0f40", "#9a031e", "#fb8b24", "#e36414", "#0f4c5c"],
    11: ["#22223b", "#4a4e69", "#9a8c98", "#c9ada7", "#f2e9e4"],
    12: ["#606c38", "#283618", "#fefae0", "#dda15e", "#bc6c25"],
    13: ["#1d3557", "#457b9d", "#a8dadc", "#f1faee", "#e63946"],
    14: ["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"],
    15: ["#cdb4db", "#ffc8dd", "#ffafcc", "#bde0fe", "#a2d2ff"],
    16: ["#000000", "#14213d", "#fca311", "#e5e5e5", "#ffffff"],
    17: ["#50514f", "#f25f5c", "#ffe066", "#247ba0", "#70c1b3"],
    18: ["#7400b8", "#6930c3", "#5e60ce", "#5390d9", "#4ea8de"],
    19: ["#386641", "#6a994e", "#a7c957", "#f2e8cf", "#bc4749"],
    20: ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b"],
}

SCHEME_INDEX = 1  # 设置当前使用的颜色方案
current_colors = COLOR_SCHEMES[SCHEME_INDEX]  # 提取颜色


# =========================================================================================
# ====================================== 3. 绘图函数==========================
# =========================================================================================
def draw_and_save_chart(categories, labels, data, colors):
    fig, ax = plt.subplots(figsize=(10, 6))  # 创建画布
    bar_height = 0.75  # 水平柱状图的宽度
    y_pos = np.arange(len(categories))  # 生成Y轴刻度的位置索引
    # 初始化左边距，用于记录每行堆叠柱子的起始X坐标
    left_bottom = np.zeros(len(categories))
    # 保存每一段的右边缘坐标，用于后续画连接线
    segments_right_edges = np.zeros((len(categories), len(labels)))

    # 循环绘制每一层堆叠柱状图
    for i, (label, color) in enumerate(zip(labels, colors)):  # 遍历每个标签和颜色
        values = data[:, i]  # 提取当前细分项在所有类别中的数值
        # 绘制水平条形
        bars = ax.barh(
            y_pos,
            values,
            height=bar_height,
            left=left_bottom,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label.replace("\n", " "),
            zorder=0,
        )
        # 记录当前段的右边缘位置
        segments_right_edges[:, i] = left_bottom + values

        for bar, val in zip(bars, values):  # 遍历当前生成的每个柱子对象及其数值
            text_color = (
                "white" if i < 3 else "black"
            )  # 根据层级决定文字颜色：前3层用白色，否则用黑色
            # 在条形中心添加数值标注
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                va="center",
                ha="center",
                color=text_color,
                fontsize=11,
                zorder=15,
            )
        # 更新下一次绘制的左起点
        left_bottom += values
    # 绘制连接线
    for row in range(len(categories) - 1):  # 遍历每一行类别
        for col in range(len(labels)):  # 遍历列
            x1 = segments_right_edges[row, col]  # 获取当前行对应柱子片段的右边缘X坐标
            y1 = y_pos[row] + bar_height / 2  # 当前行柱子的上边缘Y坐标
            x2 = segments_right_edges[row + 1, col]  # 下一行对应柱子片段的右边缘X坐标
            y2 = y_pos[row + 1] - bar_height / 2  # 下一行柱子的下边缘Y坐标
            # 绘制连接两点的黑色直线
            ax.plot([x1, x2], [y1, y2], color="black", linewidth=1, zorder=10, clip_on=False)

    ax.set_yticks(y_pos)  # 设置Y轴的主刻度
    ax.set_yticklabels(categories, fontsize=12)  # 设置Y轴的刻度标注文本
    ax.set_xlabel("Proportion of GHG emissions (%)", fontsize=12)  # 设置X轴标题
    ax.set_xlim(0, 100)  # X轴显示范围
    ax.tick_params(axis="x", labelsize=12)  # x轴的刻度标注文本
    # 设置背景横向虚线
    minor_locs = np.arange(len(categories) - 1) + 0.5  # Y轴次刻度位置
    ax.set_yticks(minor_locs, minor=True)  # 设置Y轴次刻度
    ax.grid(
        which="minor", axis="y", linestyle="--", alpha=0.7, color="gray", zorder=0
    )  # 在次刻度处绘制Y轴背景虚线
    # 去掉指定边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # 设置刻度参数
    ax.tick_params(
        axis="y", which="major", left=True, right=False, length=5, width=1, direction="out"
    )
    ax.tick_params(axis="x", direction="out")

    handles, plot_labels = ax.get_legend_handles_labels()  # 获取当前图表的图例句柄和标签
    # 添加图例
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=12,
        handlelength=1.0,
        handleheight=1.0,
    )

    plt.tight_layout()  # 自动调整布局
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"chart_{SCHEME_INDEX}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"chart_{SCHEME_INDEX}.pdf"), format="pdf", bbox_inches="tight")


# =========================================================================================
# ====================================== 4. 数据分析与执行 =================================
# =========================================================================================
if __name__ == "__main__":
    categories = ["BS", "S1", "S2", "S3", "S4", "S5"]  # Y轴类别列表
    labels = [
        "Rice",
        "Agricultural\nland",
        "Diesel use",
        "Irrigation",
        "Indirect",
    ]  # 堆叠块的标签列表
    # 定义数据矩阵，每行对应一个categories，每列对应一个labels
    data = np.array(
        [
            [42.96, 9.57, 16.38, 16.60, 14.49],
            [38.28, 10.21, 17.49, 18.55, 15.47],
            [43.07, 9.35, 16.42, 16.64, 14.53],
            [44.19, 9.85, 16.85, 14.21, 14.90],
            [45.65, 10.17, 13.92, 17.63, 12.63],
            [42.36, 11.02, 15.48, 17.10, 14.04],
        ]
    )
    draw_and_save_chart(categories, labels, data, current_colors)  # 调用函数进行绘图并保存

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

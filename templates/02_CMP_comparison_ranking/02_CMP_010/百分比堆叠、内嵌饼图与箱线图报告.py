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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2.颜色库=========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["#7b97c3", "#e8a97d", "#f7e193", "#e9a7b9", "#92b995"],
    2: ["#2c3e50", "#e74c3c", "#ecf0f1", "#3498db", "#2980b9"],
    3: ["#a8aabc", "#8f8f8f", "#d6cfcb", "#c4a69d", "#8ba1a3"],
    4: ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    5: ["#ffbe0b", "#fb5607", "#ff006e", "#8338ec", "#3a86ff"],
    6: ["#03045e", "#0077b6", "#00b4d8", "#90e0ef", "#caf0f8"],
    7: ["#ef476f", "#ffd166", "#06d6a0", "#118ab2", "#073b4c"],
    8: ["#ff00ff", "#00ffff", "#ffff00", "#00ff00", "#7a04eb"],
    9: ["#222222", "#444444", "#aaaaaa", "#dddddd", "#d9534f"],
    10: ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff"],
    11: ["#540d6e", "#ee4266", "#ffd23f", "#3bceac", "#0ead69"],
    12: ["#606c38", "#283618", "#fefae0", "#dda15e", "#bc6c25"],
    13: ["#3a0ca3", "#4361ee", "#4cc9f0", "#f72585", "#7209b7"],
    14: ["#d9ed92", "#b5e48c", "#99d98c", "#76c893", "#52b69a"],
    15: ["#4a4e69", "#9a8c98", "#c9ada7", "#f2e9e4", "#22223b"],
    16: ["#d00000", "#ffba08", "#3f88c5", "#032b43", "#136f63"],
    17: ["#003049", "#d62828", "#f77f00", "#fcbf49", "#eae2b7"],
    18: ["#2b2d42", "#8d99ae", "#edf2f4", "#ef233c", "#d90429"],
    19: ["#cdb4db", "#ffc8dd", "#ffafcc", "#bde0fe", "#a2d2ff"],
    20: ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6"],
}


# =========================================================================================
# ======================================3.内嵌的3D饼图绘制函数=========================================
# =========================================================================================
def draw_3d_pie(parent_ax, data, palette, title="Global"):
    # 创建一个嵌入式坐标轴
    ax_pie = parent_ax.inset_axes(
        [
            0.55,  # x起点
            0.08,  # y起点
            0.42,  # 宽
            0.42,
        ]
    )  # 高
    explode = [0.08] * len(data)  # 饼图分裂效果
    start_angle = -240  # 绘制的起始角度
    aspect_ratio = 0.5  # 纵横比
    thickness_factor = 0.01  # 每一层饼图的厚度因子
    for i in range(1, 35):  # 循环绘制饼图，模拟3D厚度
        ax_pie.pie(
            data,  # 数据
            colors=palette,  # 颜色
            explode=explode,  # 分裂属性
            startangle=start_angle,  # 起始角度
            radius=1.0,  # 半径
            counterclock=False,  # 顺时针绘制
            center=(0, -i * thickness_factor),  # 向下移动中心点以模拟厚度堆叠
            wedgeprops={"alpha": 1, "edgecolor": "none"},
        )  # 不透明且无边框
    _, _, autotexts = ax_pie.pie(
        data,  # 绘制最顶层的饼图
        colors=palette,  # 颜色
        explode=explode,  # 分裂属性
        startangle=start_angle,  # 起始角度
        autopct="%1.1f%%",  # 百分比显示格式
        pctdistance=0.7,  # 百分比标签距离圆心的距离
        counterclock=False,  # 顺时针绘制
        radius=1.0,  # 半径
        wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
    )  # 扇形边缘为白色细线
    for text in autotexts:
        text.set_fontsize(18)  # 设置所有百分比文本的字体大小
    # 创建一个矩形对象作为外框
    rect = Rectangle(
        (-1.25, -1.8),  # 左下角坐标
        2.4,  # 宽度
        3.5,  # 高度
        fill=False,  # 不填充颜色
        edgecolor="black",  # 边框颜色
        linewidth=2,  # 边框线宽
        transform=ax_pie.transData,  # 坐标系
        clip_on=False,
    )  # 允许矩形超出坐标轴边界显示
    ax_pie.add_patch(rect)  # 将矩形添加到饼图坐标轴中
    # 设置标题文本
    ax_pie.text(
        -1.2,  # X坐标
        2.45 * aspect_ratio,  # Y坐标
        title,  # 内容
        fontweight="bold",  # 字体加粗
        fontsize=20,  # 字体大小
        transform=ax_pie.transData,
    )  # 坐标系
    ax_pie.set_aspect(aspect_ratio)  # 应用纵横比设置
    ax_pie.axis("off")  # 隐藏刻度和边框


# =========================================================================================
# ====================================== 4. 主图绘制函数 =========================================
# =========================================================================================
def plot_analysis_by_scheme(scheme_id):
    current_colors = COLOR_SCHEMES[scheme_id]  # 获取颜色方案
    fig = plt.figure(figsize=(24, 20), dpi=120)  # 创建画布
    a_left, a_right = 0.02, 0.70  # 区域A（左侧）的左右边界位置比例
    b_left, b_right = 0.71, 0.98  # 区域B（右侧）的左右边界位置比例
    title_h = 0.045  # 标题栏的高度比例
    top_y = 1.05  # 顶部的Y坐标基准
    # 背景颜色矩形
    # 左侧标题背景
    fig.patches.extend(
        [
            Rectangle(
                (a_left, top_y),  # 左下角坐标
                a_right - a_left,  # 宽度
                title_h,  # 高度
                transform=fig.transFigure,  # 图形坐标系
                color="#eef5fb",  # 填充颜色
                zorder=1,
            ),
            # 右侧标题背景
            Rectangle(
                (b_left, top_y),
                b_right - b_left,
                title_h,
                transform=fig.transFigure,
                color="#eef5fb",
                zorder=1,
            ),
        ]
    )
    # 设置左侧大标题的X坐标（居中）
    fig.text(
        (a_left + a_right) / 2,  # X坐标
        top_y + title_h / 2,  # Y坐标
        "(A) Global & TOP 10 countries",  # 内容
        ha="center",  # 水平居中
        va="center",  # 垂直居中
        fontsize=30,  # 字体大小
        fontweight="bold",  # 字体加粗
    )
    # 设置右侧大标题的X坐标
    fig.text(
        (b_left + b_right) / 2,  # X坐标
        top_y + title_h / 2,  # Y坐标
        "(B) Sectors",  # 标题文本
        ha="center",  # 水平对齐
        va="center",  # 垂直对齐
        fontsize=30,  # 字体大小
        fontweight="bold",  # 加粗
    )
    # 创建大外框矩形
    main_border = Rectangle(
        (a_left, 0.01),
        b_right - a_left,  # 宽度
        top_y + title_h - 0.05,  # 高度
        transform=fig.transFigure,  # 使用图形坐标系
        fill=False,  # 不填充
        edgecolor="black",  # 黑色边框
        linewidth=4,  # 线宽
        zorder=10,
    )
    fig.patches.append(main_border)  # 添加外框到图形
    # 创建网格布局
    gs = gridspec.GridSpec(
        4,  # 行
        3,  # 列
        figure=fig,  # 归属于当前figure
        width_ratios=[1, 1, 0.8],  # 列宽比例
        left=0.07,  # 左边距
        right=0.975,  # 右边距
        bottom=0.06,  # 下边距
        top=top_y - 0.035,  # 上边距
        wspace=0.15,  # 子图水平间距
        hspace=0.15,
    )  # 子图垂直间距

    years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]  # 分析的年份列表

    for i, year in enumerate(years):  # 遍历每一年份进行绘图
        row, col = divmod(i, 2)  # 计算当前年份对应的网格行和列索引
        ax = fig.add_subplot(gs[row, col])  # 在指定网格位置添加子图
        df = df_stacked[df_stacked["Year"] == year].sort_values("Total")  # 筛选当年数据并按总值排序
        # 计算当年总和
        current_year_sums = df[sectors].sum()

        # 将求和后的Series转换为列表，用于绘制饼图
        global_shares = current_year_sums.values
        left_vals = np.zeros(len(df))  # 初始化堆叠条形图的起始左侧位置
        # 遍历每个部分扇区
        for s_idx, s in enumerate(sectors):
            # 绘制水平条形图
            ax.barh(
                df["Country"],  # Y轴
                df[s],  # X轴
                left=left_vals,  # 条形的起始位置
                color=current_colors[s_idx],  # 颜色
                height=0.7,
            )  # 条形高度
            left_vals += df[s]  # 更新左侧位置，为下一层堆叠做准备
        current_max = df["Total"].max()  # 获取当前数据的最大总值
        ax.set_xlim(0, current_max * 1.3)  # 设置X轴范围
        for y_idx, val in enumerate(df["Total"]):  # 遍历总值以便添加数值标签
            ax.text(
                val + current_max * 0.02,  # 标签X坐标
                y_idx,  # 标签Y坐标
                f"{val:.1f}",  # 标签文本
                va="center",  # 垂直居中
                ha="left",  # 左对齐
                fontsize=20,
            )  # 字体大小
        # 设置子图标题
        ax.set_title(
            f"({chr(97 + i)}) {year}",  # 自动生成a, b, c序号
            loc="left",  # 左对齐
            fontsize=26,  # 字体大小
            fontweight="bold",  # 加粗
            pad=8,
        )  # 标题与图表的间距
        ax.spines[["top", "right"]].set_visible(False)  # 隐藏顶部和右侧的边框轴
        ax.tick_params(
            direction="in",  # 刻度线向内
            labelsize=24,
        )  # 刻度标签字体大小
        ax.set_xlabel(
            "(Mt)",  # X轴标签
            loc="right",  # 右
            fontsize=24,
        )  # 字体大小
        for spine in ax.spines.values():  # 遍历所有边框轴
            spine.set_linewidth(3)  # 设置边框线
        draw_3d_pie(ax, global_shares, current_colors)  # 调用函数在当前子图中绘制3D饼图
    # 在原有网格的第3列中创建嵌套网格
    gs_right = gridspec.GridSpecFromSubplotSpec(
        5,  # 行
        1,  # 列
        subplot_spec=gs[:, 2],  # 指定放置在gs的第3列所有行区域
        hspace=0.35,
    )  # 垂直间距
    # 遍历绘制右侧图表
    for i, sector_name in enumerate(sectors):
        ax_b = fig.add_subplot(gs_right[i])  # 添加子图到右侧嵌套网格
        target_years = [1990, 2000, 2010, 2020]  # 定义右侧图表显示的特定年份
        color_b = current_colors[i]  # 颜色
        ax_b.set_title(
            sector_name,  # 子图标题
            loc="right",  # 右对齐
            fontsize=26,  # 字体大小
            pad=12,
        )  # 间距
        sector_data = df_scatter[df_scatter["Sector"] == sector_name]  # 筛选数据
        if not sector_data.empty:  # 如果数据不为空
            max_val = sector_data["Value"].max()  # 获取最大值
            top_limit = max_val * 1.1 if max_val > 0 else 1.0  # Y轴上限
            ax_b.set_ylim(0, top_limit)  # 设置Y轴范围
        else:
            ax_b.set_ylim(0, 1)  # 如果无数据，设置默认Y轴范围

        for j, yr in enumerate(target_years):  # 遍历特定年份绘制箱型图元素
            vals = sector_data[sector_data["Year"] == yr]["Value"].values  # 获取当年数值
            if len(vals) == 0:
                continue  # 如果无数据则跳过
            q1, median, q3 = np.percentile(vals, [25, 50, 75])  # 计算四分位数和中位数
            low, high = np.min(vals), np.max(vals)  # 获取最小值和最大值
            box_w = 2.5  # 设置箱体的一半宽度
            rect = Rectangle(
                (yr - box_w, q1),  # 创建箱体矩形
                box_w,  # 宽度
                q3 - q1,  # 高度
                facecolor="none",  # 无填充颜色
                edgecolor=color_b,  # 边框颜色
                lw=3,
            )  # 线宽
            ax_b.add_patch(rect)  # 添加箱体到图表
            ax_b.plot(
                [yr - box_w, yr],  # 中位线X坐标
                [median, median],  # 中位线Y坐标
                color=color_b,  # 颜色
                lw=3,
            )  # 线宽
            ax_b.plot(
                [yr, yr],  # 上须竖线X坐标
                [q3, high],  # 上须竖线Y坐标
                color=color_b,  # 颜色
                lw=3,
            )  # 线宽
            ax_b.plot(
                [yr - box_w, yr],  # 最大值横线X坐标
                [high, high],  # 最大值横线Y坐标
                color=color_b,  # 颜色
                lw=3,
            )  # 线宽
            ax_b.plot(
                [yr, yr],  # 下须竖线X坐标
                [q1, low],  # 下须竖线Y坐标
                color=color_b,  # 颜色
                lw=3,
            )  # 线宽
            ax_b.plot(
                [yr - box_w, yr],  # 最小值横线X坐标
                [low, low],  # 最小值横线Y坐标
                color=color_b,  # 颜色
                lw=3,
            )  # 线宽
            for spine in ax_b.spines.values():  # 遍历边框
                spine.set_linewidth(3)  # 设置边框线宽
            x_jitter = yr + np.random.uniform(
                1,  # 生成散点图的X轴随机抖动
                1.5,  # 抖动范围
                size=len(vals),
            )  # 数量
            ax_b.scatter(
                x_jitter,  # 散点图X坐标
                vals,  # Y坐标
                s=14,  # 点的大小
                color=color_b,  # 颜色
                marker="x",  # 标记形状
                alpha=0.8,  # 透明度
                lw=3,
            )  # 线宽
            ax_b.scatter(
                yr,  # 均值点X坐标
                np.mean(vals),  # 均值点Y坐标
                color="black",  # 颜色
                s=25,  # 点的大小
                zorder=5,
            )
        ax_b.set_xticks(target_years)  # 设置X轴刻度
        ax_b.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )  # 设置Y轴标签格式
        ax_b.tick_params(
            direction="in",  # 刻度向内
            labelsize=24,
        )  # 刻度字体大小

    # 分隔线
    ax_sep = plt.Line2D(
        [b_left - 0.005, b_left - 0.005],  # 设置X坐标
        [0.01, top_y + title_h - 0.05],  # Y坐标
        transform=fig.transFigure,  # 使用图形坐标系
        color="black",  # 黑色
        linestyle="--",  # 样式
        linewidth=4,  # 线宽
        zorder=11,
    )
    fig.lines.append(ax_sep)  # 添加线条到图形
    handles = [
        plt.Rectangle(
            (0, 0),  # 创建自定义图例句柄，矩形块
            1,  # 宽
            1,  # 高
            color=current_colors[idx],
        )
        for idx in range(5)
    ]  # 为每种颜色生成一个矩形
    fig.legend(
        handles,  # 图例句柄
        sectors,  # 图例文本
        loc="upper left",  # 图例位置
        bbox_to_anchor=(0.38, 0.28),  # 位置
        ncol=1,  # 列数
        frameon=False,  # 不显示图例边框
        fontsize=36,  # 字体大小
        title="Legend",  # 图例标题
        title_fontproperties={"weight": "bold", "size": 42},  # 图例标题属性
        alignment="left",
    )  # 图例标题相对位置
    # 保存
    save_png_path = str(OUTPUT_DIR / "plot_scheme_{}.png").format(scheme_id)
    save_pdf_path = str(OUTPUT_DIR / "plot_scheme_{}.pdf").format(scheme_id)
    plt.savefig(save_png_path, dpi=300, bbox_inches="tight")
    plt.savefig(save_pdf_path, bbox_inches="tight")


# =========================================================================================
# ======================================5.执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    file_path = str(DATA_DIR / "data.xlsx")  # 文件路径
    # 读取本地数据
    df_stacked = pd.read_excel(file_path, sheet_name="StackedBarData")
    df_scatter = pd.read_excel(file_path, sheet_name="SectorDistribution")
    # 定义分区名称
    sectors = [
        "Power Industry",
        "Building",
        "Transport",
        "Other Industrial Combustion",
        "Other Sectors",
    ]
    scheme_index = 20  # 选择颜色方案ID
    # 调用主绘图函数
    plot_analysis_by_scheme(scheme_index)

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

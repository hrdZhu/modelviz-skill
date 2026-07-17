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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

sns.set_style("ticks")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ======================================2.边缘密度图样式库=======================================
# =========================================================================================
STYLE_SCHEMES = {
    1: {"type": "kde", "params": {"fill": True, "alpha": 0.4, "linewidth": 1.5}},
    2: {"type": "kde", "params": {"fill": False, "linewidth": 2.5}},
    3: {"type": "kde", "params": {"fill": True, "alpha": 1.0, "linewidth": 0}},
    4: {"type": "kde", "params": {"fill": True, "alpha": 0.2, "linewidth": 1, "linestyle": "--"}},
    5: {"type": "hist", "params": {"element": "bars", "multiple": "dodge", "alpha": 0.5}},
    6: {"type": "hist", "params": {"element": "bars", "multiple": "stack", "alpha": 0.6}},
    7: {"type": "hist", "params": {"element": "bars", "multiple": "layer", "alpha": 0.4}},
    8: {"type": "hist", "params": {"element": "step", "fill": False, "linewidth": 2}},
    9: {"type": "hist", "params": {"element": "step", "fill": True, "alpha": 0.4}},
    10: {"type": "hist", "params": {"element": "poly", "fill": True, "alpha": 0.5}},
    11: {
        "type": "hist",
        "params": {"element": "poly", "fill": False, "linewidth": 2, "linestyle": ":"},
    },
    12: {"type": "hist", "params": {"element": "poly", "multiple": "stack", "fill": True}},
    13: {"type": "kde", "params": {"bw_adjust": 0.2, "fill": True, "alpha": 0.4}},
}
# =========================================================================================
# ======================================3.颜色库=======================================
# =========================================================================================
COLOR_SCHEMES = {
    1: {"Day": "#8CDD68", "Night": "#56B4E9"},
    2: {"Day": "#FF9999", "Night": "#9999FF"},
    3: {"Day": "#FFA07A", "Night": "#20B2AA"},
    4: {"Day": "#FFD700", "Night": "#8A2BE2"},
    5: {"Day": "#2E8B57", "Night": "#CD5C5C"},
    6: {"Day": "#00CED1", "Night": "#FF4500"},
    7: {"Day": "#778899", "Night": "#BC8F8F"},
    8: {"Day": "#3CB371", "Night": "#6A5ACD"},
    9: {"Day": "#F08080", "Night": "#4682B4"},
    10: {"Day": "#DAA520", "Night": "#556B2F"},
    11: {"Day": "#FF69B4", "Night": "#1E90FF"},
    12: {"Day": "#D2691E", "Night": "#008080"},
    13: {"Day": "#BDB76B", "Night": "#483D8B"},
    14: {"Day": "#E9967A", "Night": "#8FBC8F"},
    15: {"Day": "#CD853F", "Night": "#4169E1"},
    16: {"Day": "#DC143C", "Night": "#00BFFF"},
    17: {"Day": "#800080", "Night": "#FFD700"},
    18: {"Day": "#A0522D", "Night": "#00FA9A"},
    19: {"Day": "#FF6347", "Night": "#40E0D0"},
    20: {"Day": "#C71585", "Night": "#7B68EE"},
}
# =========================================================================================
# ======================================4.形状标记库=======================================
# =========================================================================================
MARKER_SCHEMES = {
    0: {"Day": "o", "Night": "o"},
    1: {"Day": r"$\spadesuit$", "Night": r"$\heartsuit$"},
    2: {"Day": r"$\clubsuit$", "Night": r"$\diamondsuit$"},
    3: {"Day": r"$\star$", "Night": r"$\bullet$"},
    4: {"Day": r"$\oplus$", "Night": r"$\otimes$"},
    5: {"Day": r"$\alpha$", "Night": r"$\beta$"},
    6: {"Day": r"$\gamma$", "Night": r"$\delta$"},
    7: {"Day": r"$\Delta$", "Night": r"$\nabla$"},
    8: {"Day": r"$\theta$", "Night": r"$\phi$"},
    9: {"Day": r"$\lambda$", "Night": r"$\mu$"},
    10: {"Day": r"$\odot$", "Night": r"$\oslash$"},
    11: {"Day": r"$\leftarrow$", "Night": r"$\rightarrow$"},
    12: {"Day": r"$\uparrow$", "Night": r"$\downarrow$"},
    13: {"Day": r"$\infty$", "Night": r"$\propto$"},
    14: {"Day": r"$\cup$", "Night": r"$\cap$"},
    15: {"Day": r"$\top$", "Night": r"$\bot$"},
    16: {"Day": r"$\angle$", "Night": r"$\measuredangle$"},
    17: {"Day": r"$\sharp$", "Night": r"$\flat$"},
    18: {"Day": r"$\S$", "Night": r"$\P$"},
    19: {"Day": r"$\Sigma$", "Night": r"$\Pi$"},
    20: {"Day": r"$\omega$", "Night": r"$\Omega$"},
}


# =========================================================================================
# ======================================5.绘图函数=======================================
# =========================================================================================
def draw_joint_plot(df, style_id, color_id, marker_id):
    style_settings = STYLE_SCHEMES.get(style_id, STYLE_SCHEMES[1])  # 获取样式配置
    colors = COLOR_SCHEMES.get(color_id, COLOR_SCHEMES[1])  # 获取颜色方案
    markers = MARKER_SCHEMES.get(marker_id, MARKER_SCHEMES[1])  # 获取标记符号方案
    g = sns.JointGrid(
        data=df, x="TD", y="AD", height=8, ratio=4
    )  # 设置数据、x轴、y轴、图像高度和主图与边缘图的比例
    g.ax_joint.set_xlim((10, 35))  # x轴的显示范围
    g.ax_joint.set_ylim((10, 35))  # y轴的显示范围

    # 循环绘制每一组数据
    for group, color in colors.items():  # 遍历颜色字典中的每一组
        subset = df[df["Group"] == group]  # 从数据中筛选出当前分组的数据子集
        if subset.empty:  # 如果子集为空
            continue  # 跳过当前循环
        x_data = subset["TD"]  # x轴数据
        y_data = subset["AD"]  # y轴数据
        marker_shape = markers.get(group, "o")  # 获取当前分组对应的标记形状
        # 绘制散点图和回归线
        sns.regplot(
            data=subset,  # 数据源
            x="TD",  # x轴列名
            y="AD",  # y轴列名
            ax=g.ax_joint,  # 绘制的目标坐标轴
            color=color,  # 绘图颜色
            marker=marker_shape,  # 散点标记形状
            scatter_kws={
                "s": 80,  # 散点大小
                "alpha": 0.8,  # 填充透明度
                # 'edgecolor': 'white',  #外圈颜色
                # 'linewidths': 1.0  #外圈粗细
            },
            line_kws={"linewidth": 2},  # 回归线参数
            truncate=False,  # 回归线不截断
            ci=95,  # 95%置信区间
        )

        # 绘制边缘分布图
        if style_settings["type"] == "kde":  # 如果配置的样式类型是核密度估计
            sns.kdeplot(
                data=subset,  # 数据源
                x="TD",  # 顶部边缘图的分布
                ax=g.ax_marg_x,  # 目标坐标轴
                color=color,  # 颜色
                **style_settings["params"],
            )  # 其他参数
            sns.kdeplot(
                data=subset,
                y="AD",  # 右侧边缘图分布
                ax=g.ax_marg_y,  # 目标坐标轴，右侧边缘轴
                color=color,  # 颜色
                **style_settings["params"],
            )  # 其他参数
        # 进行线性回归，获取斜率、截距、相关系数、p值和标准误
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)

        y_pred = slope * x_data + intercept  # 根据回归方程计算预测值 y
        r_squared = r2_score(y_data, y_pred)  # 计算R2
        rmse = np.sqrt(((y_data - y_pred) ** 2).mean())  # RMSE
        mae = np.abs(y_data - y_pred).mean()  # MAE
        sign = "+" if intercept >= 0 else "-"  # 判断截距的正负，用于生成公式字符串
        formula = f"y = {slope:.2f}x {sign} {abs(intercept):.2f}"  # 格式化回归方程字符串
        p_text = "< 0.001" if p_value < 0.001 else f"= {p_value:.3f}"  # 格式化P值显示文本
        label_text = (
            f"{group}\n"  # 组名
            f"{formula}\n"  # 回归公式
            f"$R^2$ = {r_squared:.3f}, p {p_text}\n"  # R2和P值
            f"RMSE = {rmse:.3f}, MAE = {mae:.3f}"  # RMSE\MAE
        )
        if group == "Day":  # 如果组名为Day
            pos_x, pos_y = 0.04, 0.98  # 位置坐标
            align_ha, align_va = "left", "top"  # 对齐方式
        else:
            pos_x, pos_y = 0.98, 0.04  # 坐标
            align_ha, align_va = "right", "bottom"  # 对齐方式
        # 添加文本
        g.ax_joint.text(
            x=pos_x,
            y=pos_y,  # 坐标
            s=label_text,  # 文本内容
            transform=g.ax_joint.transAxes,  # 使用轴坐标系
            color=color,  # 文本颜色
            fontsize=12,  # 字体大小
            fontweight="bold",  # 字体加粗
            ha=align_ha,  # 水平对齐方式
            va=align_va,  # 垂直对齐方式
            linespacing=1.5,  # 设置行间距
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="beige", edgecolor="gray", alpha=0.8
            ),  # 圆角，米色背景，灰色边框，透明度
        )
    if style_settings["type"] == "hist":  # 是否为直方图
        sns.histplot(
            data=df,  # 数据源
            x="TD",  # x轴
            hue="Group",  # 根据分组着色
            palette=colors,  # 配色
            ax=g.ax_marg_x,  # 指定绘制的目标坐标轴
            legend=False,  # 关闭图例显示
            kde=True,  # 在直方图上叠加核密度估计曲线
            line_kws={"linewidth": 1.5},  # 设置叠加的KDE曲线的线宽
            **style_settings["params"],  # 其他参数
        )
        sns.histplot(
            data=df,
            y="AD",
            hue="Group",
            palette=colors,
            ax=g.ax_marg_y,
            legend=False,
            kde=True,
            line_kws={"linewidth": 1.5},
            **style_settings["params"],
        )
    # 设置主图 x 轴 y轴的标签及字体大小
    g.set_axis_labels("TD", "AD", fontsize=20, fontweight="bold")
    # 控制刻度线样式
    g.ax_joint.tick_params(which="major", width=2.0, length=4.0, labelsize=16, direction="in")
    # 遍历主图的刻度标签并设置为粗体
    for label in g.ax_joint.get_xticklabels() + g.ax_joint.get_yticklabels():
        label.set_fontweight("bold")
    # 配置顶部边缘图的刻度参数
    g.ax_marg_x.tick_params(
        axis="both",  # 同时作用于X轴和Y轴
        which="major",  # 针对主刻度
        left=False,  # 隐藏左侧的刻度线
        bottom=True,  # 显示下侧的刻度线
        top=False,  # 隐藏上侧的刻度线
        right=False,  # 隐藏右侧的刻度线
        width=2.0,
        length=4.0,  # 刻度线的粗细，长度
        direction="out",  # 方向为朝外
        labelleft=False,  # 不显示数字
        labelbottom=False,  # 不显示数字
    )
    g.ax_marg_y.tick_params(
        axis="both",
        which="major",
        left=True,
        bottom=False,
        top=False,
        right=False,
        width=2.0,
        length=4.0,
        direction="out",
        labelleft=False,
        labelbottom=False,
    )
    # 控制图框粗细
    for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.1)  # 调整子图布局
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"joint_plot_S{style_id}_C{color_id}_M{marker_id}.png"),
        bbox_inches="tight",
    )
    plt.savefig(
        str(OUTPUT_DIR / f"joint_plot_S{style_id}_C{color_id}_M{marker_id}.pdf"),
        bbox_inches="tight",
    )


# =========================================================================================
# ======================================6.主程序执行部分=======================================
# =========================================================================================
if __name__ == "__main__":
    # 请确保该路径下有对应的Excel文件
    data_path = str(DATA_DIR / "data.xlsx")  # 设置输入数据文件的路径
    df = pd.read_excel(data_path)  # 读取指定路径的 Excel 文件数据到 DataFrame
    style_index = 1  # 使用的样式
    scheme_index = 10  # 使用的颜色方案
    MARKER_index = 4  # 使用的标记方案
    draw_joint_plot(df, style_index, scheme_index, MARKER_index)  # 调用绘图函数

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

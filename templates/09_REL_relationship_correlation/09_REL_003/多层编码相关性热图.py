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

# 导入pandas库，用于数据处理和分析，通常简写为pd
# 导入matplotlib的colors模块，用于处理颜色和颜色映射
import matplotlib.colors as mcolors

# 导入matplotlib的pyplot模块，用于创建图表和可视化，通常简写为plt
import matplotlib.pyplot as plt

# 导入numpy库，用于进行大量的数值计算，通常简写为np
import numpy as np
import pandas as pd

# 从matplotlib的patches模块导入Ellipse类，用于绘制椭圆
from matplotlib.patches import Ellipse

# 从scipy库的stats模块导入pearsonr, spearmanr, kendalltau函数，分别用于计算皮尔逊、斯皮尔曼、肯德尔相关性检验
from scipy.stats import kendalltau, pearsonr, spearmanr

# 设置matplotlib的全局字体参数，用于正确显示中文字符（这里设置为'Times New Roman'字体）
plt.rcParams["font.family"] = "Times New Roman"
# 设置matplotlib的全局参数，用于正常显示负号
plt.rcParams["axes.unicode_minus"] = False
# 导入matplotlib库本身
import matplotlib

# 设置matplotlib的后端为'TkAgg'，这是一个与Tkinter GUI库集成的后端，通常用于在脚本中显示图形窗口
matplotlib.use("Agg")


# 定义一个函数，用于绘制相关性椭圆矩阵图
def plot_corr_ellipses(data, p_vals, cmap, corr_max_scale=1.0):
    """
    绘制一个混合相关性矩阵图：下三角为椭圆，上三角为数值。
    """
    # 检查输入的相关系数矩阵(data)和p值矩阵(p_vals)是否都是方阵
    if data.shape[0] != data.shape[1] or p_vals.shape[0] != p_vals.shape[1]:
        # 如果输入的不是方阵，则抛出一个值错误(ValueError)异常
        raise ValueError("输入的DataFrame必须是方阵")

    # 获取矩阵的维度（即变量的数量）
    n = data.shape[0]
    # 创建一个图形对象(fig)和一个子图坐标轴对象(ax)，并设置图形大小为11x10英寸，分辨率为100DPI
    fig, ax = plt.subplots(1, 1, figsize=(11, 10), dpi=100)

    # 从输入数据(data)的列名中获取所有变量的标签，并转换为列表
    labels = data.columns.tolist()

    # 开始双层循环，遍历矩阵中的每一个单元格
    for i in range(n):  # 外层循环，遍历行
        for j in range(n):  # 内层循环，遍历列
            # 获取在第i行第j列的相关系数值
            corr = data.iloc[i, j]
            # 获取在第i行第j列的p值
            p = p_vals.iloc[i, j]
            # 将[-1, 1]范围的相关系数值映射到[0, 1]范围，并从传入的cmap中获取对应的颜色
            color = cmap((corr + 1) / 2)
            # 判断当前位置是否在矩阵的对角线上 (i等于j)
            if i == j:
                # 在对角线上以粗体显示变量名，并将其居中放置
                ax.text(
                    j,
                    i,
                    labels[i],
                    ha="center",
                    va="center",
                    fontsize=12,
                    weight="bold",
                    color="black",
                )
                # 跳过本次内层循环的剩余部分，直接进入下一次循环
                continue
            # 判断当前位置是否在矩阵的下三角区域 (i大于j)
            if i > j:
                # 根据用户定义的corr_max_scale对相关性进行归一化，并确保结果不超过1.0
                scale = min(abs(corr) / corr_max_scale, 1.0)
                # 根据相关性计算椭圆的高度，相关性越强，椭圆越扁平
                height = np.sqrt(1 - scale**2)
                # 椭圆的宽度等于归一化后的相关性的绝对值
                width = scale
                # 如果相关性为正，椭圆倾斜45度；如果为负，则倾斜-45度
                angle = 45 if corr > 0 else -45
                # 创建一个椭圆(Ellipse)对象，设置其中心坐标、宽高、角度、填充颜色和边框颜色
                ellipse = Ellipse(
                    xy=(j, i),
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor=color,
                    edgecolor="none",
                )
                # 将创建好的椭圆对象添加到子图中进行绘制
                ax.add_patch(ellipse)
                # 初始化一个空字符串，用于存放显著性星号
                stars = ""
                # 如果p值小于0.01，表示极显著
                if p < 0.01:
                    # 标记为两个星号
                    stars = "**"
                # 否则，如果p值小于0.05（但不小于0.01），表示显著
                elif p < 0.05:
                    # 标记为一个星号
                    stars = "*"
                # 在图的(j, i)位置添加显著性星号文本，并居中显示
                ax.text(j, i, stars, ha="center", va="center", fontsize=20, color="black")
            # 判断当前位置是否在矩阵的上三角区域 (i小于j)
            elif i < j:
                # 在图的(j, i)位置添加格式化为两位小数的相关系数值文本，并使用对应的颜色和粗体
                ax.text(
                    j,
                    i,
                    f"{corr:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                    weight="bold",
                )
    # --- 图表格式化 ---
    # 设置x轴的刻度位置
    ax.set_xticks(np.arange(n))
    # 设置y轴的刻度位置
    ax.set_yticks(np.arange(n))
    # 设置x轴的刻度标签，并将其旋转45度，右对齐，设置为粗体
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12, weight="bold")
    # 设置y轴的刻度标签，并设置为粗体
    ax.set_yticklabels(labels, fontsize=12, weight="bold")
    # 设置x轴的显示范围，使其两边留有一定空白
    ax.set_xlim(-0.5, n - 0.5)
    # 设置y轴的显示范围，并将其反转以使(0,0)点位于左上角
    ax.set_ylim(n - 0.5, -0.5)
    # 遍历图表的所有边框（上下左右四条）
    for spine in ax.spines.values():
        # 设置边框的线宽为1.5
        spine.set_linewidth(1.5)
    # 设置主刻度线的参数（长度、宽度、颜色）
    ax.tick_params(axis="both", which="major", length=4, width=1.5, colors="black")
    # 创建一个标量可映射对象，用于生成颜色条，其数值范围从-1到1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    # 为标量可映射对象设置一个空数组（这是生成颜色条所必需的步骤）
    sm.set_array([])
    # 在图表右侧添加一个垂直的颜色条
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.02, aspect=30)
    # 设置颜色条的标签及其字体大小和粗细
    cbar.set_label("correlation coefficient(r)", size=12, weight="bold")
    # 设置颜色条轮廓线的宽度
    cbar.outline.set_linewidth(1.5)
    # 设置颜色条刻度标签的字体大小和刻度线宽度
    cbar.ax.tick_params(labelsize=11, width=1.5)
    # 将颜色条的刻度标签设置为粗体
    plt.setp(cbar.ax.get_yticklabels(), fontweight="bold")
    # 自动调整图表布局以适应窗口大小，并留出顶部空间
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 将最终的图表保存为PNG图片文件，并裁剪掉周围的空白
    plt.savefig(str(OUTPUT_DIR / "9.png"), bbox_inches="tight")
    # 显示生成的图表窗口
    plt.close("all")  # Interactive display removed; assets were exported above.


# 这是一个Python的惯用写法，确保以下代码只在直接运行该脚本文件时执行
if __name__ == "__main__":
    # --- 1. 加载数据 ---
    # 定义要读取的Excel文件的路径
    file_path = str(DATA_DIR / "部分相关性.xlsx")
    # 使用try-except语句块来处理可能发生的错误（如文件未找到）
    try:
        # 使用pandas的read_excel函数读取指定路径的Excel文件
        data_df = pd.read_excel(file_path)
        # 将DataFrame中的所有数据尝试转换为数值类型，无法转换的值会变成NaN（Not a Number）
        data_df = data_df.apply(pd.to_numeric, errors="coerce")
        # 删除所有包含NaN值的行，inplace=True表示直接在原DataFrame上修改
        data_df.dropna(inplace=True)
        # 检查处理后的数据是否为空
        if data_df.empty:
            # 如果数据为空，则抛出一个值错误，提示用户检查文件内容
            raise ValueError("数据为空或所有行都包含非数值数据。请检查您的Excel文件。")
    # 如果在try块中发生了FileNotFoundError（文件未找到）
    except FileNotFoundError:
        # 抛出一个新的FileNotFoundError，并附带更详细的错误信息
        raise FileNotFoundError(f"错误: 在路径 {file_path} 未找到文件。")
    # --- 2. 计算相关性和P值 ---
    # 确认数据DataFrame已成功加载且不为None
    if data_df is not None:
        # 获取数据中的变量数量（即列数）
        n_vars = data_df.shape[1]
        # 获取所有列的名称作为标签
        labels = data_df.columns.tolist()
        # --- 相关性方法选择 ---
        # 在这里选择您想要使用的相关性分析方法
        # 可选值: 'pearson', 'spearman', 'kendall'
        # 'pearson': 皮尔逊相关，适用于线性的连续数据。
        # 'spearman': 斯皮尔曼等级相关，适用于非线性单调关系，对异常值不敏感。
        # 'kendall': 肯德尔秩相关，类似于斯皮尔曼，但在小样本时更稳健。
        selected_method = "pearson"  # <--- 修改这里来切换方法
        # 打印出当前选择的方法，这是一个提示信息，告诉用户当前正在使用哪种方法
        print(f"正在使用 {selected_method} 方法进行相关性分析...")
        # 根据选择的方法计算相关系数矩阵，调用DataFrame的.corr()方法，并传入选择的方法
        corr_df = data_df.corr(method=selected_method)
        # 创建一个与数据维度相同，并用1填充的DataFrame，用于存放p值
        p_vals_df = pd.DataFrame(np.ones((n_vars, n_vars)), columns=labels, index=labels)
        # 开始双层循环，计算每对变量之间的p值
        for i in range(n_vars):
            # 内层循环从i开始，避免重复计算（因为相关矩阵是对称的）
            for j in range(i, n_vars):
                # 如果是对角线上的元素（变量与自身的相关性）
                if i == j:
                    # p值设为0.0
                    p_vals_df.iloc[i, j] = 0.0
                # 如果是非对角线元素
                else:
                    # 根据选择的方法调用不同的函数来计算p值
                    if selected_method == "pearson":  # 如果选择的是皮尔逊
                        # 调用pearsonr函数计算相关系数和p值
                        corr_val, p_val = pearsonr(data_df.iloc[:, i], data_df.iloc[:, j])
                    elif selected_method == "spearman":  # 如果选择的是斯皮尔曼
                        # 调用spearmanr函数计算相关系数和p值
                        corr_val, p_val = spearmanr(data_df.iloc[:, i], data_df.iloc[:, j])
                    elif selected_method == "kendall":  # 如果选择的是肯德尔
                        # 调用kendalltau函数计算相关系数和p值
                        corr_val, p_val = kendalltau(data_df.iloc[:, i], data_df.iloc[:, j])

                    # 将计算出的p值赋给p值矩阵的(i, j)位置
                    p_vals_df.iloc[i, j] = p_val
                    # 同时也将p值赋给对称的(j, i)位置，保持矩阵的对称性
                    p_vals_df.iloc[j, i] = p_val

        # --- 新增: 配色方案选择 ---
        # 设置要选择的配色方案的索引（0表示第一个方案）
        selected_scheme_index = 0
        # 定义一个包含多种matplotlib颜色映射方案的列表
        colormap_options = [
            # 自定义颜色映射：从深蓝到深绿再到黄
            mcolors.LinearSegmentedColormap.from_list(
                "custom_cmap",
                [(0, "#00008B"), (0.45, "#006400"), (0.55, "#006400"), (1, "#FFFF00")],
            ),
            # 其他预设的颜色映射方案
            plt.get_cmap("coolwarm"),
            plt.get_cmap("bwr"),
            plt.get_cmap("RdBu_r"),
            plt.get_cmap("seismic"),
            plt.get_cmap("PRGn"),
            plt.get_cmap("PiYG"),
            plt.get_cmap("BrBG"),
            plt.get_cmap("viridis"),
            plt.get_cmap("plasma"),
        ]
        # 使用try-except来处理可能发生的索引越界错误
        try:
            # 根据索引从列表中选择一个颜色映射方案
            selected_cmap = colormap_options[selected_scheme_index]
        # 如果索引超出了列表的范围
        except IndexError:
            # 打印一条警告信息
            print(f"警告：无效的索引 {selected_scheme_index}。将使用默认方案 0。")
            # 使用列表中的第一个（默认）颜色映射方案
            selected_cmap = colormap_options[0]
        # --- 3. 调用绘图函数---
        # 计算数据中实际的最大绝对相关系数（不包括对角线上的1）
        corr_max_actual = corr_df[corr_df < 1.0].abs().max().max()
        # 打印出计算得到的最大绝对相关系数，格式化为两位小数
        print(f"数据中实际的最大绝对相关系数是: {corr_max_actual:.2f}")
        # 手动设置用于椭圆缩放的最大相关性值
        manual_scale = 1
        # 打印出将要使用的手动比例尺
        print(f"使用手动设置的比例尺 {manual_scale} 进行绘图。")
        # 调用绘图函数，传入相关系数矩阵、p值矩阵、选定的颜色映射以及手动设置的比例尺
        plot_corr_ellipses(corr_df, p_vals_df, selected_cmap, corr_max_scale=manual_scale)

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

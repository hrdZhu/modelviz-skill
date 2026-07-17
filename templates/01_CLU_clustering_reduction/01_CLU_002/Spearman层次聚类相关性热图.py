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

# 导入pandas库，用于数据处理和创建DataFrame，简写为pd
# 导入matplotlib库
import matplotlib

# 导入matplotlib的pyplot模块，用于底层绘图和图表定制，简写为plt
import matplotlib.pyplot as plt

# 导入numpy库，用于高效的数值和数组运算，简写为np
import numpy as np
import pandas as pd

# 导入seaborn库，用于绘制更美观的统计图表，简写为sns
import seaborn as sns

# 从scipy的stats模块导入spearmanr函数，用于计算斯皮尔曼等级相关性
from scipy.stats import spearmanr

# 指定matplotlib使用的图形后端为'TkAgg'，确保在某些环境下能正确弹出图形窗口
matplotlib.use("Agg")


def plot_spearman_clustermap(data1, data2):
    """
    计算两个DataFrame之间的Spearman相关性，并绘制带星号标记的方形单元格聚类热图。
    (此函数无需修改，可直接使用)
    参数:
    data1 (pd.DataFrame): 第一个数据集 (例如: 微生物丰度, 行=样本, 列=物种/类群)。
    data2 (pd.DataFrame): 第二个数据集 (例如: 环境因子, 行=样本, 列=因子)。
    """
    # --- 1. 计算相关系数 (r) 和 P值矩阵 ---
    # 在控制台打印信息，提示正在对齐数据
    print("\n--- 正在对齐两个数据集的样本行 ---")
    # 打印对齐前第一个数据集的前5个索引，用于调试
    print(f"对齐前, data1索引: {data1.index[:5]}...")
    # 打印对齐前第二个数据集的前5个索引，用于调试
    print(f"对齐前, data2索引: {data2.index[:5]}...")
    # 使用align方法按行索引对齐两个数据集，join='inner'表示只保留共有的行
    data1, data2 = data1.align(data2, join="inner", axis=0)
    # 打印对齐后共同样本的数量，确认对齐是否成功
    print(f"对齐后, 共同样本数量: {data1.shape[0]}")
    # 检查对齐后共同样本数是否为0
    if data1.shape[0] == 0:
        # 如果没有共同样本，则抛出一个ValueError异常并终止程序，给出清晰的错误提示
        raise ValueError(
            "错误：两个数据集没有共同的样本索引，无法进行相关性分析！请检查Excel文件中的样本ID是否一致。"
        )
    # 获取第一个数据集的列名，这将作为热图的行标签
    rows = data1.columns
    # 获取第二个数据集的列名，这将作为热图的列标签
    cols = data2.columns
    # 初始化一个空的DataFrame，用于存储斯皮尔曼相关系数值(r值)
    corr_matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)
    # 初始化一个空的DataFrame，用于存储对应的p值
    p_matrix = pd.DataFrame(index=rows, columns=cols, dtype=float)
    # 通过双层循环，计算每一对变量（data1的列 vs data2的列）的相关性
    for r in rows:  # 开始外层循环，遍历第一个数据集的每一列（即热图的每一行）
        for c in cols:  # 开始内层循环，遍历第二个数据集的每一列（即热图的每一列）
            # 调用spearmanr函数计算相关系数(corr)和p值(p_val)，nan_policy='omit'会自动忽略计算中的空值
            corr, p_val = spearmanr(data1[r], data2[c], nan_policy="omit")
            # 将计算出的相关系数存入矩阵的相应位置
            corr_matrix.loc[r, c] = corr
            # 将计算出的p值存入矩阵的相应位置
            p_matrix.loc[r, c] = p_val
    # 终极修复：将任何可能因为数据点不足（即使忽略了NaN之后）而产生的NaN相关性值填充为0
    corr_matrix.fillna(0, inplace=True)
    # --- 2. 创建显著性星号的注释矩阵 ---
    # 遍历p值矩阵，使用lambda函数根据p值的大小，生成对应的星号字符串用于显著性标记
    annot_matrix = p_matrix.map(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ""))
    # --- 3. 绘制聚类热图 ---
    # 设置matplotlib的全局字体为'Times New Roman'，以满足期刊发表要求
    plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    # 设置matplotlib正常显示负号，而不是一个方框
    plt.rcParams["axes.unicode_minus"] = False
    # 定义一个颜色映射方案，'PuOr_r'是一个从紫色到橙色的反向渐变色
    cmap = "PuOr_r"
    # 获取相关系数矩阵的行数和列数
    n_rows, n_cols = corr_matrix.shape
    # 根据行列数动态计算合适的图形宽度，确保热图单元格近似方形
    fig_width = n_cols * 0.5 + 4
    # 根据行列数动态计算合适的图形高度
    fig_height = n_rows * 0.5 + 4
    # 调用seaborn的clustermap函数进行绘图，它会自动完成聚类和热图的绘制
    g = sns.clustermap(
        corr_matrix,  # 绘图使用的数据：我们计算出的相关系数矩阵
        method="average",  # 指定聚类算法为“平均连接法” (UPGMA)
        metric="euclidean",  # 指定计算样本/特征间距离的方法为“欧氏距离”
        cmap=cmap,  # 使用我们上面定义的'PuOr_r'颜色映射
        annot=annot_matrix,  # 传入我们生成的星号注释矩阵，在单元格中显示星号
        fmt="s",  # 指定注释的格式为字符串（'s' for string）
        annot_kws={
            "size": 16,
            "color": "black",
            "fontweight": "bold",
        },  # 设置注释（星号）的字体属性
        linewidths=0.5,  # 设置热图单元格之间的白色网格线的宽度
        linecolor="white",  # 设置网格线的颜色为白色
        dendrogram_ratio=0.1,  # 设置聚类树（dendrogram）所占的尺寸比例
        figsize=(fig_width, fig_height),  # 使用我们动态计算出的图形尺寸
        cbar_pos=(1.15, 0.65, 0.03, 0.2),  # 手动设置颜色条的位置(x, y, width, height)，放在图的右侧
        cbar_kws={"label": "Spearman's r"},  # 设置颜色条的标签文字
        vmin=-0.5,
        vmax=0.5,  # 设置颜色条的数值范围，使其对称，便于观察正负相关
    )
    # --- 4. 调整和美化图形 ---
    # 使用plt.setp统一设置X轴刻度标签的属性：旋转90度，并设置字体大小为24
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=24)
    # 使用plt.setp统一设置Y轴刻度标签的属性：不旋转（0度），并设置字体大小为24
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=24)
    # 获取颜色条的坐标轴(g.cax)，并设置其Y轴标签（即颜色条标题）的字体大小和粗细
    g.cax.set_ylabel("Spearman's r", fontsize=20, fontweight="bold")
    # 设置颜色条刻度数字的字体大小
    g.cax.tick_params(labelsize=20)
    # 访问行聚类图(左侧树状图)的线条集合并设置线宽为1.5
    g.ax_row_dendrogram.collections[0].set_linewidth(1.5)
    # 访问列聚类图(顶部树状图)的线条集合并设置线宽为1.5
    g.ax_col_dendrogram.collections[0].set_linewidth(1.5)
    # --- 5. 手动添加P值图例 ---
    # 定义P值图例的文本内容，使用换行符\n
    legend_text = "P value\n*** < 0.01\n** 0.01-0.05"
    # 在图形的指定位置添加一个文本框作为图例
    plt.text(
        1.15,
        0.6,
        legend_text,
        transform=g.fig.transFigure,  # 使用相对整个图形的坐标系
        fontsize=20,
        verticalalignment="top",  # 设置字体和对齐方式
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1),
    )  # 设置文本框的样式
    # 将生成的图形保存到指定的文件路径
    plt.savefig(
        str(OUTPUT_DIR / "55.png"),
        bbox_inches="tight",  # 自动调整边界框，裁剪掉多余的空白，确保所有内容都被保存
        pad_inches=0.2,  # 在紧密边界框之外额外添加0.2英寸的空白边距
        dpi=300,
    )  # 设置图片的分辨率为300点/英寸，以获得高清晰度
    # 在屏幕上显示最终的图形窗口
    plt.close("all")  # Interactive display removed; assets were exported above.


# 只有当这个脚本被直接运行时，以下代码才会执行
if __name__ == "__main__":
    # 定义包含原始数据的Excel文件的路径
    input_excel_path = str(DATA_DIR / "simulated_data.xlsx")
    # 定义要读取的第一个工作表的名称
    sheet_name_1 = "Microbial_Guilds"
    # 定义要读取的第二个工作表的名称
    sheet_name_2 = "Environmental_Factors"
    # 在控制台打印信息，告知用户正在从哪个文件加载数据
    print(f"正在从 '{input_excel_path}' 加载数据...")
    # 使用pandas的read_excel函数从第一个工作表加载微生物数据
    # 这里不设置index_col，让pandas自动创建从0开始的整数索引
    microbe_data = pd.read_excel(input_excel_path, sheet_name=sheet_name_1)
    # 从第二个工作表加载环境因子数据
    # index_col=0 表示使用Excel的第一列作为DataFrame的索引（通常是样本ID）
    env_data = pd.read_excel(input_excel_path, sheet_name=sheet_name_2, index_col=0)
    # 在控制台打印成功加载的信息
    print("数据加载成功！")
    # 调用我们上面定义好的绘图函数，传入清洗后的两个数据集作为参数
    plot_spearman_clustermap(microbe_data, env_data)

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

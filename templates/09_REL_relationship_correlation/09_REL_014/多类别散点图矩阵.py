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

import os  # 导入os库，用于与操作系统交互，如此处的创建文件夹

import matplotlib  # 导入matplotlib库，用于配置图形属性
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图，通常简写为plt
import numpy as np  # 导入numpy库，用于数值计算
import pandas as pd  # 导入pandas库，用于数据处理，通常简写为pd
import seaborn as sns  # 导入seaborn库，用于数据可视化，通常简写为sns
from scipy.stats import (
    gaussian_kde,  # 从scipy.stats库中导入gaussian_kde，用于计算核密度估计
    pearsonr,  # 从scipy.stats库中导入pearsonr函数，用于计算皮尔逊相关系数和p值
)

matplotlib.use("Agg")  # 尝试设置matplotlib的后端为'TkAgg'，这是一个常用的图形界面后端


# 一个辅助函数，用于根据p值返回显著性标记
def get_significance_marker(
    p_value,
):  # 定义一个名为get_significance_marker的函数，接收一个参数p_value
    """根据p值返回星号标记"""  # 函数的文档字符串（docstring），解释函数的功能
    if p_value < 0.001:  # 如果p值小于0.001
        return "***"  # 返回三个星号，表示极显著
    elif p_value < 0.01:  # 否则，如果p值小于0.01
        return "**"  # 返回两个星号，表示非常显著
    elif p_value < 0.05:  # 否则，如果p值小于0.05
        return "*"  # 返回一个星号，表示显著
    else:  # 如果以上条件都不满足
        return ""  # 返回一个空字符串，表示不显著


# 定义数据所在的文件夹和完整文件路径
output_folder = str(OUTPUT_DIR)  # 定义一个原始字符串变量，存储输出文件夹的路径
excel_file_path = os.path.join(
    output_folder, str(DATA_DIR / "iris_data.xlsx")
)  # 使用os.path.join拼接文件夹和文件名，生成完整的文件路径
# 加载和准备数据
iris_df = pd.read_excel(excel_file_path)  # 尝试使用pandas的read_excel函数从指定的路径读取Excel文件
print(f"数据已成功从本地文件加载: {excel_file_path}")  # 如果成功，打印一条成功加载的提示信息
# ==============================================================================
if not iris_df.empty:  # 检查加载后的DataFrame是否为空，如果不为空则执行后续代码
    iris_df.columns = [
        "Sepal.Length",
        "Sepal.Width",
        "Petal.Length",
        "Petal.Width",
        "Species",
    ]  # 重命名DataFrame的列名
    # 定义美学元素
    species_list = ["setosa", "versicolor", "virginica"]  # 定义一个列表，包含鸢尾花的三个种类名称
    columns = iris_df.columns  # 获取DataFrame的所有列名
    # ==============================================================================
    #                             多种配色方案选择区域
    #   使用时，请取消注释您想要的一套配色方案，并确保其他方案被注释（即行首有#）
    # ==============================================================================
    # --- 配色方案1---
    colors_list = ["#02AFBB", "#E7B800", "#F8766D"]  # 定义一个包含三个十六进制颜色码的列表
    color_map = dict(zip(species_list, colors_list))  # 创建一个字典，将物种名和颜色一一对应起来
    # # --- 配色方案2---
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 稳重蓝, 活力橙, 自然绿
    color_map = dict(zip(species_list, colors_list))
    # # --- 配色方案3---
    colors_list = ["#a1c9f4", "#ffb482", "#8de5a1"]  # 淡蓝, 淡橙, 淡绿
    color_map = dict(zip(species_list, colors_list))
    # # --- 配色方案4---
    colors_list = ["#333333", "#808080", "#CCCCCC"]  # 深灰, 中灰, 浅灰
    color_map = dict(zip(species_list, colors_list))
    # # --- 配色方案5---
    colors_list = ["#440154", "#21918c", "#fde725"]  # 深紫, 青色, 亮黄
    color_map = dict(zip(species_list, colors_list))
    # # --- 配色方案6 ---
    colors_list = ["#F8766D", "#7CAE00", "#00BFC4"]  # 三文鱼红, 青柠绿, 亮青色
    color_map = dict(zip(species_list, colors_list))
    # ==============================================================================
    sns.set_style("ticks")  # 设置seaborn的绘图风格为"ticks"，这会显示坐标轴刻度线
    # 创建 5x5 的子图网格
    fig, axes = plt.subplots(
        5, 5, figsize=(15, 15)
    )  # 创建一个5x5的子图网格，返回图形对象(fig)和所有子图的二维数组(axes)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)  # 调整子图之间的水平(hspace)和垂直(wspace)间距
    # 遍历网格并填充绘图
    for i, row_var in enumerate(
        columns
    ):  # 开始外层循环，用enumerate遍历所有列名，i为行索引，row_var为行对应的变量名
        for j, col_var in enumerate(
            columns
        ):  # 开始内层循环，用enumerate遍历所有列名，j为列索引，col_var为列对应的变量名
            ax = axes[i, j]  # 获取当前循环对应的子图(Axes)对象
            if i == j:  # --- 对角线图 --- # 判断当前子图是否在网格的对角线上（行索引等于列索引）
                if i < 4:  # 如果是前四个对角线图（对应四个数值型特征）
                    for species in species_list:  # 遍历每个物种
                        sns.kdeplot(
                            data=iris_df[iris_df["Species"] == species],
                            x=col_var,
                            ax=ax,  # 筛选出当前物种的数据
                            fill=True,
                            color=color_map[species],
                            legend=False,
                            linewidth=0,
                        )  # 在子图上绘制填充的核密度估计图，不显示图例，线条宽度为0
                else:  # 如果是第五个对角线图（对应Species列）
                    sns.countplot(
                        data=iris_df,
                        x="Species",
                        ax=ax,
                        hue="Species",
                        palette=color_map,
                        legend=False,
                    )  # 绘制一个计数图（条形图），统计各种类的数量
            elif i > j:  # --- 左下三角图 --- # 判断当前子图是否在对角线的左下方（行索引大于列索引）
                if i < 4:  # 如果当前行对应的变量是数值型变量
                    # 首先，像原来一样绘制所有物种的散点图。稍微增加透明度，让拟合线更清晰。
                    sns.scatterplot(
                        data=iris_df,
                        x=col_var,
                        y=row_var,
                        hue="Species",
                        palette=color_map,
                        ax=ax,
                        # 在子图上绘制散点图
                        legend=False,
                        s=40,
                        edgecolor="none",
                        alpha=0.7,
                    )  # 不显示图例，点大小为40，无边缘色，透明度为0.7
                    # 接着，为每个物种循环叠加一条拟合线和置信区间
                    for species in species_list:  # 遍历每个物种
                        subset = iris_df[iris_df["Species"] == species]  # 筛选出当前物种的数据子集
                        # 使用regplot绘制，但设置scatter=False以避免重复绘制散点
                        sns.regplot(
                            data=subset,
                            x=col_var,
                            y=row_var,
                            ax=ax,  # 在同一子图上叠加回归图
                            color=color_map[species],  # 设置线条颜色与该物种的散点颜色一致
                            scatter=False,
                            truncate=False,
                        )  # 设置scatter为False表示不绘制散点，truncate为False确保拟合线画满x轴范围
                    # ==================================================================
                else:  # 如果当前行对应的变量是'Species'
                    sns.histplot(
                        data=iris_df,
                        x=col_var,
                        hue="Species",
                        multiple="stack",  # 绘制一个堆叠直方图
                        bins=15,
                        shrink=0.8,
                        palette=color_map,
                        ax=ax,
                        legend=False,
                        alpha=0.9,
                    )  # 设置参数，不显示图例，透明度为0.9
            else:  # i < j, --- 右上三角图 --- # 判断当前子图是否在对角线的右上方（行索引小于列索引）
                if j < 4:  # 如果当前列对应的变量是数值型变量
                    ax.set_facecolor("#F0F0F0")  # 设置子图的背景颜色为浅灰色
                    ax.set_xticks([])  # 移除x轴的刻度
                    ax.set_yticks([])  # 移除y轴的刻度
                    overall_corr, overall_p = pearsonr(
                        iris_df[row_var], iris_df[col_var]
                    )  # 计算两个变量在整个数据集上的皮尔逊相关系数和p值
                    stars = get_significance_marker(
                        overall_p
                    )  # 调用辅助函数，根据p值获取显著性星号
                    corr_text = (
                        f"Cor : {overall_corr:.3f}{stars}"  # 格式化字符串，用于显示总体相关性结果
                    )
                    ax.text(
                        0.5,
                        0.82,
                        corr_text,
                        ha="center",
                        va="center",
                        fontsize=16,  # 在子图的指定相对位置添加文本
                        family="Times New Roman",
                        fontweight="bold",
                    )  # 设置文本的字体、字重等属性
                    y_pos = 0.62  # 初始化y轴的相对位置，用于逐行显示各种类的相关性
                    abbr_map = {
                        "setosa": "osa",
                        "versicolor": "lor",
                        "virginica": "ica",
                    }  # 创建一个字典，用于缩写物种名称
                    for species in species_list:  # 遍历每个物种
                        subset = iris_df[iris_df["Species"] == species]  # 筛选出当前物种的数据
                        if len(subset) > 2:  # 检查数据点是否足够进行相关性计算
                            corr, p_val = pearsonr(
                                subset[row_var], subset[col_var]
                            )  # 计算该物种内两个变量的相关系数和p值
                            stars = get_significance_marker(p_val)  # 根据p值获取显著性星号
                            text_line = f"{abbr_map[species]}: {corr:.3f}{stars}"  # 格式化字符串，用于显示该物种的相关性结果
                        else:  # 如果数据点不足
                            text_line = f"{abbr_map[species]}: N/A"  # 显示为'N/A'
                        ax.text(
                            0.5,
                            y_pos,
                            text_line,
                            ha="center",
                            va="center",
                            fontsize=14,  # 在子图上添加该物种的相关性文本
                            color=color_map[species],
                            family="Times New Roman",
                        )  # 设置文本颜色与物种颜色一致
                        y_pos -= 0.22  # 更新y轴位置，为下一个物种的文本留出空间
                else:  # j == 4, i < 4 # 如果当前列是'Species'列，而行是数值型变量
                    # 【代码修改区域】手动绘制最终版云雨图
                    # --------------------------------------------------------------------
                    # 定义每个类别在x轴上的数值中心位置
                    positions = np.arange(len(species_list))

                    # 循环遍历每个物种，分别绘制其云雨图
                    for pos, species in zip(positions, species_list):
                        # 筛选出当前物种的数据
                        subset_data = iris_df[iris_df["Species"] == species][row_var]
                        species_color = color_map[species]

                        # 定义组合图在中心位置的左侧
                        left_center = pos - 0.2

                        # --- 绘制“箱线图与散点图”的组合 (在左侧) ---
                        # a) 绘制完整的箱线图，并使用物种颜色
                        ax.boxplot(
                            subset_data,
                            positions=[left_center],
                            manage_ticks=False,
                            patch_artist=True,
                            widths=0.2,
                            # 设置箱体、中位数线、胡须和顶帽的样式
                            medianprops={"color": "white", "linewidth": 1.5},
                            boxprops={
                                "facecolor": species_color,
                                "alpha": 0.7,
                                "edgecolor": "black",
                            },
                            whiskerprops={"color": species_color, "linewidth": 1.5},
                            capprops={"color": species_color, "linewidth": 1.5},
                            # 不单独显示异常值，因为散点图会包含所有点
                            showfliers=False,
                            zorder=2,
                        )

                        # b) 在箱线图上叠加散点图（雨点）
                        # 为散点添加抖动，使其围绕箱线图中心散开
                        jitter = np.random.normal(
                            loc=left_center, scale=0.03, size=len(subset_data)
                        )
                        ax.scatter(
                            jitter,
                            subset_data,
                            s=20,
                            # 使用物种颜色填充，并添加黑色边缘以突出显示
                            facecolor=species_color,
                            edgecolor="black",
                            linewidth=0.5,
                            alpha=0.8,
                            zorder=3,
                        )  # zorder=3确保散点在最上层

                        # --- 绘制“云朵”(Cloud) (在右侧) ---
                        if len(subset_data) > 1:
                            # 使用scipy计算核密度估计 (KDE)
                            kde = gaussian_kde(subset_data)
                            y_range = np.linspace(subset_data.min(), subset_data.max(), 200)
                            density = kde(y_range)

                            # 将密度值进行缩放，作为云朵的宽度
                            scaled_density = density / density.max() * 0.4

                            # 将云朵画在中心位置'pos'的右侧
                            ax.fill_betweenx(
                                y_range,
                                pos,
                                pos + scaled_density,
                                color=species_color,
                                alpha=0.5,
                                zorder=1,
                            )

                    # --- 格式化当前子图的坐标轴 ---
                    # 将刻度设置在每个组合图的视觉中心
                    ax.set_xticks(positions)
                    ax.set_xticklabels(species_list)
                    ax.set_xlim(-0.5, len(species_list) - 0.5)
                    # --------------------------------------------------------------------

    # 统一调整格式
    for i in range(5):  # 再次遍历所有行
        for j in range(5):  # 再次遍历所有列
            ax = axes[i, j]  # 获取当前子图对象
            ax.set_xlabel("")  # 清空x轴标签
            ax.set_ylabel("")  # 清空y轴标签
            if i == 0:  # 如果是第一行子图
                ax.set_title(columns[j], fontsize=16, pad=15)  # 设置子图的标题为列名
            if j == 0:  # 如果是第一列子图
                ax.set_ylabel(columns[i], fontsize=16, labelpad=15)  # 设置子图的y轴标签为行名
            ax.spines["top"].set_visible(False)  # 设置顶部坐标轴脊（边框）不可见
            ax.spines["right"].set_visible(False)  # 设置右侧坐标轴脊（边框）不可见
            if i < j and j < 4:  # 如果是右上三角区域（不包括最后一列）的图
                ax.spines["left"].set_visible(False)  # 隐藏左边框
                ax.spines["bottom"].set_visible(False)  # 隐藏下边框
            if j == 4:  # 如果是最后一列的图
                ax.spines["top"].set_visible(True)  # 重新显示顶部边框
                ax.spines["right"].set_visible(True)  # 重新显示右侧边框
                # 将X轴标签旋转以免重叠
                ax.tick_params(axis="x", labelrotation=45)
            if j > 0:  # 如果不是第一列的图
                ax.tick_params(axis="y", which="both", length=0)  # 隐藏y轴的刻度线
            if i < 4:  # 如果不是最后一行的图
                ax.tick_params(axis="x", which="both", length=0)  # 隐藏x轴的刻度线
    # 确保保存图片的文件夹存在
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹，如果文件夹已存在则不报错
    plt.tight_layout()  # 自动调整子图参数，使其紧密排列，防止重叠
    plt.savefig(
        os.path.join(output_folder, str(OUTPUT_DIR / "1_final_adjusted_raincloud.png")), dpi=300
    )  # 将最终的图形保存为PNG文件，分辨率为300 DPI
    plt.close("all")  # Interactive display removed; assets were exported above.
else:  # 如果最初加载数据时DataFrame为空
    print("由于数据加载失败，无法继续执行绘图。")  # 打印一条提示信息，告知用户无法绘图

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

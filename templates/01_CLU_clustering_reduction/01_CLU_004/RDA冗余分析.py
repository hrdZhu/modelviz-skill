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
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import pandas as pd
from skbio.stats.ordination import rda
from sklearn.preprocessing import StandardScaler

# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["#2a6a66", "#7bc8c1", "#ee8b73", "#cf2e2e"],
    2: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    3: ["#E64B35", "#4DBBD5", "#00A087", "#3C5488"],
    4: ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"],
    5: ["#003f5c", "#58508d", "#bc5090", "#ff6361"],
    6: ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"],
    7: ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"],
    8: ["#000000", "#525252", "#969696", "#d9d9d9"],
    9: ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"],
    10: ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"],
    11: ["#9e0142", "#d53e4f", "#f46d43", "#fdae61"],
    12: ["#5e4fa2", "#3288bd", "#66c2a5", "#abdda4"],
    13: ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
    14: ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4"],
    15: ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"],
    16: ["#762a83", "#af8dc3", "#7fbf7b", "#1b7837"],
    17: ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"],
    18: ["#a50026", "#d73027", "#f46d43", "#fdae61"],
    19: ["#313695", "#4575b4", "#74add1", "#abd9e9"],
    20: ["#8c510a", "#bf812d", "#dfc27d", "#f6e8c3"],
}
SELECTED_SCHEME_ID = 20  # 配色方案


# =========================================================================================
# ====================================== 3.RDA 计算=========================================
# =========================================================================================
def run_rda_with_skbio(X, Y, scaling_type=2):
    # 标准化处理
    scaler_X = StandardScaler()
    X_scaled_array = scaler_X.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    # 执行RDA分析
    rda_result = rda(
        Y, X_scaled, scale_Y=True, scaling=scaling_type
    )  # Y为响应变量，X_scaled为预测变量，scale_Y=True表示对Y也进行标准化
    # 提取分析结果，样本得分的前两列 (RDA1, RDA2)
    rda_scores = rda_result.samples.iloc[:, :2].values
    # 提取性状得分
    trait_loadings = rda_result.features.iloc[:, :2].values
    # 提取环境因子得分
    env_loadings = rda_result.biplot_scores.iloc[:, :2].values
    # 提取方差解释率
    variance_ratios = rda_result.proportion_explained.iloc[:2].values
    print(f"RDA1={variance_ratios[0]:.2%}, RDA2={variance_ratios[1]:.2%}")
    return rda_scores, trait_loadings, env_loadings, variance_ratios


# =========================================================================================
# ====================================== 4.绘图函数=========================================
# =========================================================================================
def plot_and_save_rda_results(
    X, Y, meta, rda_scores, trait_loadings, env_loadings, variance, site_colors_map
):
    fig, ax = plt.subplots(figsize=(10, 7))  # 创建图形
    plt.subplots_adjust(right=0.75)  # 调整子图参数，为图例留出空间
    shape_map = {"Deciduous": "o", "Evergreen": "^"}  # 设置不同的标记形状
    wuei_vals = Y["WUEi"].values  # 获取指定列的所有值
    sizes = (
        (wuei_vals - wuei_vals.min()) / (wuei_vals.max() - wuei_vals.min())
    ) * 150 + 30  # 根据WUEi的值计算点的大小
    # 遍历组别数据
    for i in range(len(meta)):
        current_site = meta.iloc[i]["Site"]  # 获取站点，用于配色
        color = site_colors_map.get(current_site, "#333333")  # 获取该站点的颜色
        # 绘制散点图
        ax.scatter(
            rda_scores[i, 0],  # x轴为第i个样本的RDA1得分
            rda_scores[i, 1],  # y轴为第i个样本的RDA2得分
            c=color,  # 颜色
            marker=shape_map[meta.iloc[i]["Leaf_type"]],  # 对应的形状
            s=sizes[i],  # 根据WUEi大小设置点的大小
            alpha=0.85,  # 点的透明度
            edgecolors="none",
        )  # 边缘颜色

    max_score = np.max(np.abs(rda_scores))  # 所有样本得分中的最大绝对值
    scale_factor = max_score * 1.0  # 定义一个缩放因子，用于箭头缩放
    vectors = []  # 用于存储所有向量（环境因子和性状）的信息
    for i, col in enumerate(X.columns):
        vectors.append(
            {"x": env_loadings[i, 0], "y": env_loadings[i, 1], "label": col}
        )  # 遍历X，将其RDA1/RDA2负载和标签存入列表
    for i, col in enumerate(Y.columns):
        vectors.append(
            {"x": trait_loadings[i, 0], "y": trait_loadings[i, 1], "label": col}
        )  # 遍历Y，将其RDA1/RDA2负载和标签存入列表
    vec_df = pd.DataFrame(vectors)  # 转换为DataFrame
    vec_df["angle"] = np.arctan2(vec_df["y"], vec_df["x"])  # 计算每个向量与x轴正方向的夹角
    vec_df = vec_df.sort_values("angle").reset_index(drop=True)  # 按照角度对向量进行排序
    vec_df["radius_mult"] = 1.15
    for i in range(len(vec_df)):  # 遍历排序后的所有向量
        prev = i - 1  # 获取前一个向量的索引
        if prev < 0:
            prev = len(vec_df) - 1  # 如果当前是第一个向量，则将其前一个向量设置为最后一个
        diff = abs(
            vec_df.loc[i, "angle"] - vec_df.loc[prev, "angle"]
        )  # 计算当前向量与前一个向量的角度差
        if diff > np.pi:
            diff = 2 * np.pi - diff  # 如果角度差大于180度，则用360度减去它，取较小的夹角
        if diff < 0.30:  # 如果角度差小于多少的时候，认为标签可能重叠
            if vec_df.loc[prev, "radius_mult"] <= 1.05:  # 检查前一个标签是否在默认位置
                vec_df.loc[i, "radius_mult"] = 1.30  # 如果是，则将当前标签的半径调大
            else:
                vec_df.loc[i, "radius_mult"] = 0.98  # 则将当前标签的半径拉近
    for _, row in vec_df.iterrows():  # 遍历调整了半径后的向量DataFrame
        x = row["x"] * scale_factor  # 将向量的x负载乘以缩放因子
        y = row["y"] * scale_factor  # 将向量的y负载乘以缩放因子
        # 绘制箭头
        ax.arrow(
            0,  # 起点x坐标
            0,  # 起点y坐标
            x,  # 终点x坐标
            y,  # 终点y坐标
            color="black",  # 颜色
            width=0.006,  # 箭头线的宽度
            head_width=0.06,  # 箭头头宽度
            length_includes_head=True,  # 箭头的总长度包含头部
            zorder=5,
        )
        tx = x * row["radius_mult"]  # 标签的x坐标
        ty = y * row["radius_mult"]  # 标签的y坐标
        ang = row["angle"]  # 当前向量的角度
        ha = "left" if -np.pi / 2 <= ang <= np.pi / 2 else "right"  # 标签的水平对齐方式
        va = "bottom" if 0 < ang < np.pi else "top"  # 标签的垂直对齐方式
        # 添加文本
        ax.text(
            tx,  # x坐标
            ty,  # y坐标
            row["label"],  # 文本内容
            color="black",  # 颜色
            fontsize=12,  # 字体大小
            ha=ha,  # 水平对齐方式
            va=va,  # 垂直对齐方式
            zorder=6,
        )
    ax.set_xlabel(f"RDA1 ({variance[0]:.2%})", fontsize=14)  # x轴标题
    ax.set_ylabel(f"RDA2 ({variance[1]:.2%})", fontsize=14)  # y轴标题
    ax.set_xlim(-max_score * 2.2, max_score * 2.2)  # x轴的显示范围
    ax.set_ylim(-max_score * 2.2, max_score * 2.2)  # y轴的显示范围
    # 设置坐标轴刻度样式
    ax.tick_params(
        axis="both",  # 作用于X轴和Y轴
        which="major",  # 作用于主刻度
        direction="out",  # 刻度线朝外
        width=1.5,  # 刻度线粗细
        length=3,  # 度线长度
        labelsize=16,  # 字体大小
    )
    # 遍历边框
    for spine in ax.spines.values():
        spine.set_visible(True)  # 可见
        spine.set_color("black")  # 颜色
        spine.set_linewidth(1.5)  # 线宽
    # 使用列表推导式为每个站点创建一个图例句柄
    site_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="None", markersize=10, label=s)
        for s, c in site_colors_map.items()
    ]
    # 创建第一个图例站点颜色的图例
    leg1 = ax.legend(
        handles=site_handles,
        title="Site",  # 标题
        loc="upper left",  # 图例位置
        bbox_to_anchor=(1.02, 1.0),  # 详细坐标
        frameon=False,  # 不显示边框
        fontsize=14,  # 字体大小
        title_fontsize=16,
    )  # 标题字体大小
    leg1.get_title().set_fontweight("bold")  # 获取图例1的标题并设置为粗体
    ax.add_artist(leg1)  # 将第一个图例添加到坐标轴上

    # 绘制形状的图例
    type_handles = [
        mlines.Line2D(
            [], [], color="black", marker="o", linestyle="None", markersize=8, label="Deciduous"
        ),  # 创建圆形的图例句柄
        mlines.Line2D(
            [], [], color="black", marker="^", linestyle="None", markersize=8, label="Evergreen"
        ),
    ]  # 创建三角形的图例句柄
    # 创建第二个图例
    leg2 = ax.legend(
        handles=type_handles,
        title="Leaf_type",  # 标题
        loc="upper left",  # 位置
        bbox_to_anchor=(1.02, 0.65),
        frameon=False,  # 不显示图例边框
        fontsize=14,  # 字体大小
        title_fontsize=16,
    )  # 标题字体大小
    leg2.get_title().set_fontweight("bold")  # 获取图例2的标题并设置为粗体
    ax.add_artist(leg2)  # 将第二个图例添加到坐标轴上
    # 最大、最小值
    min_wuei = wuei_vals.min()
    max_wuei = wuei_vals.max()
    # 创建等距的区间
    breakpoints = np.linspace(min_wuei, max_wuei, 5)
    # 空的标签列表
    labels = []
    for i in range(4):
        # 格式化字符串来创建区间标签
        label_text = f"{breakpoints[i]:.1f}-{breakpoints[i + 1]:.1f}"
        # 将刚刚创建的文本标签添加到列表
        labels.append(label_text)
    # 定义图例中对应的点大小
    sizes = [4, 7, 10, 13]
    circle_handles = [
        mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=s)
        for s in sizes
    ]  # 创建圆形句柄
    triangle_handles = [
        mlines.Line2D([], [], color="black", marker="^", linestyle="None", markersize=s)
        for s in sizes
    ]  # 创建三角形句柄
    combined_handles = list(
        zip(circle_handles, triangle_handles)
    )  # 将圆形和三角形句柄配对成元组列表
    # 创建第三个图例
    leg3 = ax.legend(
        handles=combined_handles,
        labels=labels,
        title=r"WUEi (mmol mol$^{-1}$)",  # 图例标题
        loc="upper left",
        bbox_to_anchor=(1.02, 0.45),  # 位置
        frameon=False,  # 不显示图例边框
        fontsize=14,  # 字体大小
        title_fontsize=16,  # 标题字体大小
        labelspacing=1.2,  # 标签之间的垂直间距
        handlelength=4,  # 调整图例句柄的长
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    leg3.get_title().set_fontweight("bold")  # 获取图例3的标题并设置为粗体
    ax.add_artist(leg3)  # 将第三个图例添加到坐标轴上
    # 小标题
    ax.text(
        -0.12,  # x坐标
        1.02,  # y坐标
        "(d)",  # 文本内容
        transform=ax.transAxes,  # 指定坐标系
        fontsize=18,  # 字体大小
        fontweight="bold",
    )  # 字体粗细
    plt.savefig(str(OUTPUT_DIR / f"{SELECTED_SCHEME_ID}.png"), dpi=300)
    plt.savefig(str(OUTPUT_DIR / f"{SELECTED_SCHEME_ID}.pdf"), dpi=300)


# =========================================================================================
# ====================================== 5.执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    excel_path = str(DATA_DIR / "data.xlsx")  # 定义数据文件的路径
    X = pd.read_excel(excel_path, sheet_name="Environment_Factors")  # 读取X
    Y = pd.read_excel(excel_path, sheet_name="Traits")  # 读取Y
    meta = pd.read_excel(excel_path, sheet_name="Metadata")  # 读取区域类别
    current_palette = COLOR_SCHEMES.get(SELECTED_SCHEME_ID, COLOR_SCHEMES[1])  # 获取配色方案
    unique_sites = sorted(meta["Site"].unique())  # 获取站点用于配色
    site_colors_map = dict(zip(unique_sites, current_palette))  # 建立站点颜色的映射
    X_numeric = X.select_dtypes(include=[np.number])  # x
    print("X_numeric", X_numeric)
    Y_numeric = Y.select_dtypes(include=[np.number])  # Y
    print("Y_numeric", Y_numeric)
    print("RDA分析")
    scores, t_loads, e_loads, var_ratios = run_rda_with_skbio(X_numeric, Y_numeric, scaling_type=2)

    # 定义输出结果
    output_excel_path = str(OUTPUT_DIR / "rda_analysis_results.xlsx")

    # 创建样本得分的DataFrame
    df_scores = pd.DataFrame(scores, columns=["RDA1", "RDA2"], index=meta.index)
    # 沿着列方向 合并meta和 df_scores
    df_scores_with_meta = pd.concat([meta, df_scores], axis=1)
    # 创建性状负载的DataFrame
    df_t_loads = pd.DataFrame(t_loads, columns=["RDA1", "RDA2"], index=Y_numeric.columns)
    # 索引名称
    df_t_loads.index.name = "Trait"
    # 创建环境因子负载
    df_e_loads = pd.DataFrame(e_loads, columns=["RDA1", "RDA2"], index=X_numeric.columns)
    df_e_loads.index.name = "Environment_Factor"  # 索引名称
    # 创建方差解释率
    df_var_ratios = pd.DataFrame(
        var_ratios, columns=["Proportion_Explained"], index=["RDA1", "RDA2"]
    )
    df_var_ratios.index.name = "Axis"  # 索引名称
    print("df_scores_with_meta", df_scores_with_meta)
    print("df_t_loads", df_t_loads)
    print("df_e_loads", df_e_loads)
    print("df_var_ratios", df_var_ratios)
    # 保存结果的Excel文件
    with pd.ExcelWriter(output_excel_path) as writer:
        df_scores_with_meta.to_excel(writer, sheet_name="Sample_Scores_with_Meta", index=False)
        df_t_loads.to_excel(writer, sheet_name="Trait_Loadings")
        df_e_loads.to_excel(writer, sheet_name="Env_Factor_Loadings")
        df_var_ratios.to_excel(writer, sheet_name="Variance_Explained")

    # 调用前面定义的绘图函数
    plot_and_save_rda_results(
        X_numeric,
        Y_numeric,  # Y
        meta,  # 分组数据
        scores,  # RDA样本得分
        t_loads,  # RDA性状负载
        e_loads,  # RDA环境因子负载
        var_ratios,  # 方差解释率
        site_colors_map=site_colors_map,  # 站点颜色
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

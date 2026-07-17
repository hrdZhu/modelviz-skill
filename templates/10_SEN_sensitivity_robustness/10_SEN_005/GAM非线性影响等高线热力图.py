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
import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, s, te
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ====================================== 2. 颜色库 =========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725", "#fd9a44", "#d73027"],
    2: [
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6dcd59",
        "#b4de2c",
        "#fde725",
    ],
    3: [
        "#0d0887",
        "#46039f",
        "#7201a8",
        "#9c179e",
        "#bd3786",
        "#d8576b",
        "#ed7953",
        "#fb9f3a",
        "#fdca26",
        "#f0f921",
    ],
    4: [
        "#000004",
        "#140e36",
        "#3b0f70",
        "#641a80",
        "#8c2981",
        "#b73779",
        "#de4968",
        "#f1704b",
        "#fe9f6d",
        "#fcfdbf",
    ],
    5: [
        "#000004",
        "#160b39",
        "#420a68",
        "#6a176e",
        "#932667",
        "#bb3754",
        "#dd513a",
        "#f37819",
        "#fca50a",
        "#fcffa4",
    ],
    6: ["#3b4cc0", "#6788ee", "#9abbff", "#c9d7f0", "#edd1c2", "#f7a889", "#e26952", "#b40426"],
    7: [
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",
    ],
    8: [
        "#ffffd9",
        "#edf8b1",
        "#c7e9b4",
        "#7fcdbb",
        "#41b6c4",
        "#1d91c0",
        "#225ea8",
        "#253494",
        "#081d58",
    ],
    9: [
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
    ],
    10: [
        "#8e0152",
        "#c51b7d",
        "#de77ae",
        "#f1b6da",
        "#fde0ef",
        "#f7f7f7",
        "#e6f5d0",
        "#b8e186",
        "#7fbc41",
        "#4d9221",
        "#276419",
    ],
    11: [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ],
    12: [
        "#f7fcf5",
        "#e5f5e0",
        "#c7e9c0",
        "#a1d99b",
        "#74c476",
        "#41ab5d",
        "#238b45",
        "#006d2c",
        "#00441b",
    ],
    13: [
        "#67000d",
        "#a50f15",
        "#cb181d",
        "#ef3b2c",
        "#fb6a4a",
        "#fc9272",
        "#fcbba1",
        "#fee0d2",
        "#fff5f0",
    ][::-1],
    14: [
        "#e2d9e2",
        "#9d9bc3",
        "#615b8d",
        "#422c5e",
        "#281b37",
        "#361633",
        "#5e1f40",
        "#8e3c4e",
        "#c0665e",
        "#dca794",
    ],
    15: [
        "#ffffff",
        "#f0f0f0",
        "#d9d9d9",
        "#bdbdbd",
        "#969696",
        "#737373",
        "#525252",
        "#252525",
        "#000000",
    ],
    16: [
        "#543005",
        "#8c510a",
        "#bf812d",
        "#dfc27d",
        "#f6e8c3",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
        "#003c30",
    ],
    17: [
        "#000000",
        "#180c26",
        "#390947",
        "#6d005e",
        "#9e0066",
        "#d40060",
        "#ff0050",
        "#ff6438",
        "#ffaf21",
        "#ffff00",
    ],
    18: [
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#ffffbf",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
    ][::-1],
    19: [
        "#1a9850",
        "#66bd63",
        "#a6d96a",
        "#d9ef8b",
        "#ffffbf",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d73027",
    ],
    20: ["#a6611a", "#dfc27d", "#f5f5f5", "#80cdc1", "#018571"],
}
SELECTED_SCHEME_ID = 20  # 颜色方案
colors = COLOR_SCHEMES[SELECTED_SCHEME_ID]  # 提取方案
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    f"custom_scheme_{SELECTED_SCHEME_ID}", colors, N=256
)  # 创建自定义的线性分段颜色映射对象


# =========================================================================================
# ======================================3. 绘图函数 =========================================
# =========================================================================================
def draw_single_plot_content(ax, Xi, Yi, Zi, r2, rmse, title_str, bounds, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(Zi)
    if vmax is None:
        vmax = np.max(Zi)
    # 填充等高线绘制函数，
    cf = ax.contourf(
        Xi,  # X网格
        Yi,  # Y网格
        Zi,  # Z值，颜色值
        levels=20,  # 等高线层数
        cmap=custom_cmap,  # 颜色映射方案
        vmin=vmin,  # 范围最小值
        vmax=vmax,
    )  # 范围最大值
    # 调用等高线线条绘制函数，传入X网格
    ax.contour(
        Xi,  # 调用等高线线条绘制函数，传入X网格
        Yi,  # 传入Y网格
        Zi,  # 传入Z值
        levels=5,  # 线条层数
        colors="white",  # 线条颜色
        linewidths=1,  # 线条宽度
        alpha=1,
    )  # 线条透明度
    # line_levels = np.linspace(vmin, vmax, 6)
    # 绘制等高线
    # CS = ax.contour(Xi,  # X网格
    #                 Yi,  # Y网格
    #                 Zi,  # Z数值数据
    #                 levels=5,  #线条层数
    #                 colors='white',  #线条颜色
    #                 linewidths=1,  #线条宽度
    #                 alpha=0.8)  #线条透明度
    # #为等高线添加数值标注
    # ax.clabel(CS,  #指定要标注哪组线
    #           inline=True,  #打断线条
    #           fontsize=13,  #字体大小
    #           fmt='%.2f',  # 格式化字
    #           colors='white')  #颜色
    # 绘制等高线虚线
    ax.contour(
        Xi,  # X网格
        Yi,  # Y网格
        Zi,  # Z值
        levels=10,  # 线条层数
        colors="white",  # 线条颜色
        linewidths=1,  # 线条宽度
        linestyles="dashed",  # 线条样式
        alpha=1,
    )  # 透明度

    stat_str = f"R²: {r2:.2f} RMSE: {rmse:.3f}"  # 数值标注
    ax.text(
        0.02,  # X坐标
        1.02,  # Y坐标
        title_str,  # 文本内容
        transform=ax.transAxes,  # 指定坐标变换方式
        fontsize=10,  # 字体大小
        fontweight="bold",  # 字体加粗
        ha="left",
    )  # 水平对齐方式
    ax.text(
        0.98,  # X坐标
        1.02,  # Y坐标
        stat_str,  # 文本内容
        transform=ax.transAxes,  # 坐标变换方式
        fontsize=10,  # 字体大小
        ha="right",
    )  # 水平对齐方式

    for spine in ax.spines.values():  # 遍历坐标轴的所有边框
        spine.set_visible(False)  # 去掉

    x_min, x_max = bounds["x"]  # X轴范围
    y_min, y_max = bounds["y"]  # Y轴范围
    ax.set_xlim(x_min, x_max)  # 设置X轴的显示范围
    ax.set_ylim(y_min, y_max)  # 设置Y轴的显示范围
    x_ticks = np.linspace(x_min, x_max, 5)  # 生成均匀分布的X轴刻度
    y_ticks = np.linspace(y_min, y_max, 5)  # 生成均匀分布的Y轴刻度
    ax.set_xticks(x_ticks)  # 设置X轴刻度位置
    ax.set_xticklabels([f"{v:.0f}" for v in x_ticks])  # 设置X轴刻度标签
    ax.set_yticks(y_ticks)  # 设置Y轴刻度位置
    ax.set_yticklabels([f"{v:.0f}" for v in y_ticks])  # 设置Y轴刻度标签
    # 设置刻度参数
    ax.tick_params(
        axis="both",  # 应用于X和Y轴
        which="major",  # 应用于主刻度
        length=0,  # 设置刻度线长度
        labelsize=12,
    )  # 设置刻度标签字体大小
    return cf  # 返回填充等高线对象


# =========================================================================================
# ====================================== 7. 主程序执行 ========================================
# =========================================================================================
if __name__ == "__main__":
    df = pd.read_excel(str(DATA_DIR / "Data.xlsx"))  # 读取Excel数据文件
    SAVE_DIR = str(OUTPUT_DIR)  # 结果保存路径
    TARGET_COL_NAME = "Target_CE"  # 目标变量
    GROUP_ROW_COL_NAME = "Income_Group"  # 行分组
    GROUP_COL_COL_NAME = "Climate_Zone"  # 列分组
    ANALYSIS_X_COL_NAME = "BD"  # X轴对应的特征名称
    ANALYSIS_Y_COL_NAME = "BH"  # Y轴对应的特征名称
    exclude_cols = [
        TARGET_COL_NAME,  # 定义排除列列表，加入目标列名
        GROUP_ROW_COL_NAME,  # 加入行分组列名
        GROUP_COL_COL_NAME,
    ]  # 加入列分组列名
    feature_cols = [
        c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]  # 筛选出所有数值型且不在排除列表中的特征列
    print(f"特征: {feature_cols}")  # 打印筛选出的特征列列表
    analysis_x_idx = feature_cols.index(ANALYSIS_X_COL_NAME)  # 获取分析X列在特征列表中的索引
    analysis_y_idx = feature_cols.index(ANALYSIS_Y_COL_NAME)  # 获取分析Y列在特征列表中的索引
    row_unique_vals = df[GROUP_ROW_COL_NAME].dropna().unique()  # 获取行分组列的唯一值
    col_unique_vals = df[GROUP_COL_COL_NAME].dropna().unique()  # 获取列分组列的唯一值
    num_rows = len(row_unique_vals)  # 计算行分组的数量
    num_cols = len(col_unique_vals)  # 计算列分组的数量
    fig_width = 4 * num_cols  # 组合图宽度
    fig_height = 3.5 * num_rows  # 组合图高度
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(fig_width, fig_height)
    )  # 创建包含子图网格的主图形对象
    plt.subplots_adjust(hspace=0.15, wspace=0.15)  # 调整子图间的垂直和水平间距
    if num_rows == 1 and num_cols == 1:  # 检查是否只有1行1列
        axes = np.array([[axes]])  # 包装成二维数组
    elif num_rows == 1:  # 检查是否只有1行
        axes = axes.reshape(1, -1)  # 重塑为1行多列的二维数组
    elif num_cols == 1:  # 检查是否只有1列
        axes = axes.reshape(-1, 1)  # 重塑为多行1列的二维数组
    print(f"开始分析... \n行分组: {row_unique_vals} \n列分组: {col_unique_vals}")
    print(
        f"分析特征: {ANALYSIS_X_COL_NAME} (索引{analysis_x_idx}) & {ANALYSIS_Y_COL_NAME} (索引{analysis_y_idx})"
    )
    for i, row_val in enumerate(row_unique_vals):  # 遍历行分组
        for j, col_val in enumerate(col_unique_vals):  # 遍历列分组
            ax = axes[i, j]  # 获取当前位置的子图坐标轴对象
            # 对原始数据框进行布尔索引筛选
            sub_df = df[
                (df[GROUP_ROW_COL_NAME] == row_val)  # 筛选行分组列等于当前行标签的数据
                & (df[GROUP_COL_COL_NAME] == col_val)  # 且筛选列分组列等于当前列标签的数据
            ]
            # 提取特征 和 目标变量
            X_matrix = sub_df[feature_cols].values  # 获取特征列的数值，转换为numpy数组
            Y_target = sub_df[TARGET_COL_NAME].values  # 获取目标列的数值，转换为numpy数组
            # 获取用于绘图的特定特征数据
            x_data = sub_df[ANALYSIS_X_COL_NAME].values  # 提取分析用的X轴特征数据
            y_data = sub_df[ANALYSIS_Y_COL_NAME].values  # 提取分析用的Y轴特征数据
            # 边界字典，存储X和Y的范围
            bounds = {
                "x": (x_data.min(), x_data.max()),  # 计算并存储X轴特征的最小值和最大值
                "y": (y_data.min(), y_data.max()),  # 计算并存储Y轴特征的最小值和最大值
            }
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_matrix, Y_target, test_size=0.2, random_state=42
            )
            gam_terms = te(analysis_x_idx, analysis_y_idx, n_splines=10)  # 初始化GAM模型项

            for k in range(len(feature_cols)):  # 遍历所有特征的索引
                if k != analysis_x_idx and k != analysis_y_idx:  # 如果当前特征不是主要的X或Y特征
                    gam_terms += s(k)  # 将该特征作为平滑样条项加入模型，作为控制变量
            # 使用 gridsearch 自动寻找最优平滑参数
            # gam = LinearGAM(gam_terms).gridsearch(X_train, y_train)
            gam = LinearGAM(gam_terms).fit(X_train, y_train)  # 拟合GAM模型
            y_pred_test = gam.predict(X_test)  # 使用模型对测试集进行预测
            r2 = r2_score(y_test, y_pred_test)  # R2
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # RMSE
            p_values = gam.statistics_["p_values"]
            interaction_p_value = p_values[0]
            x_min, x_max = bounds["x"]  # 获取X轴最小值和最大值
            y_min, y_max = bounds["y"]  # 获取Y轴最小值和最大值
            xi = np.linspace(x_min, x_max, 100)  # 在X轴范围内生成100个等间距点
            yi = np.linspace(y_min, y_max, 100)  # 在Y轴范围内生成100个等间距点
            Xi, Yi = np.meshgrid(xi, yi)  # 生成网格坐标矩阵
            Xi_flat = Xi.ravel()  # 将X网格矩阵展平为一维数组
            Yi_flat = Yi.ravel()  # 将Y网格矩阵展平为一维数组
            n_grid = len(Xi_flat)  # 获取网格点总数
            grid_matrix = np.zeros((n_grid, len(feature_cols)))  # 初始化全零的预测输入矩阵
            mean_values = np.mean(X_test, axis=0)  # 计算测试集各特征的均值
            for k in range(len(feature_cols)):  # 遍历所有特征
                grid_matrix[:, k] = mean_values[k]  # 将矩阵对应列填充为该特征的均值
            grid_matrix[:, analysis_x_idx] = Xi_flat  # 将分析X列替换为网格点数据
            grid_matrix[:, analysis_y_idx] = Yi_flat  # 将分析Y列替换为网格点数据
            Zi = gam.predict(grid_matrix).reshape(Xi.shape)  # 预测网格点结果并重塑为二维网格形
            is_bottom_row = i == num_rows - 1  # 判断当前是否为最后一行
            is_left_col = j == 0  # 判断当前是否为第一列

            global_vmin = 0.0
            global_vmax = 0.4
            p_str = f"p < 0.001" if interaction_p_value < 0.001 else f"p={interaction_p_value:.3f}"
            hhstr = f"{row_val}-{col_val}"  # 构造子图标题
            title_str = f"{row_val}-{col_val}-({p_str})"  # 构造子图标题
            draw_single_plot_content(
                ax,  # 坐标轴对象
                Xi,  # X网格数据
                Yi,  # Y网格数据
                Zi,  # Z数据
                r2,  # 入R2
                rmse,  # RMSE
                title_str,  # 标题
                bounds,  # 边界信息
                vmin=global_vmin,
                vmax=global_vmax,
            )
            if is_bottom_row:  # 如果需要显示X轴标签
                ax.set_xlabel(
                    ANALYSIS_X_COL_NAME,  # 设置X轴标签文本
                    fontsize=13,  # 设置字体大小
                    fontweight="bold",
                )  # 设置字体加粗
            if is_left_col:  # 如果需要显示Y轴标签
                ax.set_ylabel(
                    ANALYSIS_Y_COL_NAME,  # 设置Y轴标签文本
                    fontsize=13,  # 设置字体大小
                    fontweight="bold",
                )  # 设置字体加粗
            # 保存单独图片
            fig_temp, ax_temp = plt.subplots(
                figsize=(4, 3.5)
            )  # 创建一个临时的图形和坐标轴用于单独保存
            draw_single_plot_content(ax_temp, Xi, Yi, Zi, r2, rmse, title_str, bounds)
            ax_temp.set_xlabel(
                ANALYSIS_X_COL_NAME,  # X轴标签
                fontsize=13,  # 字体大小
                fontweight="bold",
            )  # 字体加粗
            ax_temp.set_ylabel(
                ANALYSIS_Y_COL_NAME,  # Y轴标签
                fontsize=13,  # 字体大小
                fontweight="bold",
            )  # 字体加粗
            # 保存
            single_filename = os.path.join(SAVE_DIR, f"Subplot_{hhstr}.png")
            fig_temp.savefig(single_filename, dpi=300, bbox_inches="tight")
            plt.close(fig_temp)  # 关闭

    cbar_ax = fig.add_axes([0.38, 0.94, 0.5, 0.02])  # 添加一个新轴用于颜色条 [左, 底, 宽, 高]
    norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)  # 创建颜色归一化对象
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)  # 创建ScalarMappable对象关联颜色和归一化
    sm.set_array([])  # 设置空数据数组
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")  # 绘制水平颜色条
    cb.outline.set_visible(False)  # 隐藏颜色条轮廓线
    cb.ax.tick_params(length=0, labelsize=16)  # 颜色条刻度参数
    cb.set_label(f"CE(℃)", labelpad=-34, x=-0.08, fontsize=18, fontweight="bold")  # 颜色条标签
    ticks = np.linspace(global_vmin, global_vmax, 5)
    cb.set_ticks(ticks)  # 颜色条刻度值
    # 保存
    plt.savefig(
        str(OUTPUT_DIR / f"Analysis_Result_Scheme{SELECTED_SCHEME_ID}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(OUTPUT_DIR / f"Analysis_Result_Scheme{SELECTED_SCHEME_ID}.pdf"), bbox_inches="tight"
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

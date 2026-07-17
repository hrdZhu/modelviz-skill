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

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
# =========================================================================================
# ====================================== 2. 颜色库设置===============================
# =========================================================================================
color_schemes = {
    1: {"LST1": "#eb847b", "LST2": "#f6c096", "LST3": "#c1dbe8", "LST4": "#a3b7d6"},
    2: {"LST1": "#d62728", "LST2": "#ff7f0e", "LST3": "#17becf", "LST4": "#1f77b4"},
    3: {"LST1": "#4C72B0", "LST2": "#DD8452", "LST3": "#55A868", "LST4": "#C44E52"},
    4: {"LST1": "#1f77b4", "LST2": "#ff7f0e", "LST3": "#2ca02c", "LST4": "#d62728"},
    5: {"LST1": "#0173B2", "LST2": "#DE8F05", "LST3": "#029E73", "LST4": "#D55E00"},
    6: {"LST1": "#E41A1C", "LST2": "#FF7F00", "LST3": "#4DAF4A", "LST4": "#377EB8"},
    7: {"LST1": "#66C2A5", "LST2": "#FC8D62", "LST3": "#8DA0CB", "LST4": "#E78AC3"},
    8: {"LST1": "#A6CEE3", "LST2": "#1F78B4", "LST3": "#B2DF8A", "LST4": "#33A02C"},
    9: {"LST1": "#FBB4AE", "LST2": "#FED9A6", "LST3": "#B3CDE3", "LST4": "#CCEBC5"},
    10: {"LST1": "#8C564B", "LST2": "#BC8E5A", "LST3": "#8FBC8F", "LST4": "#708090"},
    11: {"LST1": "#003F5C", "LST2": "#367E8A", "LST3": "#85BCAF", "LST4": "#D1F2EB"},
    12: {"LST1": "#F94144", "LST2": "#F8961E", "LST3": "#F9C74F", "LST4": "#90BE6D"},
    13: {"LST1": "#5D5C61", "LST2": "#374B4A", "LST3": "#6369D1", "LST4": "#90BE6D"},
    14: {"LST1": "#D4B996", "LST2": "#A07C5B", "LST3": "#61412D", "LST4": "#3E281C"},
    15: {"LST1": "#F038FF", "LST2": "#F4A261", "LST3": "#2A9D8F", "LST4": "#264653"},
    16: {"LST1": "#7FC97F", "LST2": "#BEAED4", "LST3": "#FDC086", "LST4": "#FFFF99"},
    17: {"LST1": "#1B9E77", "LST2": "#D95F02", "LST3": "#7570B3", "LST4": "#E7298A"},
    18: {"LST1": "#4E79A7", "LST2": "#F28E2C", "LST3": "#E15759", "LST4": "#76B7B2"},
    19: {"LST1": "#333333", "LST2": "#666666", "LST3": "#999999", "LST4": "#CCCCCC"},
    20: {"LST1": "#4A148C", "LST2": "#7B1FA2", "LST3": "#AB47BC", "LST4": "#CE93D8"},
    21: {"LST1": "#d15b52", "LST2": "#e09a63", "LST3": "#7fb4cf", "LST4": "#5a7baa"},
}

selected_scheme = 1  # 选择要使用的配色方案
COLORS = color_schemes[selected_scheme]  # 提取配色
# =========================================================================================
# ====================================== 3.数据的加载及处理===============================
# =========================================================================================
# 读取数据
df = pd.read_excel(str(DATA_DIR / "data.xlsx"))
street_types = ["A", "B", "C", "D", "E"]  # 区域
lst_times = ["LST1", "LST2", "LST3", "LST4"]  # 定义目标变量
morph_features = ["BVF", "BCG", "SVF", "SCG", "TVF", "TCG"]  # 用于绘图的特征
covariates = ["WS", "ABH", "NDVI", "BD"]  # 其他特征
features_for_training = morph_features + covariates  # 完整的特征
required_cols = set(features_for_training + lst_times + ["street_type"])  # 特征+目标+分组

# =========================================================================================
# ====================================== 4.模型构建===============================
# =========================================================================================
plot_data_all_streets = {}  # 用于存储所有的 SHAP结果
# 超参数网格
param_grid = {
    "depth": [1, 2],
    "learning_rate": [0.05, 0.1],
}

for street_type in street_types:  # 遍历街道类型
    plot_data_all_streets[street_type] = {}  # 用于存储当前区域的结果
    df_street = df[df["street_type"] == street_type].copy()  # 筛选当前区域数据
    # 划分训练集和测试集
    train_data, test_data = train_test_split(df_street, test_size=0.3, random_state=42)
    X_train = train_data[features_for_training]  # 从训练集提取所有特征列
    X_test = test_data[features_for_training]  # 从测试集中提取所有特征列

    for lst_time in lst_times:  # 遍历目标
        print(f"  [{street_type}] 正在训练目标: {lst_time}")
        y_train = train_data[lst_time]  # 提取训练目标
        y_test = test_data[lst_time]  # 提取测试目标
        # 初始化CatBoostRegressor模型
        model_proto = CatBoostRegressor(
            iterations=100,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
            allow_writing_files=False,
        )
        # 初始化GridSearchCV，用于网格搜索
        grid_search = GridSearchCV(
            estimator=model_proto,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
        )
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_  # 获取最好的模型
        # 打印模型的精度
        train_r2 = best_model.score(X_train, y_train)
        test_r2 = best_model.score(X_test, y_test)
        print(f"模型R2-训练集: {train_r2:.4f}, 测试集: {test_r2:.4f}")
        # 模型的保存路径和文件名
        model_save_path = os.path.join(str(OUTPUT_DIR), f"model_{street_type}_{lst_time}.joblib")
        # 保存最佳模型
        joblib.dump(best_model, model_save_path)
        explainer = shap.TreeExplainer(best_model)  # 初始化一个SHAP解释器
        shap_values = explainer.shap_values(X_test)  # 计算SHAP值
        # 将SHAP值转换为DataFrame，并指定列名和索引
        shap_df = pd.DataFrame(shap_values, columns=features_for_training, index=X_test.index)
        # 拼接两 DataFrame
        plot_df = pd.concat(
            [
                X_test,  # 原始特征数据
                shap_df.add_suffix("_shap"),  # SHAP值
            ],
            axis=1,
        )  # 按列合并
        plot_data_all_streets[street_type][lst_time] = plot_df  # 存储到字典中


# =========================================================================================
# ====================================== 5.辅助函数==============================
# =========================================================================================
def calculate_adj_r2(x_data, y_data):
    X_poly = np.column_stack([x_data, x_data**2])  # 创建一个包含X和X2两列的矩阵,二阶多项式
    X_poly = sm.add_constant(X_poly, prepend=False)  # 为多项式矩阵添加常数项
    try:
        model_fit = sm.OLS(y_data, X_poly).fit()  # 使用普通最小二乘法拟合模型
        return f"{model_fit.rsquared_adj:.2f}"  # 返回拟合模型的调整后R2
    except (ValueError, np.linalg.LinAlgError):
        return "N/A"  # 如果拟合失败，则返回字符串N/A


# =========================================================================================
# ====================================== 6.注释文本添加函数==============================
# =========================================================================================
def add_plot_text(ax, r_squared_values, feature, street_type):
    fontsize = 22  # 文本的字体大小
    # 在图内右下角添加特征名称
    ax.text(
        0.95,
        0.05,
        f"{feature}({street_type.lower()})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=24,
    )
    # 获取每个目标y对应r2值
    r_val_1 = r_squared_values[0]
    r_val_2 = r_squared_values[1]
    r_val_3 = r_squared_values[2]
    r_val_4 = r_squared_values[3]
    text_x_pos = 0.78  # 左上角R2标注的起始x坐标，右边缘
    # 添加左上角的R2标注
    ax.text(
        text_x_pos,  # x
        0.95,  # y
        f"{r_val_4}",  # 内容
        transform=ax.transAxes,  # 坐标系
        ha="right",  # 水平对齐
        va="top",  # 处置对齐
        fontsize=fontsize,  # 大小
        color=COLORS["LST4"],
    )  # 颜色
    ax.text(
        text_x_pos - 0.12,
        0.95,
        f"{r_val_3};",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=COLORS["LST3"],
    )
    ax.text(
        text_x_pos - 0.24,
        0.95,
        f"{r_val_2};",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=COLORS["LST2"],
    )
    ax.text(
        text_x_pos - 0.36,
        0.95,
        f"{r_val_1};",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=COLORS["LST1"],
    )
    ax.text(
        text_x_pos - 0.48,
        0.95,
        "Adj.R² = ",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color="black",
        weight="bold",
    )


# =========================================================================================
# ====================================== 7.组合图绘制==============================
# =========================================================================================
def plot_shap_grid(plot_data_dict, colors_dict, features_list, streets_list):
    fig, axes = plt.subplots(6, 5, figsize=(32, 30), sharey=False)  # 创建子图网格
    for row_idx, feature in enumerate(features_list):  # 遍历特征列表
        for col_idx, street_type in enumerate(streets_list):  # 遍历区域列表
            ax = axes[row_idx, col_idx]  # 通过行列索引，获取当前循环对应的子图
            plot_data_for_street = plot_data_dict[street_type]  # 获取当前区域的所有LST的SHAP数据
            shap_col_name = f"{feature}_shap"  # 当前特征对应的SHAP列名
            r_squared_values = []  # 用于存储所有LST的R2值
            for lst_time in colors_dict.keys():  # 遍历颜色字典的键
                plot_data_for_time = plot_data_for_street[
                    lst_time
                ]  # 获取特定街道和特定 LST 时间的绘图数据 (DataFrame)
                # 调用辅助函数计算R2
                adj_r2 = calculate_adj_r2(
                    plot_data_for_time[feature].values, plot_data_for_time[shap_col_name].values
                )
                r_squared_values.append(adj_r2)  # 将计算得到的R2添加到列表中
                # 绘制散点图和回归拟合线
                sns.regplot(
                    x=plot_data_for_time[feature],  # 特征数据
                    y=plot_data_for_time[shap_col_name],  # 对应的SHAP值
                    ax=ax,  # 指定在哪个子图上绘制
                    color=colors_dict[lst_time],  # 颜色
                    label=lst_time,  # 图例标签
                    scatter_kws={"alpha": 1, "s": 16},  # 散点图的属性设置
                    line_kws={"lw": 1.5},  # 拟合曲线属性设置
                    order=2,  # 2阶多项式拟合
                    ci=95,  # 绘制95%的置信区间
                )

            ax.axhline(0, ls="--", color="grey", lw=1)  # 绘制SHAP值为0的基准线
            # 图框粗细
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            ax.set_xlabel(None)  # 去掉x轴标题
            ax.set_title(None)  # 去掉子图标题
            if col_idx == 0:  # 检查是否为第一列的子图
                ax.set_ylabel("", fontsize=24)  # 设置y轴标题
            else:
                ax.set_ylabel(None)  # 不设置
            # 设置x、y轴标注
            ax.tick_params(axis="both", which="major", labelsize=24, width=2)
            # 调用文本设置函数
            add_plot_text(ax, r_squared_values, feature, street_type)
    patches = [
        mpatches.Patch(color=colors_dict[time], label=time) for time in colors_dict.keys()
    ]  # 创建图例色块
    # 添加图例
    fig.legend(
        handles=patches,
        loc="lower right",
        bbox_to_anchor=(0.96, 0.03),
        ncol=4,
        fontsize=24,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # 自动调整布局
    # 保存
    fig.savefig(
        str(OUTPUT_DIR / f"shap_combined_{selected_scheme}.png"), dpi=300, bbox_inches="tight"
    )
    fig.savefig(str(OUTPUT_DIR / f"shap_combined_{selected_scheme}.pdf"), bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# ====================================== 8.子图绘制函数==============================
# =========================================================================================
def plot_and_save_individual_plots(plot_data_dict, colors_dict, features_list, streets_list):
    for feature in features_list:  # 遍历特征
        for street_type in streets_list:  # 遍历街道类型
            fig_single, ax_single = plt.subplots(1, 1, figsize=(7, 6))  # 创建图形
            plot_data_for_street = plot_data_dict[street_type]  # 获取当前区域的数据
            shap_col_name = f"{feature}_shap"  # 构造SHAP列名
            r_squared_values = []  # 初始化r1列表
            for lst_time in colors_dict.keys():  # 遍历目标
                plot_data_for_time = plot_data_for_street[lst_time]  # 获取当前目标
                # 计算R2
                adj_r2 = calculate_adj_r2(
                    plot_data_for_time[feature].values, plot_data_for_time[shap_col_name].values
                )
                r_squared_values.append(adj_r2)  # 存储R2到列表
                # 绘图
                sns.regplot(
                    x=plot_data_for_time[feature],  # x轴
                    y=plot_data_for_time[shap_col_name],  # y轴
                    ax=ax_single,  # 在当前这个独立的子图上绘制
                    color=colors_dict[lst_time],  # 颜色
                    label=lst_time,  # 标签
                    scatter_kws={"alpha": 1, "s": 10},  # 散点设置
                    line_kws={"lw": 1},  # 拟合曲线设置
                    order=2,  # 2阶拟合
                    ci=95,  # 置信区间
                )
            ax_single.axhline(0, ls="--", color="grey", lw=1)  # 绘制基准线
            ax_single.set_xlabel(None)  # 去掉x轴标题
            ax_single.set_ylabel("SHAP Value (Contribution)", fontsize=16)  # y轴标题
            ax_single.set_title(None)  # 去掉标题
            ax_single.tick_params(axis="both", which="major", labelsize=16)  # 设置x、y轴标注
            # 添加文本标注
            add_plot_text(ax_single, r_squared_values, feature, street_type)
            patches = [
                mpatches.Patch(color=colors_dict[time], label=time) for time in colors_dict.keys()
            ]  # 创建图例元素
            # 添加图例
            fig_single.legend(
                handles=patches,
                loc="lower right",
                bbox_to_anchor=(0.9, -0.02),
                ncol=4,
                fontsize=10,
                frameon=False,
            )
            file_name = f"shap_{feature}_{street_type}"  # 构造文件名
            # 保存
            fig_single.savefig(str(OUTPUT_DIR / f"{file_name}.png"), dpi=200, bbox_inches="tight")
            fig_single.savefig(str(OUTPUT_DIR / f"{file_name}.pdf"), bbox_inches="tight")
            plt.close(fig_single)


# =========================================================================================
# ====================================== 9.执行绘图==============================
# =========================================================================================
# 绘制组合图
plot_shap_grid(
    plot_data_all_streets,  # SHAP数据
    COLORS,  # 颜色方案
    morph_features,  # 特征
    street_types,  # 区域
)

# 绘制并保存所有独立子图
plot_and_save_individual_plots(
    plot_data_all_streets,  # SHAP数据
    COLORS,  # 颜色方案
    morph_features,  # 特征
    street_types,  # 区域
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

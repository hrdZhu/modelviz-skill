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

import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.unicode_minus"] = False


def draw_single_plot(
    ax, y_true_train, y_pred_train, y_true_test, y_pred_test, model_name, colors, is_subplot=False
):  # 用于绘制单个模型的回归散点图
    train_r2 = r2_score(y_true_train, y_pred_train)  # 计算训练集的R2分数
    test_r2 = r2_score(y_true_test, y_pred_test)  # 计算测试集的R2分数
    train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))  # 计算训练集的均方根误差
    test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))  # 计算测试集的均方根误差
    y_true_full = np.concatenate(
        (y_true_train, y_true_test)
    )  # 合并训练集和测试集的真实值，用于确定绘图范围
    y_pred_full = np.concatenate(
        (y_pred_train, y_pred_test)
    )  # 合并训练集和测试集的预测值，用于确定绘图范围
    ax.scatter(
        y_true_train,
        y_pred_train,
        marker="^",
        color=colors["train_scatter"],
        alpha=0.6,
        s=50,
        label="Training Data",
    )  # 绘制训练集数据的散点图
    ax.scatter(
        y_true_test,
        y_pred_test,
        marker="o",
        facecolors=colors["test_scatter_face"],
        edgecolors=colors["test_scatter_edge"],
        s=70,
        label="Test Data",
    )  # 绘制测试集数据的散点图
    plot_min = (
        min(y_true_full.min(), y_pred_full.min()) * 1.1
    )  # 计算绘图范围的最小值，并增加 10% 的边距
    plot_max = (
        max(y_true_full.max(), y_pred_full.max()) * 1.1
    )  # 计算绘图范围的最大值，并增加 10% 的边距
    lims = [plot_min, plot_max]  # 将最小值和最大值存入列表，作为坐标轴的范围
    ax.plot(lims, lims, colors["ideal_line_style"], label="Ideal Line (y=x)")  # 绘制1：1线

    # 将pandas Series转换为NumPy数组，以使用位置索引而不是标签索引
    y_true_test_np = np.asarray(y_true_test)
    y_pred_test_np = np.asarray(y_pred_test)

    # 获取排序所需的位置索引
    sort_indices = np.argsort(y_true_test_np)
    # 使用位置索引来排序NumPy数组
    y_true_test_sorted = y_true_test_np[sort_indices]
    y_pred_test_sorted = y_pred_test_np[sort_indices]

    X_fit_test = sm.add_constant(y_true_test_sorted)  # 为排序后的测试集真实值添加常数项
    ols_model_test = sm.OLS(y_pred_test_sorted, X_fit_test).fit()  # 仅使用测试集数据进行OLS拟合
    y_pred_sorted = ols_model_test.predict(X_fit_test)
    # 获取拟合模型的预测结果，包括置信区间
    predictions_summary = ols_model_test.get_prediction(X_fit_test).summary_frame(alpha=0.05)
    y_fit = predictions_summary["mean"]  # 提取预测的均值，即拟合线
    lower_bound = predictions_summary["obs_ci_lower"]  # 提取 95% 置信区间的下界
    upper_bound = predictions_summary["obs_ci_upper"]  # 提取 95% 置信区间的上界

    # 使用传入的 `colors` 字典设置拟合线样式和置信区间颜色
    ax.plot(
        y_true_test_sorted,
        y_pred_sorted,
        colors["fit_line_style"],
        linewidth=2,
        label="Fitted Line",
    )  # 绘制拟合线
    ax.fill_between(
        y_true_test_sorted,
        lower_bound,
        upper_bound,
        color=colors["confidence_interval"],
        alpha=0.25,
        label="95% Confidence Interval",
    )  # 置信区间

    tick_label_size = 10  # 设置刻度标签的字体大小
    axis_label_size = 12  # 设置坐标轴标签的字体大小
    tick_length = 6  # 设置刻度线的长度
    tick_width = 1.5  # 设置刻度线的宽度
    frame_width = 1.5  # 设置图框（坐标轴边框）的宽度
    if not is_subplot:  # 判断当前绘图是否是子图
        ax.set_xlabel("Actual Values", fontsize=axis_label_size)  # 如果不是子图，则设置 X 轴标签
        ax.set_ylabel("Predicted Values", fontsize=axis_label_size)  # 如果不是子图，则设置 Y 轴标签
    ax.tick_params(
        axis="both", which="major", labelsize=tick_label_size, length=tick_length, width=tick_width
    )  # 设置主刻度的样式
    for spine in ax.spines.values():  # 遍历图框的四个边
        spine.set_linewidth(frame_width)  # 设置每个边的宽度
    ax.set_xlim(lims)  # 设置 X 轴的显示范围
    ax.set_ylim(lims)  # 设置 Y 轴的显示范围
    ax.legend(loc="upper left", fontsize=10)  # 在左上角显示图例，并设置字体大小
    stats_text = (  # 创建一个包含模型性能指标的字符串
        f"$\\bf{{{model_name}}}$\n"  # 模型名称
        f"Train R$^2$: {train_r2:.3f}\n"  # 训练集 R²
        f"Test R$^2$: {test_r2:.3f}\n"  # 测试集 R²
        f"Train RMSE: {train_rmse:.3f}\n"  # 训练集 RMSE
        f"Test RMSE: {test_rmse:.3f}"  # 测试集 RMSE
    )
    ax.text(
        0.95,
        0.05,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
    )  # 在图的右下角添加性能指标文本框


def plot_regression_results(
    y_true_train, y_pred_train, y_true_test, y_pred_test, model_name, colors, save_path=None
):  # 用于绘制并保存单个模型的性能图
    fig, ax = plt.subplots(figsize=(10, 7))  # 创建一个图形和一个坐标轴，设置图形大小为 10x7 英寸
    # 调用内部绘图函数时，传入 `colors` 参数
    draw_single_plot(
        ax,
        y_true_train,
        y_pred_train,
        y_true_test,
        y_pred_test,
        model_name,
        colors,
        is_subplot=False,
    )  # 调用内部绘图函数来绘制图形内容
    ax.set_title(f"Model Performance: {model_name}", fontsize=18, pad=15)  # 设置图形的标题
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close("all")  # Interactive display removed; assets were exported above.


if __name__ == "__main__":
    # 选择配色方案
    color_scheme_choice = 10
    # 配色库
    color_schemes = {
        1: {
            "train_scatter": "#0072B2",
            "test_scatter_face": "#E69F00",
            "test_scatter_edge": "black",
            "ideal_line_style": "k--",
            "fit_line_style": "#D55E00",
            "confidence_interval": "#F0E442",
        },
        2: {
            "train_scatter": "#2E8B57",
            "test_scatter_face": "#90EE90",
            "test_scatter_edge": "#2F4F4F",
            "ideal_line_style": "k--",
            "fit_line_style": "#8B4513",
            "confidence_interval": "#D2B48C",
        },
        3: {
            "train_scatter": "#008080",
            "test_scatter_face": "#AFEEEE",
            "test_scatter_edge": "#4682B4",
            "ideal_line_style": "b--",
            "fit_line_style": "#DC143C",
            "confidence_interval": "#FFB6C1",
        },
        4: {
            "train_scatter": "#483D8B",
            "test_scatter_face": "#E6E6FA",
            "test_scatter_edge": "#6A5ACD",
            "ideal_line_style": "k--",
            "fit_line_style": "#C71585",
            "confidence_interval": "#DB7093",
        },
        5: {
            "train_scatter": "#FF4500",
            "test_scatter_face": "#FFDAB9",
            "test_scatter_edge": "#A52A2A",
            "ideal_line_style": "k--",
            "fit_line_style": "#006400",
            "confidence_interval": "#98FB98",
        },
        6: {
            "train_scatter": "#555555",
            "test_scatter_face": "#DDDDDD",
            "test_scatter_edge": "black",
            "ideal_line_style": "k--",
            "fit_line_style": "black",
            "confidence_interval": "#AAAAAA",
        },
        7: {  #
            "train_scatter": "#440154",
            "test_scatter_face": "#21908d",
            "test_scatter_edge": "black",
            "ideal_line_style": "k--",
            "fit_line_style": "#fde725",
            "confidence_interval": "#5dc863",
        },
        8: {
            "train_scatter": "#C71585",
            "test_scatter_face": "#FFD700",
            "test_scatter_edge": "#B8860B",
            "ideal_line_style": "k--",
            "fit_line_style": "#8B0000",
            "confidence_interval": "#FFC0CB",
        },
        9: {
            "train_scatter": "#5F9EA0",
            "test_scatter_face": "#F4A460",
            "test_scatter_edge": "#8B4513",
            "ideal_line_style": "k--",
            "fit_line_style": "#A0522D",
            "confidence_interval": "#ADD8E6",
        },
        10: {
            "train_scatter": "#FF00FF",
            "test_scatter_face": "#00FFFF",
            "test_scatter_edge": "black",
            "ideal_line_style": "k--",
            "fit_line_style": "#32CD32",
            "confidence_interval": "#B0E0E6",
        },
    }
    selected_colors = color_schemes.get(color_scheme_choice, color_schemes[1])
    print(f"选择配色方案 {color_scheme_choice} ")
    file_path = str(DATA_DIR / "04_final_features_and_target.xlsx")  # 原始数据
    # 已经训练好的本地模型
    trained_models_paths = {
        "MLR": str(DATA_DIR / "Linear_Regression.joblib"),
        "SVR": str(DATA_DIR / "SVR.joblib"),
        "RF": str(DATA_DIR / "Random_Forest.joblib"),
        "LightGBM": str(DATA_DIR / "LightGBM.joblib"),
        "CatBoost": str(DATA_DIR / "CatBoost.joblib"),
        "XGBoost": str(DATA_DIR / "XGBoost.joblib"),
    }
    output_folder = str(OUTPUT_DIR)  # 绘图结果的保存位置
    if not os.path.exists(output_folder):  # 如果没有文件夹，则进行创建
        os.makedirs(output_folder)
    data = pd.read_excel(file_path)  # 读取数据
    X = data.iloc[:, :-1]  # 特征变量
    y = data.iloc[:, -1]  # 目标变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for model_name, model_path in trained_models_paths.items():  # 遍历模型
        print(f"正在处理模型: {model_name}")
        model = joblib.load(model_path)  # 从文件加载已训练好的模型
        y_pred_train = model.predict(X_train)  # 使用模型对训练集进行预测
        y_pred_test = model.predict(X_test)  # 使用模型对测试集进行预测
        safe_model_name = model_name.replace(
            " ", "_"
        )  # 将模型名称中的空格替换为下划线，以创建安全的文件名
        save_filename = os.path.join(
            output_folder, f"{safe_model_name}_{color_scheme_choice}.png"
        )  # 在文件名中加入配色方案编号
        plot_regression_results(
            y_train,
            y_pred_train,
            y_test,
            y_pred_test,
            model_name=model_name,
            colors=selected_colors,
            save_path=save_filename,
        )  # 调用函数为当前模型生成并保存结果图
    models_list = list(trained_models_paths.items())  # 将模型字典转换为列表，以便按索引访问
    nrows, ncols = 3, 2  # 设置组合图的行数和列数
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 15))  # 创建一个包含 3x2 个子图的图形
    axes = axes.flatten()  # 将 2D 的 axes 数组展平为 1D 数组，方便遍历
    for i, (model_name, model_path) in enumerate(models_list):  # 遍历模型列表，并获取索引 i
        if i < len(axes):  # 确保模型数量不超过子图数量
            ax = axes[i]  # 获取当前要绘制的子图坐标轴
            print(f"--- 正在向组合图添加: {model_name} ---")
            model = joblib.load(model_path)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            # 在子图绘制函数中，同样传入选择的配色方案字典 `selected_colors`
            draw_single_plot(
                ax,
                y_train,
                y_pred_train,
                y_test,
                y_pred_test,
                model_name,
                colors=selected_colors,
                is_subplot=True,
            )  # 在当前子图上绘制模型性能图
            subplot_label = chr(ord("a") + i)  # 生成子图标签
            ax.text(
                0.05,
                1.02,
                f"$\\bf{{{subplot_label}}}$",
                transform=ax.transAxes,
                fontsize=16,
                va="bottom",
                ha="left",
            )  # 在每个子图的左上角添加标签
    for j in range(i + 1, len(axes)):  # 遍历剩余的、未使用的子图
        axes[j].axis("off")  # 关闭这些子图的坐标轴，使其不可见
    fig.supxlabel(
        "Actual effluent TN (mg/L)", fontsize=18, y=0.07
    )  # 为整个图形添加一个共享的 X 轴总标签
    fig.supylabel(
        "Predicted effluent TN (mg/L)", fontsize=18, x=0.06
    )  # 为整个图形添加一个共享的 Y 轴总标签
    plt.tight_layout(rect=[0.08, 0.08, 1, 1])  # 调整布局
    composite_save_path = os.path.join(
        output_folder, f"composite_plot_scheme_{color_scheme_choice}.png"
    )  # 保存组合图
    plt.savefig(composite_save_path, dpi=300, bbox_inches="tight")
    print(f"组合图已成功保存至:{composite_save_path}")
    # plt.show() # 显示图形

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

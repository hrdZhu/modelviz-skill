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
# ======================================1.库的导入=========================================
# =========================================================================================
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.patches import Patch
from PIL import Image
from pygam import LinearGAM, s
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"

# =========================================================================================
# ======================================2.颜色库=========================================
# =========================================================================================
# 颜色库
color_library = {
    1: {
        "main_curve": "#3B5998",
        "confidence_interval": "#3B5998",
        "positive_fill": "#C8E6C9",
        "negative_fill": "#FFCDD2",
    },
    2: {
        "main_curve": "#E67E22",
        "confidence_interval": "#E67E22",
        "positive_fill": "#A2D9CE",
        "negative_fill": "#F5B7B1",
    },
    3: {
        "main_curve": "#3498DB",
        "confidence_interval": "#3498DB",
        "positive_fill": "#D6EAF8",
        "negative_fill": "#FAD7A0",
    },
    4: {
        "main_curve": "#8E44AD",
        "confidence_interval": "#8E44AD",
        "positive_fill": "#E8DAEF",
        "negative_fill": "#FDEBD0",
    },
    5: {
        "main_curve": "#27AE60",
        "confidence_interval": "#27AE60",
        "positive_fill": "#D5F5E3",
        "negative_fill": "#FAE5D3",
    },
    6: {
        "main_curve": "#C0392B",
        "confidence_interval": "#C0392B",
        "positive_fill": "#D4E6F1",
        "negative_fill": "#FCF3CF",
    },
    7: {
        "main_curve": "#2C3E50",
        "confidence_interval": "#2C3E50",
        "positive_fill": "#EAEDED",
        "negative_fill": "#FFEBEE",
    },
    8: {
        "main_curve": "#556B2F",
        "confidence_interval": "#556B2F",
        "positive_fill": "#E9F7EF",
        "negative_fill": "#FFEBCD",
    },
    9: {
        "main_curve": "#D24D57",
        "confidence_interval": "#D24D57",
        "positive_fill": "#E0F2F1",
        "negative_fill": "#FFF9C4",
    },
    10: {
        "main_curve": "#283593",
        "confidence_interval": "#283593",
        "positive_fill": "#E3F2FD",
        "negative_fill": "#FFF3E0",
    },
    11: {
        "main_curve": "#FF7F50",
        "confidence_interval": "#FF7F50",
        "positive_fill": "#E0FFFF",
        "negative_fill": "#FFF5EE",
    },
    12: {
        "main_curve": "#228B22",
        "confidence_interval": "#228B22",
        "positive_fill": "#F0FFF4",
        "negative_fill": "#FDF5E6",
    },
    13: {
        "main_curve": "#4682B4",
        "confidence_interval": "#4682B4",
        "positive_fill": "#F0F8FF",
        "negative_fill": "#FFF0F5",
    },
    14: {
        "main_curve": "#FFC107",
        "confidence_interval": "#FFC107",
        "positive_fill": "#E6F7FF",
        "negative_fill": "#FFFFF0",
    },
    15: {
        "main_curve": "#B22222",
        "confidence_interval": "#B22222",
        "positive_fill": "#F5FFFA",
        "negative_fill": "#FFE4C4",
    },
    16: {
        "main_curve": "#5D4037",
        "confidence_interval": "#5D4037",
        "positive_fill": "#F1F8E9",
        "negative_fill": "#FFFDE7",
    },
    17: {
        "main_curve": "#00838F",
        "confidence_interval": "#00838F",
        "positive_fill": "#E0F7FA",
        "negative_fill": "#FFEFE2",
    },
    18: {
        "main_curve": "#4B0082",
        "confidence_interval": "#4B0082",
        "positive_fill": "#F3E5F5",
        "negative_fill": "#FFFACD",
    },
    19: {
        "main_curve": "#008080",
        "confidence_interval": "#008080",
        "positive_fill": "#E0F2F1",
        "negative_fill": "#FBEAE5",
    },
    20: {
        "main_curve": "#FF4500",
        "confidence_interval": "#FF4500",
        "positive_fill": "#EDFDF4",
        "negative_fill": "#FFF6ED",
    },
}


# =========================================================================================
# ======================================3.GAM拟合和绘图函数=========================================
# =========================================================================================
# def plot_gam_style_from_shap(feature_x_std, feature_x_orig, shap_y, feature_name, plot_label, output_dir, colors):
def plot_gam_style_from_shap(
    feature_x_std,
    feature_x_orig,
    shap_y,
    feature_name,
    plot_label,
    output_dir,
    colors,
    feature_mean,
    feature_std,
):
    # 按标准化后的特征值进行排序，同时保持原始值和SHAP值的对应关系
    sort_order = np.argsort(feature_x_std)  # 获取根据特征值从小到大排序后的索引顺序。
    feature_x_std_sorted = feature_x_std[sort_order]  # 根据排序索引，对标准化特征值数组进行排序。
    feature_x_orig_sorted = feature_x_orig[
        sort_order
    ]  # 根据相同排序索引，对原始特征值数组进行排序。
    shap_y_sorted = shap_y[
        sort_order
    ]  # 根据相同的排序索引，对SHAP值数组进行排序，以保持数据点对应关系。

    # 在标准化数据上拟合GAM模型s()代表spline。告诉GAM不要用一条简单的直线去拟合数据，而是用一条滑的曲线0:代表这个平滑项作用于输入数据第一列。每次只处理一个特征，所以这里总是0
    # lam平滑惩罚系数，值越大，惩罚就越重，越平滑，n_splines样条基函数的数量，捕捉细节越多
    gam = LinearGAM(s(0, n_splines=20, lam=0.6)).fit(
        feature_x_std_sorted, shap_y_sorted
    )  # 构建并拟合GAM模型

    # 创建平滑曲线的X坐标
    x_smooth_std = feature_x_std_sorted  # np.linspace(feature_x_std_sorted.min(), feature_x_std_sorted.max(), 500)
    # 使用GAM模型在标准化的X坐标上进行预测
    y_smooth = gam.predict(
        x_smooth_std
    )  # 使用训练好的GAM模型，在平滑的x轴坐标上进行预测，得到平滑的y轴坐标。
    # 计算95%的置信区间
    confidence_interval_bands = gam.prediction_intervals(
        x_smooth_std, width=0.95
    )  # 计算在x_smooth_std上每个点的95%置信区间

    # # 建立一个从标准化尺度到原始尺度的线性映射，用于转换X轴
    # # 通过线性回归，找到标准化值与原始值之间的换算关系（斜率和截距）
    # map_slope, map_intercept, _, _, _ = linregress(feature_x_std_sorted, feature_x_orig_sorted)
    # # 应用这个换算关系，将用于绘图的平滑X轴坐标从标准化尺度转换回原始数据尺度
    # x_smooth_orig = x_smooth_std * map_slope + map_intercept
    # 使用传入的均值和标准差，将平滑X轴坐标从标准化尺度转换回原始数据尺度
    x_smooth_orig = x_smooth_std * feature_std + feature_mean

    # 找到平滑曲线与y=0的交点（在标准化尺度上计算）
    intersection_spline = UnivariateSpline(
        x_smooth_std, y_smooth, s=0
    )  # 基于平滑曲线的点，创建一个单变量样条插值函数
    intersection_points_x_std = (
        intersection_spline.roots()
    )  # 调用roots()方法找到样条函数等于零的根（标准化尺度）
    # 将交点转换回原始数据尺度
    # intersection_points_x_orig = intersection_points_x_std * map_slope + map_intercept
    intersection_points_x_orig = intersection_points_x_std * feature_std + feature_mean

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.set_facecolor("white")  # 设置图形区域的背景颜色为白色。
    ax.set_facecolor("#EAEAEA80")  # 设置主图的背景色

    spine_width = 3  # 设置图框的粗细
    ax.spines["top"].set_linewidth(spine_width)  # 设置上图框
    ax.spines["bottom"].set_linewidth(spine_width)  # 下图框
    ax.spines["left"].set_linewidth(spine_width)  # 左图框
    ax.spines["right"].set_linewidth(spine_width)  # 右图框
    tick_length = 6  # 刻度线的长短
    tick_width = 1.5  # 设置刻度线的粗细
    ax.tick_params(
        axis="both", which="major", direction="in", length=tick_length, width=tick_width
    )  # 设置刻度线的朝内朝外、长短、粗细
    label_size = 26  # 坐标轴数值标注大小
    label_weight = "bold"  # 坐标轴数值标注的粗细
    for label in ax.get_xticklabels() + ax.get_yticklabels():  # 遍历X轴和Y轴的所有的刻度的数值标注
        label.set_fontsize(label_size)  # 字体大小
        label.set_fontweight(label_weight)  # 字体粗细

    # 使用GAM的结果进行绘图（X轴使用转换回原始尺度的坐标）
    ax.plot(
        x_smooth_orig, y_smooth, color=colors["main_curve"], linewidth=2.5, label="SHAP(GAM)"
    )  # 绘制平滑的GAM曲线

    # 在置信区间的下界和上界之间填充颜色，可视化95%的置信区间。
    ax.fill_between(
        x_smooth_orig,
        confidence_interval_bands[:, 0],
        confidence_interval_bands[:, 1],
        color=colors["confidence_interval"],
        alpha=0.2,
        edgecolor="none",
    )
    ax.axhline(
        0, color="gray", linestyle="--", linewidth=1
    )  # 在y=0的位置绘制一条灰色虚线作为参考线。
    # 在平滑曲线大于等于0的部分，填充曲线与y=0轴之间的区域
    ax.fill_between(
        x_smooth_orig,
        y_smooth,
        0,
        where=(y_smooth >= 0),
        color=colors["positive_fill"],
        alpha=0.5,
        interpolate=True,
        zorder=0,
    )
    # 在平滑曲线小于等于0的部分，填充曲线与y=0轴之间的区域
    ax.fill_between(
        x_smooth_orig,
        y_smooth,
        0,
        where=(y_smooth <= 0),
        color=colors["negative_fill"],
        alpha=0.5,
        interpolate=True,
        zorder=0,
    )

    # 在原始数据尺度的交点位置进行标注
    if len(intersection_points_x_orig) > 0:  # 检查是否存在曲线与y=0的交点。
        unique_points = np.unique(
            np.round(intersection_points_x_orig, 2)
        )  # 对交点的x坐标四舍五入到两位小数并去重。
        sorted_points = sorted(unique_points)  # 对去重后的交点进行排序。
        points_to_plot = (
            sorted_points[:2] + sorted_points[-2:] if len(sorted_points) > 4 else sorted_points
        )  # 如果交点超过4个，只选择最小的两个和最大的两个进行标注。
        ax.scatter(
            points_to_plot,
            [0] * len(points_to_plot),
            color="#D32F2F",
            s=80,
            zorder=10,
            edgecolor="white",
            linewidth=1.5,
        )  # 在要标注的交点位置绘制红色散点。
        pos_offset, neg_offset = (
            (y_smooth.max() - y_smooth.min()) * 0.05,
            -(y_smooth.max() - y_smooth.min()) * 0.2,
        )  # 计算文本标签在y轴上的正向和负向偏移量。
        for i, x_val in enumerate(points_to_plot):  # 遍历每一个要标注的交点。
            ax.axvline(
                x=x_val, color="red", linestyle="--", alpha=0.6
            )  # 在交点的x坐标位置绘制一条红色垂直虚线。
            y_offset = (
                pos_offset if i % 2 == 0 else neg_offset
            )  # 交错设置文本标签的垂直偏移量，防止重叠。
            va = "bottom" if i % 2 == 0 else "top"  # 根据偏移量设置文本的垂直对齐方式。
            ax.text(
                x_val,
                y_offset,
                f"{x_val:.2f}",
                ha="center",
                va=va,
                fontsize=26,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
            )  # 在交点附近添加文本标签。

    legend_elements = [
        Patch(
            facecolor=colors["positive_fill"], alpha=0.8, label="Positive"
        ),  # 创建一个图例元素，设置正向影响区域
        Patch(facecolor=colors["negative_fill"], alpha=0.8, label="Negative"),
    ]  # 创建一个图例元素，设置负向影响区域
    ax.legend(
        handles=legend_elements, loc="lower right", fontsize=26, frameon=True, edgecolor="black"
    )  # 自定义的图例，设置其位置和字体大小。

    # R2、想关系数、显著性计算（在原始数据和SHAP值之间计算）
    _, _, r_value, p_value, _ = linregress(
        feature_x_orig, shap_y
    )  # 对原始特征值和SHAP值进行线性回归
    r_squared = r_value**2  # 计算决定系数R2
    if p_value < 0.001:  # 判断p值是否小于0.001
        p_text = "$p < 0.001$"  # 如果是格式化为p<0.001
    else:
        p_text = f"$p = {p_value:.3f}$"  # 如果不是显示实际值

    ax.text(
        0.05,
        0.9,
        f"$R^2 = {r_squared:.3f}$",
        transform=ax.transAxes,
        fontsize=26,
        fontweight="bold",
    )  # 设置R2的文本标注
    ax.text(
        0.75, 0.9, p_text, transform=ax.transAxes, fontsize=26, fontweight="bold"
    )  # 设置显著性的p值标注

    ax.text(
        0.05, 0.05, f"({plot_label})", transform=ax.transAxes, fontsize=26, fontweight="bold"
    )  # 在图的左下角添加子图标签
    ax.set_xlabel(f"{feature_name}", fontsize=26, fontweight="bold")  # 设置X轴的标签为特征名称。
    ax.set_ylabel("SHAP Value", fontsize=26, fontweight="bold")  # 设置Y轴的标题
    ax.set_xlim(
        feature_x_orig.min() * 1.1, feature_x_orig.max() * 1.1
    )  # 设置X轴的显示范围（使用原始数据范围）
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域，防止标签重叠。
    # 保存
    file_basename = (
        f"{plot_label}_{feature_name.replace(' ', '_')}_gam_style_shap"  # 构建保存文件的基本名称
    )
    png_filename = os.path.join(
        output_dir, f"{file_basename}_{selected_color}.png"
    )  # PNG文件的保存
    pdf_filename = os.path.join(output_dir, f"{file_basename}_{selected_color}.pdf")  # PDF的保存
    plt.savefig(png_filename, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.close(fig)
    return png_filename


if __name__ == "__main__":
    # =========================================================================================
    # =====================================4.数据得输入、结果输出设置=========================================
    # =========================================================================================
    excel_path = str(DATA_DIR / "simulated_regression_data.xlsx")  # 输入数据的路径。
    output_dir = str(OUTPUT_DIR)  # 输出结果
    target_column_name = "Target"  # 目标变量
    os.makedirs(output_dir, exist_ok=True)  # 如果不存在就创建
    data = pd.read_excel(excel_path)  # 读取数据

    X = data.drop(columns=[target_column_name])  # 提取特征数据
    y = data[target_column_name]  # 提取目标数据
    # =========================================================================================
    # =====================================5.标准化处理=========================================
    # =========================================================================================
    # 将数据划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 划分后，复制一份原始的测试集特征数据，用于后续绘图
    X_test_original = X_test.copy()
    # 标准化处理
    scaler = StandardScaler()  # 初始化标准化器
    scaler.fit(X_train)
    # 分别对训练集和测试集进行转换
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # 将标准化后的numpy数组转回DataFrame，保留列名
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    # =========================================================================================
    # =====================================6.超参数设置，寻找最佳模型=========================================
    # =========================================================================================
    # 设置超参数，这是我图方便瞎写得，你的改
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
    }
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)  # 模型实例化
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )  # 设置网格搜索
    # 在标准化的训练数据上执行网格搜索寻找最佳参数组合
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    print("最佳超参数: ", grid_search.best_params_)

    # ==================================================================================
    # 切换配色方案
    selected_color = 1
    colors = color_library[selected_color]  # 根据索引从颜色库中获取配色方案
    # ==================================================================================

    # =========================================================================================
    # =====================================7.使用最佳模型进行shap分析，使用测试集，选出最重要特征=========================================
    # =========================================================================================
    explainer = shap.TreeExplainer(model)  # 创建一个针对树模型的SHAP解释器实例
    shap_values = explainer.shap_values(X_test)  # 使用解释器计算测试集X中每个样本每个特征的SHAP值

    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # 计算每个特征SHAP值的绝对值的平均值
    importance_df = pd.DataFrame(
        {"feature": X_test.columns, "importance": mean_abs_shap}
    ).sort_values("importance", ascending=False)
    # 要绘制得特征，这里默认是最重要得9个
    num_top_features_to_plot = 9
    sorted_features_to_plot = (
        importance_df["feature"].head(num_top_features_to_plot).tolist()
    )  # 获取排序后最重要的N个特征名称
    print(
        f"-->选择并绘制测试集上最重要的 {len(sorted_features_to_plot)} 个特征: {sorted_features_to_plot}"
    )
    # =========================================================================================
    # ====================================8.开始绘图=========================================
    # =========================================================================================
    generated_png_paths = []  # 初始化一个空列表，用于存储所有生成的所有图的路径
    feature_indices = {name: i for i, name in enumerate(X_test.columns)}  # 创建特征名称到索引的映射

    for i, feature_name in enumerate(sorted_features_to_plot):  # 遍历排序后的重要特征列表。
        print(f"--- 正在处理第 {i + 1} 个特征: {feature_name} ---")
        plot_label = chr(ord("a") + i)  # 生成子图标签，如 a, b, c
        feature_idx = feature_indices[feature_name]  # 使用字典查找当前特征的索引。

        # # 提取当前特征的标准化数据(来自X_test)、原始数据(来自X_test_original)和对应的SHAP值
        # feature_x_data_std = X_test[feature_name].values
        # feature_x_data_orig = X_test_original[feature_name].values
        # shap_y_data = shap_values[:, feature_idx]
        #
        # # 调用绘图函数
        # png_path = plot_gam_style_from_shap(
        #     feature_x_data_std,  # 当前特征的标准化数据，用于GAM拟合
        #     feature_x_data_orig,  #原始数据，用于绘图X轴
        #     shap_y_data,  #SHAP值
        #     feature_name,  # 特征的名称
        #     plot_label,  # 子图的标签（a）
        #     output_dir,  #保存地址
        #     colors  #配色
        # )

        # 提取当前特征的标准化数据(来自X_test)、原始数据(来自X_test_original)和对应的SHAP值
        feature_x_data_std = X_test[feature_name].values
        feature_x_data_orig = X_test_original[feature_name].values
        shap_y_data = shap_values[:, feature_idx]

        # 从scaler对象获取当前特征的均值和标准差
        feature_mean = scaler.mean_[feature_idx]
        feature_std = scaler.scale_[feature_idx]

        # 调用绘图函数
        png_path = plot_gam_style_from_shap(
            feature_x_data_std,  # 当前特征的标准化数据，用于GAM拟合
            feature_x_data_orig,  # 原始数据，用于绘图X轴
            shap_y_data,  # SHAP值
            feature_name,  # 特征的名称
            plot_label,  # 子图的标签（a）
            output_dir,  # 保存地址
            colors,  # 配色
            feature_mean,  # 均值
            feature_std,  # 标准差
        )
        generated_png_paths.append(png_path)  # 将生成的图片路径添加到列表中。

    # =========================================================================================
    # =====================================9.组合图设置=========================================
    # =========================================================================================
    if generated_png_paths:  # 检查是否成功生成了任何图片。
        print("\n--> 所有子图已生成，正在拼接组合图")
        composite_n_cols = 3  # 设置列数
        images_to_composite = [
            Image.open(path) for path in generated_png_paths
        ]  # 打开所有已保存的PNG图片文件。
        width, height = images_to_composite[0].size  # 获取单张图片的宽度和高度。
        num_images = len(images_to_composite)  # 获取图片的总数。

        composite_n_rows = (num_images + composite_n_cols - 1) // composite_n_cols  # 行数。
        composite_image = Image.new(
            "RGB", (width * composite_n_cols, height * composite_n_rows), color="white"
        )  # 创建一个新的空白RGB图像
        for i, img in enumerate(images_to_composite):  # 遍历每一张要拼接的图片。
            row, col = (
                i // composite_n_cols,
                i % composite_n_cols,
            )  # 计算当前图片在组合图中的行号和列号。
            paste_position = (
                col * width,
                row * height,
            )  # 计算当前图片应该被粘贴到的左上角坐标位置。
            composite_image.paste(img, paste_position)  # 将当前图片粘贴到画布的指定位置。
            img.close()
        # 保存
        composite_filename_base = "SHAP_Plots_Composite"  # 定义组合图的名称。
        png_path_composite = os.path.join(
            output_dir, f"{composite_filename_base}_{selected_color}.png"
        )
        pdf_path_composite = os.path.join(
            output_dir, f"{composite_filename_base}_{selected_color}.pdf"
        )
        composite_image.save(png_path_composite, "PNG", resolution=300.0)
        composite_image.save(pdf_path_composite, "PDF", resolution=300.0)
        print(f"组合图已成功保存到: {output_dir}")

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

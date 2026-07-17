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

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ======================================2.颜色库设置=========================================
# =========================================================================================
color_schemes = {
    1: {
        "scatter": "#0047AB",
        "fit": "darkorange",
        "fill": "darkorange",
        "threshold_line": "#E3242B",
        "threshold_text_bg": "#FFF8DC",
        "histogram": "black",
    },
    2: {
        "scatter": "#2E8B57",
        "fit": "#8A2BE2",
        "fill": "#8A2BE2",
        "threshold_line": "#FF4500",
        "threshold_text_bg": "#F0FFF0",
        "histogram": "#6B8E23",
    },
    3: {
        "scatter": "dimgray",
        "fit": "#D32F2F",
        "fill": "#D32F2F",
        "threshold_line": "navy",
        "threshold_text_bg": "#E6E6E6",
        "histogram": "#708090",
    },
    4: {
        "scatter": "#008080",
        "fit": "#FF7F50",
        "fill": "#FF7F50",
        "threshold_line": "#C71585",
        "threshold_text_bg": "#F0FFFF",
        "histogram": "#20B2AA",
    },
    5: {
        "scatter": "#228B22",
        "fit": "#FFBF00",
        "fill": "#FFBF00",
        "threshold_line": "#B22222",
        "threshold_text_bg": "#FAFAD2",
        "histogram": "#556B2F",
    },
    6: {
        "scatter": "#4B0082",
        "fit": "#FFD700",
        "fill": "#FFD700",
        "threshold_line": "#32CD32",
        "threshold_text_bg": "#F5F5DC",
        "histogram": "#800080",
    },
    7: {
        "scatter": "#2F4F4F",
        "fit": "#DC143C",
        "fill": "#DC143C",
        "threshold_line": "#00BFFF",
        "threshold_text_bg": "#F8F8FF",
        "histogram": "#778899",
    },
    8: {
        "scatter": "#082567",
        "fit": "#ADFF2F",
        "fill": "#ADFF2F",
        "threshold_line": "#FF69B4",
        "threshold_text_bg": "#F0FFF4",
        "histogram": "#191970",
    },
    9: {
        "scatter": "#483D8B",
        "fit": "#FFA500",
        "fill": "#FFA500",
        "threshold_line": "#FF1493",
        "threshold_text_bg": "#FFF0F5",
        "histogram": "#6A5ACD",
    },
    10: {
        "scatter": "#DE3163",
        "fit": "#00CED1",
        "fill": "#00CED1",
        "threshold_line": "#9400D3",
        "threshold_text_bg": "#E0FFFF",
        "histogram": "#C71585",
    },
    11: {
        "scatter": "#E6E6FA",
        "fit": "#5F9EA0",
        "fill": "#5F9EA0",
        "threshold_line": "#800080",
        "threshold_text_bg": "#F5FFFA",
        "histogram": "#B0C4DE",
    },
    12: {
        "scatter": "#964B00",
        "fit": "#87CEEB",
        "fill": "#87CEEB",
        "threshold_line": "#A52A2A",
        "threshold_text_bg": "#F0F8FF",
        "histogram": "#D2691E",
    },
    13: {
        "scatter": "#36454F",
        "fit": "#00FFFF",
        "fill": "#00FFFF",
        "threshold_line": "#FFD700",
        "threshold_text_bg": "#DCDCDC",
        "histogram": "#536872",
    },
    14: {
        "scatter": "#BC8F8F",
        "fit": "#808000",
        "fill": "#808000",
        "threshold_line": "#CD5C5C",
        "threshold_text_bg": "#FAF0E6",
        "histogram": "#CD853F",
    },
    15: {
        "scatter": "#D2B48C",
        "fit": "#1E90FF",
        "fill": "#1E90FF",
        "threshold_line": "#228B22",
        "threshold_text_bg": "#F5F5F5",
        "histogram": "#BDB76B",
    },
    16: {
        "scatter": "#696969",
        "fit": "#FF0000",
        "fill": "#FF0000",
        "threshold_line": "#000000",
        "threshold_text_bg": "#D3D3D3",
        "histogram": "#A9A9A9",
    },
    17: {
        "scatter": "#ADD8E6",
        "fit": "#00008B",
        "fill": "#00008B",
        "threshold_line": "#FF8C00",
        "threshold_text_bg": "#EAF4FF",
        "histogram": "#B0E0E6",
    },
    18: {
        "scatter": "#90EE90",
        "fit": "#006400",
        "fill": "#006400",
        "threshold_line": "#DC143C",
        "threshold_text_bg": "#F0FFF0",
        "histogram": "#3CB371",
    },
    19: {
        "scatter": "#F4A460",
        "fit": "#8B0000",
        "fill": "#8B0000",
        "threshold_line": "#2E8B57",
        "threshold_text_bg": "#FFF5EE",
        "histogram": "#D2691E",
    },
    20: {
        "scatter": "#FFB6C1",
        "fit": "#20B2AA",
        "fill": "#20B2AA",
        "threshold_line": "#DB7093",
        "threshold_text_bg": "#FFFFFF",
        "histogram": "#FFA07A",
    },
}


selected_scheme = 7  # 选择配色

colors = color_schemes.get(selected_scheme, color_schemes[1])


# =========================================================================================
# ======================================3.单特征依赖图绘图函数================================
# =========================================================================================
def plot_shap_dependence(
    feature_name,
    feature_values,
    shap_values_for_feature,
    plot_index,
    colors,
    save_dir="output_plots",
):
    print(f"正在处理：{feature_name}")

    fig = plt.figure(figsize=(10, 8))
    # 在图形中创建一个2行1列的网格布局
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4, 1],  # 行高比
        hspace=0,
    )
    ax1 = fig.add_subplot(gs[0, 0])  # 上行图
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # 下行图

    # 在上行图绘制shap散点图
    ax1.scatter(
        feature_values,
        shap_values_for_feature,
        alpha=0.1,
        label="Sample",
        s=30,
        color=colors["scatter"],
        zorder=2,
    )
    # 绘制出shap=0的横线
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=1.5, zorder=1)

    # 对特征和shap值进行多项式拟合，返回拟合系数
    coeffs = np.polyfit(feature_values, shap_values_for_feature, 5)

    poly_func = np.poly1d(coeffs)  # 使用拟合的系数创建一个多项式函数对象
    x_fit = np.linspace(
        feature_values.min(), feature_values.max(), 500
    )  # 创建模拟数据用于绘制平滑的拟合曲线
    y_fit = poly_func(x_fit)  # 计算拟合曲线上对应的y值

    # 计算R2
    y_poly_pred_on_actual_x = poly_func(feature_values)
    r_squared = r2_score(shap_values_for_feature, y_poly_pred_on_actual_x)

    residuals = shap_values_for_feature - y_poly_pred_on_actual_x  # 计算残差
    y_err = residuals.std() * 1.96  # 计算95%置信区间的误差范围
    y_upper = y_fit + y_err  # 计算置信区间的上界
    y_lower = y_fit - y_err  # 计算置信区间的下界
    # 格式化多项式方程字符串
    equation_str = (
        r"$y = "
        rf"{coeffs[0]:.2e}x^5"
        rf"{coeffs[1]:+.2e}x^4"
        rf"{coeffs[2]:+.2e}x^3"
        rf"{coeffs[3]:+.3f}x^2"
        rf"{coeffs[4]:+.3f}x"
        rf"{coeffs[5]:+.3f}"
        r"$"
    ).replace("e-0", "e-")
    # 绘制拟合曲线
    ax1.plot(
        x_fit,
        y_fit,
        color=colors["fit"],
        linestyle="--",
        lw=2.5,
        label=f"Fitted curve (R-square: {r_squared:.2f})\n{equation_str}",
        zorder=3,
    )
    # 填充拟合曲线的95%置信区间
    ax1.fill_between(
        x_fit,
        y_lower,
        y_upper,
        color=colors["fill"],
        alpha=0.2,
        label="95% Confidence interval",
        zorder=3,
    )

    # 计算并绘制阈值线
    roots = np.roots(coeffs)  # 计算拟合曲线与y=0的交点
    feature_min, feature_max = (
        feature_values.min(),
        feature_values.max(),
    )  # 获取特征值的最小值和最大值
    real_roots_in_range = [  # 创建一个列表，用于存储在特征值范围内的实数根
        root.real
        for root in roots  # 遍历所有的根
        if np.isreal(root)
        and feature_min <= root.real <= feature_max  # 判断条件：是实数根且在特征值范围内
    ]
    for threshold in real_roots_in_range:  # 遍历找到的阈值
        # 阈值线
        ax1.axvline(
            x=threshold, color=colors["threshold_line"], linestyle="--", linewidth=2.5, zorder=4
        )
        # 阈值
        ax1.text(
            x=threshold,
            y=ax1.get_ylim()[0] * 0.1,
            s=f" {threshold:.2f} ",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", fc=colors["threshold_text_bg"], ec="black", alpha=0.8
            ),
            zorder=5,
        )
        print(f"在特征 '{feature_name}'找到阈值: {threshold:.2f}")

    # 绘制下面的数据分布直方图
    ax2.hist(
        feature_values, bins=100, color=colors["histogram"], density=False, rwidth=0.8, zorder=5
    )

    # 上层图的y周的标注
    ax1.set_ylabel(f"Effect on Mig_R (%)", fontsize=22, weight="bold")
    # 上层图的背景网格线
    ax1.grid(True, which="both", linestyle=":", linewidth=1.5, color="lightgray", zorder=0)
    ax1.tick_params(axis="y", labelsize=18)  # 设置顶部子图Y轴刻度的字体大小
    plt.setp(ax1.get_xticklabels(), visible=False)  # 隐藏顶部子图的X轴刻度标签
    ax1.legend(loc="lower right", fontsize=12)  # 显示图例，并设置其位置和字体大小
    sub_caption = f"({chr(65 + plot_index)})"  # 创建子图左上角的小标题
    # 为上行图添加子图标题
    ax1.text(
        0.02,
        0.98,
        sub_caption,
        transform=ax1.transAxes,
        fontsize=24,
        weight="bold",
        va="top",
        ha="left",
    )
    ax2.set_xlabel(f"{feature_name}", fontsize=22, weight="bold")  # 下行图的x轴标注，就是特征名称
    ax2.set_ylabel("Dist.", fontsize=22, weight="bold")  # 下行图的y轴名称
    ax2.tick_params(axis="x", labelsize=18)  # 下行图X轴刻度的字体大小
    plt.setp(ax2.get_yticklabels(), visible=False)  # 隐藏下行图的Y轴刻度标签
    ax2.tick_params(axis="y", length=0)  # 隐藏下行图Y轴的刻度线
    # 为下行图添加网格线
    ax2.grid(True, which="both", linestyle=":", linewidth=1.5, color="lightgray", zorder=0)
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    file_path_png = os.path.join(
        save_dir, str(OUTPUT_DIR / f"shap_dependence_{feature_name}_{selected_scheme}.png")
    )
    file_path_pdf = os.path.join(
        save_dir, str(OUTPUT_DIR / f"shap_dependence_{feature_name}_{selected_scheme}.pdf")
    )
    plt.savefig(file_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(file_path_pdf, bbox_inches="tight")
    plt.close(fig)

    return file_path_png


# =========================================================================================
# ======================================4.拼接函数=========================================
# =========================================================================================
def stitch_images_grid(image_paths, n_cols, output_filename_base, save_dir):
    images = [
        Image.open(path) for path in image_paths
    ]  # 使用列表推导式，打开所有路径对应的图片，并将Image对象存入一个列表
    img_width, img_height = images[0].size  # 获取第一张图片的宽度和高度，假设所有子图尺寸相同
    n_images = len(images)  # 计算图片的总数量
    n_rows = (n_images + n_cols - 1) // n_cols  # 计算拼接后的网格需要多少行，使用整除法向上取整
    total_width = n_cols * img_width  # 计算最终拼接图的总宽度
    total_height = n_rows * img_height  # 计算最终拼接图的总高度

    composite_image = Image.new(
        "RGB", (total_width, total_height), color="white"
    )  # 创建一张新的RGB模式的空白图片（画布），背景色为白色，用于粘贴所有子图

    for i, img in enumerate(images):  # 遍历所有已打开的图片对象，同时获取索引(i)和图片(img)
        row = i // n_cols  # 计算当前图片应该被粘贴到哪一行
        col = i % n_cols  # 计算当前图片应该被粘贴到哪一列
        paste_x = col * img_width  # 计算粘贴位置的左上角x坐标
        paste_y = row * img_height  # 计算粘贴位置的左上角y坐标
        composite_image.paste(img, (paste_x, paste_y))  # 将当前图片粘贴到画布的指定位置
        img.close()  # 关闭已粘贴的图片对象，释放内存

    png_path_composite = os.path.join(save_dir, f"{output_filename_base}.png")  # 文件保存路径
    composite_image.save(png_path_composite)  # 将拼接好的大图保存为PNG格式
    print(f"\n组合图已保存为 '{png_path_composite}'")

    pdf_path_composite = os.path.join(
        save_dir, f"{output_filename_base}.pdf"
    )  # 构造PDF文件的保存路径
    if composite_image.mode == "RGBA":  # 检查图像模式，如果是有透明通道的'RGBA'
        composite_image = composite_image.convert(
            "RGB"
        )  # 则将其转换为'RGB'模式，因为PDF不支持透明度
    composite_image.save(pdf_path_composite)  # 将拼接好的大图保存为PDF格式
    print(f"组合图已保存为 '{pdf_path_composite}'")


# =========================================================================================
# ======================================5.数据的加载与处理=========================================
# =========================================================================================
file_path = str(DATA_DIR / "data.xlsx")  # 指定Excel文件的路径
target_column_name = "FVC"  # 目标变量

data_df = pd.read_excel(file_path)  # 读取数据
print(f"成功从 '{file_path}' 加载数据。")


feature_names = [col for col in data_df.columns if col != target_column_name]  # 获取所有特征的列名
X = data_df[feature_names]  # 提取特征数据并转换为numpy数组
y = data_df[target_column_name]  # 提取目标变量数据并转换为numpy数组

print(f"特征列表: {feature_names}")
# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 将标准化的NumPy数组转换回保留列名和索引的DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# =========================================================================================
# ======================================6.模型构建=========================================
# =========================================================================================
print("\n--- 正在使用GridSearchCV训练LGBM模型 ---")
lgb_model = lgb.LGBMRegressor(random_state=42)  # 创建一个LGBM回归器实例
# 超参数网格
param_grid = {
    "n_estimators": [100, 200, 300],
    # 'max_depth': [3, 5, 7],
    # 'learning_rate': [0.01, 0.05, 0.1],
    # 'num_leaves': [20, 31, 40]
}
# 创建GridSearchCV对象，设置模型、参数网格、交叉验证折数、并行任务数、评分标准和详细程度
grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="neg_mean_squared_error",
    verbose=1,
)
# 在标准化的训练集上执行网格搜索
grid_search.fit(X_train_scaled, y_train)
print(f"最佳超参数: {grid_search.best_params_}")
best_model = grid_search.best_estimator_  # 最佳模型
# =========================================================================================
# ======================================7.模型性能评估=========================================
# =========================================================================================
print("\n模型性能")


y_train_pred = best_model.predict(X_train_scaled)  # 使用最佳模型对训练集进行预测
r2_train = r2_score(y_train, y_train_pred)  # 训练集R2
mse_train = mean_squared_error(y_train, y_train_pred)  # 训练集均方误差
print(f"训练集：R2: {r2_train:.4f} | MSE: {mse_train:.4f}")


y_test_pred = best_model.predict(X_test_scaled)  # 使用最佳模型对测试集进行预测
r2_test = r2_score(y_test, y_test_pred)  # 测试集上的R2
mse_test = mean_squared_error(y_test, y_test_pred)  # 测试集均方误差
print(f"测试集：R2: {r2_test:.4f} | MSE: {mse_test:.4f}")


output_dir = str(OUTPUT_DIR)  # 模型保存的路径

model_path = os.path.join(output_dir, "best_lgbm_model.joblib")  # 模型的完整保存路径
joblib.dump(best_model, model_path)  # 保存

# =========================================================================================
# ======================================8.shap分析=========================================
# =========================================================================================
print("\n计算SHAP值")
explainer = shap.TreeExplainer(best_model)  # 创建一个适用于树模型的SHAP解释器
shap_values_matrix = explainer.shap_values(X_test_scaled)  # 计算测试集中每个样本每个特征的SHAP值

# 根据SHAP值的平均绝对值计算特征重要性
feature_importance = np.abs(shap_values_matrix).mean(axis=0)
# 获取按重要性降序排列的特征索引
sorted_feature_indices = np.argsort(feature_importance)[::-1]

# 创建一个从原始特征名到其在原始列表中的索引的映射
feature_name_to_original_index = {name: i for i, name in enumerate(feature_names)}

# 按照重要性排序后的特征名列表
sorted_feature_names = [feature_names[i] for i in sorted_feature_indices]
print(f"特征按重要性排序: {sorted_feature_names}")

# =========================================================================================
# ======================================9.绘图，包括子图和组合图=========================================
# =========================================================================================
saved_plot_paths = []  # 初始化一个空列表，用来收集每个子图的文件路径

# 按照特征重要性顺序进行循环绘图
for plot_idx, sorted_name in enumerate(sorted_feature_names):
    # 找到该特征在原始数据（X_test, shap_values_matrix）中的列索引
    original_idx = feature_name_to_original_index[sorted_name]

    plot_path = plot_shap_dependence(  # 调用之前定义的绘图函数
        feature_name=sorted_name,  # 传入当前特征的名称
        feature_values=X_test.iloc[:, original_idx],  # 传入当前特征在测试集上的原始值
        shap_values_for_feature=shap_values_matrix[:, original_idx],  # 传入当前特征对应的SHAP值
        plot_index=plot_idx,  # 绘图的顺序索引，用于生成子图标题 (A), (B)
        colors=colors,  # 颜色方案
        save_dir="output_shap_plots",  # 保存地址
    )
    saved_plot_paths.append(plot_path)  # 将返回的图片文件路径添加到列表中

n_cols_for_stitching = 5  # 列数
output_filename_base = f"composite_{selected_scheme}"  # 组合图的文件名
output_save_dir = str(OUTPUT_DIR)  # 组合图的保存地址
# 调用拼接函数
stitch_images_grid(
    image_paths=saved_plot_paths,  # 所有子图路径的列表
    n_cols=n_cols_for_stitching,  # 列数
    output_filename_base=output_filename_base,  # 组合图文件名
    save_dir=output_save_dir,  # 组合图的保存路径
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

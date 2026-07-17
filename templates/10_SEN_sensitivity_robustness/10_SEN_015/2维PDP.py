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
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from sklearn.inspection import partial_dependence
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ======================================2. 从本地Excel文件加载数据 =========================================
# =========================================================================================
input_excel_path = str(DATA_DIR / "simulated_environmental_data.xlsx")  # 定义文件地址
target_column_name = "Target"  # 指定目标变量
data_df = pd.read_excel(input_excel_path)  # 读取数据
# 提取目标变量和特征
X = data_df.drop(target_column_name, axis=1)
y = data_df[target_column_name]
# 划分训练数据和验证数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# =========================================================================================
# ======================================3. 标准化处理及模型训练=========================================
# =========================================================================================
# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 定义超参数范围
param_grid = {
    "n_estimators": [100, 150],  # 树的数量
    "max_depth": [4, 5, 6],  # 树的最大深度
    "learning_rate": [0.05, 0.1],  # 学习率
}

# 初始化GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective="reg:squarederror", random_state=42),  # 指定基础模型
    param_grid=param_grid,  # 传入要搜索的参数网格
    scoring="neg_mean_squared_error",  # 指定评估指标
    cv=5,  # 指定交叉验证
    n_jobs=-1,  # 使用的CPU核心
    verbose=1,  # 是否打印搜索过程
)
# 执行网格搜索寻找最佳超参数组合
grid_search.fit(X_train_scaled, y_train)
best_xgb_model = grid_search.best_estimator_  # 从网格搜索结果中获取性能最好的模型
print(f"\n网格搜索完成！最佳超参数: {grid_search.best_params_}")
# =========================================================================================
# ======================================4.设置颜色库=========================================
# =========================================================================================
color_schemes = {
    1: [
        "#534684",
        "#3B6993",
        "#3AA593",
        "#71C07A",
        "#B0D45B",
        "#FDE85B",
        "#e4cf52",
        "#cca14a",
        "#b37641",
        "#994f39",
    ],
    2: [
        "#0d3b66",
        "#faf0ca",
        "#f4d35e",
        "#ee964b",
        "#f95738",
        "#e04e32",
        "#c8452d",
        "#af3c27",
        "#963421",
        "#7d2b1c",
    ],
    3: [
        "#00b4d8",
        "#ade8f4",
        "#caf0f8",
        "#90e0ef",
        "#48cae4",
        "#41b6cd",
        "#3aa2b6",
        "#328e9f",
        "#2b7a88",
        "#246671",
    ],
    4: [
        "#d00000",
        "#ffba08",
        "#3f88c5",
        "#032b43",
        "#136f63",
        "#116359",
        "#0f584f",
        "#0d4c45",
        "#0c413b",
        "#0a3631",
    ],
    5: [
        "#2d00f7",
        "#6a00f4",
        "#8900f2",
        "#a100f2",
        "#b100e8",
        "#bc00dd",
        "#a900c7",
        "#9600b1",
        "#83009b",
        "#700085",
    ],
    6: [
        "#ffc300",
        "#ffd60a",
        "#ffc857",
        "#ffb703",
        "#fb8500",
        "#e27800",
        "#c96b00",
        "#b05e00",
        "#975100",
        "#7e4400",
    ],
    7: [
        "#006400",
        "#228B22",
        "#3CB371",
        "#66CDAA",
        "#98FB98",
        "#89e289",
        "#7ac97a",
        "#6baf6b",
        "#5c965c",
        "#4d7d4d",
    ],
    8: [
        "#800f2f",
        "#a4133c",
        "#c9184a",
        "#ff4d6d",
        "#ff758f",
        "#ff8fa3",
        "#e68093",
        "#cc7282",
        "#b36372",
        "#995562",
    ],
    9: [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#7d4d43",
        "#6e443b",
        "#5f3b33",
        "#50322b",
    ],
    10: [
        "#4a4e69",
        "#9a8c98",
        "#c9ada7",
        "#f2e9e4",
        "#22223b",
        "#1f1f35",
        "#1c1c2e",
        "#191928",
        "#161622",
        "#13131c",
    ],
    11: [
        "#4cc9f0",
        "#4895ef",
        "#4361ee",
        "#3f37c9",
        "#3a0ca3",
        "#340b93",
        "#2e0a82",
        "#280972",
        "#220862",
        "#1c0652",
    ],
    12: [
        "#f94144",
        "#f3722c",
        "#f8961e",
        "#f9c74f",
        "#90be6d",
        "#43aa8b",
        "#3b997c",
        "#34886e",
        "#2d7760",
        "#266652",
    ],
    13: [
        "#e07a5f",
        "#3d405b",
        "#81b29a",
        "#f2cc8f",
        "#f4f1de",
        "#dadce7",
        "#c1c6d0",
        "#a7b1b9",
        "#8e9ca2",
        "#75878b",
    ],
    14: [
        "#606c38",
        "#283618",
        "#fefae0",
        "#dda15e",
        "#bc6c25",
        "#a96121",
        "#96561e",
        "#834b1a",
        "#704017",
        "#5d3513",
    ],
    15: [
        "#007f5f",
        "#2b9348",
        "#55a630",
        "#80b918",
        "#aacc00",
        "#bfd200",
        "#abbd00",
        "#96a800",
        "#829300",
        "#6d7e00",
    ],
    16: [
        "#390099",
        "#9e0059",
        "#ff0054",
        "#ff5400",
        "#ffbd00",
        "#e6a900",
        "#cc9600",
        "#b38300",
        "#997000",
        "#805d00",
    ],
    17: [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#d06449",
        "#b95941",
        "#a24e3a",
        "#8b4332",
        "#74382a",
    ],
    18: [
        "#0c0a3e",
        "#7b1e7a",
        "#c75180",
        "#ef817a",
        "#f7f4f9",
        "#dedcf0",
        "#c5c3e7",
        "#acabde",
        "#9392d5",
        "#7a7acd",
    ],
    19: [
        "#003049",
        "#d62828",
        "#f77f00",
        "#fcbf49",
        "#eae2b7",
        "#d3cbc0",
        "#bbb5a9",
        "#a29f92",
        "#8a887b",
        "#727264",
    ],
    20: [
        "#7400b8",
        "#6930c3",
        "#5e60ce",
        "#5390d9",
        "#4ea8de",
        "#48bfe3",
        "#41accf",
        "#3a99bb",
        "#3286a7",
        "#2b7393",
    ],
}
# 选择配色方案
color_choice = 20


# =========================================================================================
# ======================================5.绘制二维偏依赖图=========================================
# =========================================================================================
def plot_pdp_interaction(
    xgb_model, scaler, X_data_unscaled, feature_pair, plot_label, output_dir, colors_list
):
    """
    xgb_model: 训练好的模型
    scaler:StandardScaler对象
    X_data_unscaled: 用于计算偏依赖的未经标准化的原始训练集。
    feature_pair: 一个包含两个特征名称的元组或列表。
    plot_label: 要添加到图左上角的标签，例如 '(a)'。
    output_dir: 保存图像的文件夹路径。
    colors_list: 用于绘图的颜色列表。
    """
    print(f"正在绘制'{feature_pair[0]}' 与 '{feature_pair[1]}' 的交互图")
    X_data_scaled_np = scaler.transform(X_data_unscaled)  # 使用传入的scaler对未标准化的数据进行转换
    X_data_scaled = pd.DataFrame(
        X_data_scaled_np, columns=X_data_unscaled.columns
    )  # 将转换后的numpy数组变回DataFrame，并保留列名
    # 调用partial_dependence函数计算偏依赖值
    pdp_result = partial_dependence(
        xgb_model,  # 传入训练好的模型
        X_data_scaled,  # 传入标准化后的特征数据
        features=feature_pair,  # 指定要分析的特征对
        kind="average",  # 指定计算类型为平均效应
        grid_resolution=50,  # 设置每个特征的网格点数量
    )
    Z = pdp_result["average"][0]  # 提取计算出的PDP值
    grid_values_scaled = pdp_result["grid_values"]  # 提取用于计算的网格点值（标准化后的）
    feature_indices = [
        X_data_unscaled.columns.get_loc(f) for f in feature_pair
    ]  # 获取两个特征在原始数据中的列索引
    dummy_grid = np.zeros(
        (len(grid_values_scaled[0]), len(X_data_unscaled.columns))
    )  # 创建一个与原始数据形状相似的零矩阵，用于逆转换第一个特征
    dummy_grid[:, feature_indices[0]] = grid_values_scaled[0]  # 将第一个特征的网格值填入
    dummy_grid[:, feature_indices[1]] = scaler.mean_[
        feature_indices[1]
    ]  # 将第二个特征的值用其均值填充
    grid_values_unscaled_feat1 = scaler.inverse_transform(dummy_grid)[
        :, feature_indices[0]
    ]  # 对整个矩阵进行逆标准化，并提取出第一个特征的原始刻度值

    dummy_grid = np.zeros(
        (len(grid_values_scaled[1]), len(X_data_unscaled.columns))
    )  # 同样地，为第二个特征创建零矩阵
    dummy_grid[:, feature_indices[1]] = grid_values_scaled[1]  # 将第二个特征的网格值填入
    dummy_grid[:, feature_indices[0]] = scaler.mean_[
        feature_indices[0]
    ]  # 将第一个特征的值用其均值填充
    grid_values_unscaled_feat2 = scaler.inverse_transform(dummy_grid)[
        :, feature_indices[1]
    ]  # 逆标准化并提取第二个特征的原始刻度值

    XX, YY = np.meshgrid(
        grid_values_unscaled_feat1, grid_values_unscaled_feat2
    )  # 使用逆标准化后的网格值创建二维网格坐标
    Z_grid = Z.reshape(
        len(grid_values_unscaled_feat1), len(grid_values_unscaled_feat2)
    ).T  # 将一维的PDP值Z重塑为与网格匹配的二维矩阵，并转置以匹配坐标轴

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", colors_list
    )  # 使用指定的颜色列表创建一个自定义的颜色映射
    min_val, max_val = np.min(Z_grid), np.max(Z_grid)  # 找到PDP值的最小值和最大值
    levels = np.linspace(min_val, max_val, 8)  # 在最小值和最大值之间创建等间距的等高线层级

    contour_fill = ax.contourf(
        XX, YY, Z_grid, levels=levels, cmap=custom_cmap, extend="neither"
    )  # 绘制填充的等高线图
    contour_lines = ax.contour(
        XX, YY, Z_grid, levels=levels, colors="white", linewidths=1.2
    )  # 在填充图上绘制白色的等高线
    ax.clabel(
        contour_lines, inline=True, fontsize=16, fmt="%.2f", colors="black"
    )  # 为等高线添加数值标签

    ax.set_title(
        f"Partial Dependence of {feature_pair[0]} and {feature_pair[1]}", fontsize=16, weight="bold"
    )  # 设置图表标题
    ax.set_xlabel(feature_pair[0], fontsize=16)  # 设置X轴标题
    ax.set_ylabel(feature_pair[1], fontsize=16)  # 设置Y轴标题

    # 创建一个与主图关联的布局工具
    divider = make_axes_locatable(ax)
    # 在主图的右侧创建一个新的坐标轴(cax)专门给颜色条使用
    cax = divider.append_axes(
        "right",
        size="2.5%",  # 控制了颜色条的宽度
        pad=0.1,
    )  # 控制了颜色条与主图之间的间距
    # 绘制颜色条
    cbar = fig.colorbar(contour_fill, cax=cax)
    # 控制数值标注的大小和粗细
    cbar.ax.tick_params(axis="y", labelsize=16)
    # for label in cbar.ax.get_yticklabels():
    #     label.set_fontweight('bold')
    # 控制刻度线的长短和粗细
    cbar.ax.tick_params(axis="y", which="major", length=3, width=1)
    # 控制颜色条边框的粗细
    cbar.outline.set_linewidth(1)

    # 在图的左上角添加标注
    ax.text(
        0.05,
        0.95,
        plot_label,
        transform=ax.transAxes,
        fontsize=16,  # 在坐标轴的相对位置处添加文本
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )  # 设置文本样式：粗体、垂直顶部对齐、水平左对齐、黑色
    if X_data_unscaled[feature_pair[0]].max() > 1000:  # 如果X轴特征的最大值大于1000
        ax.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )  # 将X轴刻度格式化为带千位分隔符的整数
    if X_data_unscaled[feature_pair[1]].max() > 1000:  # 如果Y轴特征的最大值大于1000
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ","))
        )  # 将Y轴刻度格式化为带千位分隔符的整数
    # 控制坐标轴刻度数值的大小和粗细
    ax.tick_params(axis="both", which="major", labelsize=16)
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     # label.set_fontsize(12)
    #     label.set_fontweight('bold')
    # 控制坐标轴刻度线的长短和粗细
    ax.tick_params(axis="both", which="major", length=3, width=1)
    # 遍历图框
    for spine in ax.spines.values():
        spine.set_linewidth(1)  # 设置边框的线宽
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    # 保存
    filename_prefix = f"{feature_pair[0]}_{feature_pair[1]}"
    path_png = os.path.join(output_dir, f"{filename_prefix}_{color_choice}.png")
    path_pdf = os.path.join(output_dir, f"{filename_prefix}_{color_choice}.pdf")
    plt.savefig(path_png)
    plt.savefig(path_pdf)
    plt.close(fig)
    return path_png


# =========================================================================================
# ======================================6. 图像拼接函数=========================================
# =========================================================================================
def stitch_images_grid(image_paths, n_cols, output_filename_base, save_dir):
    images = [
        Image.open(path) for path in image_paths
    ]  # 使用列表推导式，打开所有路径对应的图片，并将Image对象存入一个列表。
    img_width, img_height = images[0].size  # 获取第一张图片的宽度和高度，假设所有子图尺寸相同。
    n_images = len(images)  # 计算图片的总数量。
    n_rows = (n_images + n_cols - 1) // n_cols  # 计算拼接后的网格需要多少行，使用整除法向上取整。
    total_width = n_cols * img_width  # 计算最终拼接图的总宽度。
    total_height = n_rows * img_height  # 计算最终拼接图的总高度。

    composite_image = Image.new(
        "RGB", (total_width, total_height), color="white"
    )  # 创建一张新的RGB模式的空白图片（画布），背景色为白色，用于粘贴所有子图。
    for i, img in enumerate(images):  # 遍历所有已打开的图片对象，同时获取索引(i)和图片(img)。
        row = i // n_cols  # 计算当前图片应该被粘贴到哪一行。
        col = i % n_cols  # 计算当前图片应该被粘贴到哪一列。
        paste_x = col * img_width  # 计算粘贴位置的左上角x坐标。
        paste_y = row * img_height  # 计算粘贴位置的左上角y坐标。
        composite_image.paste(img, (paste_x, paste_y))  # 将当前图片粘贴到画布的指定位置。
        img.close()
    # 保存
    png_path_composite = os.path.join(save_dir, f"{output_filename_base}.png")
    composite_image.save(png_path_composite)
    print(f"\n组合图已保存为 '{png_path_composite}'")
    pdf_path_composite = os.path.join(save_dir, f"{output_filename_base}.pdf")
    if composite_image.mode == "RGBA":
        composite_image = composite_image.convert("RGB")
    composite_image.save(pdf_path_composite)
    print(f"组合图已保存为 '{pdf_path_composite}'")


# =========================================================================================
# ======================================7. 特征重要性分析、循环绘图与最终拼接========================================
# =========================================================================================
print("\n正在计算特征重要性并绘制交互作用图")
output_plot_dir = str(OUTPUT_DIR)  # 定义图表输出的文件夹路径
os.makedirs(output_plot_dir, exist_ok=True)  # 创建输出文件夹，如果已存在则不报错

final_xgb_model = best_xgb_model  # 最佳模型
# importances = final_xgb_model.feature_importances_  # 从最终模型中提取特征重要性得分
# feature_importance_df = pd.DataFrame({  # 创建一个DataFrame来存储特征名称和其重要性
#     'Feature': X.columns,  # 特征名称
#     'Importance': importances  # 对应的特征重要性
# }).sort_values(by='Importance', ascending=False)  # 按重要性降序排列
print("  正在使用SHAP计算特征重要性")
# 创建一个SHAP解释器
explainer = shap.TreeExplainer(final_xgb_model)
# 计算SHAP值
shap_values = explainer(X_train_scaled)
# 计算全局特征重要性，通过计算所有样本SHAP值的绝对值的平均值来得到的
shap_importance = np.mean(np.abs(shap_values.values), axis=0)
# 创建特征重要性的DataFrame并排序
feature_importance_df = pd.DataFrame(
    {
        "Feature": X.columns,  # 特征名称
        "Importance": shap_importance,  # 对应的SHAP重要性
    }
).sort_values(by="Importance", ascending=False)  # 按重要性降序排列
print("  SHAP特征重要性计算完成。")

# 定义要分析的最重要特征的数量
N_TOP_FEATURES = 4
top_features = feature_importance_df.head(N_TOP_FEATURES)[
    "Feature"
].tolist()  # 选取前N个最重要的特征，并将其名称存入列表
print(f"已选取最重要的 {N_TOP_FEATURES} 个特征进行分析: {top_features}")

feature_pairs = list(
    itertools.combinations(top_features, 2)
)  # 从最重要的特征列表中生成所有可能的两两组合
print(f"将为以下 {len(feature_pairs)} 对特征组合绘制PDP图：{feature_pairs}")

selected_colors = color_schemes.get(
    color_choice, color_schemes[1]
)  # 根据之前选择的配色方案从字典中获取颜色列表

generated_image_paths = []  # 初始化一个空列表，用于存储所有生成图像的路径

# 使用 enumerate 来获取循环索引，以便生成标注
for i, pair in enumerate(feature_pairs):  # 遍历所有特征对，并同时获取索引i
    # 根据索引i生成标注
    plot_label = f"({chr(ord('a') + i)})"
    # 调用绘图函数
    saved_path = plot_pdp_interaction(
        best_xgb_model, scaler, X_train, pair, plot_label, output_plot_dir, selected_colors
    )
    generated_image_paths.append(saved_path)  # 将保存的图片路径添加到列表中，后续拼接使用
# 调用图像拼接函数
stitch_images_grid(
    image_paths=generated_image_paths,  # 传入所有单个图像的路径列表
    n_cols=3,  # 列数
    output_filename_base="PDP_Composite_Figure",  # 指定拼接后图像的文件名
    save_dir=output_plot_dir,  # 指定拼接后图像的保存目录
)
print("所有任务已全部完成！")

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

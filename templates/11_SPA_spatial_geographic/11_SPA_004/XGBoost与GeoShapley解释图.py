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

# 导入所有需要的库
import matplotlib  # 导入matplotlib库，用于进行更底层的图表设置
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于创建图表，通常简写为plt
import numpy as np  # 导入numpy库，用于科学计算，特别是多维数组操作，通常简写为np
import pandas as pd  # 导入pandas库，用于数据处理和分析，通常简写为pd
import shap  # 导入shap库，用于模型解释，计算SHAP值
import xgboost as xgb  # 导入xgboost库，用于实现梯度提升决策树模型，通常简写为xgb
from geoshapley import GeoShapleyExplainer  # 从geoshapley库中导入地理空间SHAP解释器
from sklearn.model_selection import train_test_split  # 从scikit-learn库中导入数据划分工具

# 注意：此行设置matplotlib的后端为'TkAgg'，用于在某些环境下解决绘图窗口的显示问题
matplotlib.use("Agg")  # 设置matplotlib使用'TkAgg'后端，以确保图形窗口能正常弹出
# =============================================================================
# 1. 全局样式设置
# =============================================================================
# 设置全局字体系列为无衬线字体
plt.rcParams["font.family"] = "sans-serif"  # 设置图表中的默认字体族为无衬线字体
# 在无衬线字体族中，优先使用'Times New Roman'，如果找不到，则使用'SimHei'（黑体）作为中文字体
plt.rcParams["font.sans-serif"] = [
    "Times New Roman",
    "SimHei",
]  # 指定具体的无衬线字体，优先使用'Times New Roman'，并包含中文黑体'SimHei'
# 设置在图表中正常显示负号（避免显示为方框）
plt.rcParams["axes.unicode_minus"] = False  # 解决在使用中文等非Unicode字体时，负号显示为方框的问题
# =============================================================================
# 2. 从Excel文件加载您的实际数据
# =============================================================================
print("--- 步骤 1: 正在从 Excel 文件加载数据 ---")  # 打印提示信息，告知用户当前正在执行的步骤
# ---! 请将这里的路径修改为您自己电脑上的实际数据文件路径 !---
excel_path = str(
    DATA_DIR / "model_simulation_results.xlsx"
)  # 定义Excel数据文件的路径，'r'表示原始字符串，避免转义符问题
try:  # 使用try-except代码块来捕获可能发生的错误
    # 从 'Simulated_Data' 工作表中读取数据
    df_full_dataset = pd.read_excel(
        excel_path, sheet_name="Simulated_Data"
    )  # 使用pandas读取指定Excel文件中的名为'Simulated_Data'的工作表
except FileNotFoundError:  # 如果try代码块中发生“文件未找到”的错误
    print(f"错误：文件未找到 '{excel_path}'。请检查路径是否正确。")  # 打印错误提示信息
    exit()  # 退出程序
# 从加载的数据中分离特征和目标变量
# 假设目标变量列的名称是 'target'
if "target" in df_full_dataset.columns:  # 检查数据框的列名中是否包含'target'
    df_features = df_full_dataset.drop(
        "target", axis=1
    )  # 如果包含，则创建一个名为df_features的新数据框，它删除了'target'列
    y = df_full_dataset["target"]  # 将'target'列单独提取出来，作为目标变量y
else:  # 如果不包含'target'列
    print("错误：在文件中未找到名为 'target' 的列。")  # 打印错误信息
    exit()  # 退出程序
# 动态获取特征名称列表
feature_names = df_features.columns.tolist()  # 获取特征数据框的所有列名，并将其转换为列表
print(f"数据加载成功，包含 {len(feature_names)} 个特征。")  # 打印成功信息，并显示加载的特征数量
# =============================================================================
# 3. 训练 XGBoost 模型
# =============================================================================
# 打印当前步骤的提示信息
print("\n--- 步骤 2: 正在训练 XGBoost 模型 ---")  # 打印提示信息，告知用户当前正在训练模型
# 使用 train_test_split 函数将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df_features,
    y,  # 要划分的特征(X)和目标(y)数据
    test_size=0.2,  # 指定测试集占总数据集大小的20%
    random_state=42,  # 设置随机种子为42，确保每次划分的结果都一样，便于复现
)
# 初始化一个 XGBoost 回归器（Regressor）模型对象
model = xgb.XGBRegressor(
    n_estimators=100,  # 设置模型中决策树的数量为100个
    max_depth=5,  # 设置每棵决策树的最大深度为5
    learning_rate=0.1,  # 设置学习率（步长）为0.1
    objective="reg:squarederror",  # 指定目标函数为回归任务的平方误差
    random_state=42,  # 设置随机种子为42，确保模型训练过程可复现
)
# 在训练数据上拟合（训练）模型
model.fit(X_train, y_train)  # 使用训练集的特征(X_train)和目标(y_train)来训练模型
print("模型训练完成。")  # 打印模型训练完成的提示信息
# =============================================================================
# 4. 提取并绘制特征重要性图
# =============================================================================
# 打印当前步骤的提示信息
print("\n--- 步骤 3: 正在提取特征重要性并绘图 ---")  # 打印提示信息，告知用户当前步骤
# 从训练好的模型中提取特征重要性分数
importances = model.feature_importances_  # 获取模型计算出的各个特征的重要性分数
# 创建一个新的DataFrame来存储特征名称和其对应的重要性分数
df_importance = pd.DataFrame(
    {
        "feature": feature_names,  # 第一列是特征名称
        "importance": importances,  # 第二列是对应的重要性分数
    }
)
# 为了后续绘图方便，按重要性升序排列DataFrame（因为barh绘图会自动反转顺序）
df_importance = df_importance.sort_values(
    by="importance", ascending=True
)  # 按照'importance'列的值进行升序排序
# 创建一个图窗(fig)和一个坐标轴(ax)，并设置画布大小
fig, ax = plt.subplots(figsize=(12, 10))  # 创建一个12x10英寸的画布
# ---- 4.1 绘制水平条形图 ----
bars = ax.barh(
    df_importance["feature"], df_importance["importance"], color="#d62828", label="Importance"
)  # 绘制水平条形图，y轴是特征名，x轴是重要性值
ax.set_title(
    r"Feature importance values calculated using the XGBoost model", fontsize=18, pad=20
)  # 设置图表标题
ax.set_ylabel("Variable", fontsize=16)  # 设置y轴标签
ax.tick_params(axis="both", which="major", labelsize=12)  # 设置坐标轴刻度的字体大小
for bar in bars:  # 遍历每一个条形
    width = bar.get_width()  # 获取条形的宽度（即特征重要性值）
    ax.text(
        width,
        bar.get_y() + bar.get_height() / 2,  # 在条形右侧添加文本
        f" {width:.3f}",
        va="center",
        ha="left",
        fontsize=10,
    )  # 文本内容为保留三位小数的重要性值
ax.set_xlim(right=ax.get_xlim()[1] * 1.15)  # 将x轴的右边界扩大15%，以防标签超出边界
# ---- 4.2 绘制环圈图 ----
donut_features = ["LON", "UR", "TC", "RO", "ME"]  # 定义一个列表，包含我们想在环圈图中显示的特征
df_donut = df_importance[
    df_importance["feature"].isin(donut_features)
].copy()  # 从完整的重要性数据中筛选出这些特征
df_donut["feature"] = pd.Categorical(
    df_donut["feature"], categories=donut_features, ordered=True
)  # 将特征列转换为有序的类别类型，以保证绘图顺序
df_donut = df_donut.sort_values("feature")  # 按照定义的类别顺序进行排序
if (
    not df_donut.empty and df_donut["importance"].sum() > 0
):  # 检查筛选后的数据不为空且重要性总和大于0
    total_donut_importance = df_donut["importance"].sum()  # 计算这几个特征的重要性总和
    donut_percentages_raw = (
        df_donut["importance"] / total_donut_importance * 100
    )  # 计算每个特征在子集中的重要性百分比
    ax_inset = fig.add_axes(
        [0.4, 0.15, 0.3, 0.3]
    )  # 在主图的指定位置([左, 下, 宽, 高])创建一个新的内嵌坐标轴
    colors = matplotlib.colormaps.get("tab10").colors  # 获取一个名为'tab10'的颜色映射表
    wedges, texts = ax_inset.pie(  # 绘制饼图（通过设置wedgeprops实现环圈效果）
        donut_percentages_raw,  # 饼图的数值（百分比）
        colors=colors[: len(df_donut)],
        startangle=90,
        counterclock=False,  # 设置颜色、起始角度、绘制方向
        wedgeprops=dict(width=0.45, edgecolor="w"),  # 设置环圈的宽度和边缘颜色，从而制作出环圈图
    )
    subset_importance_ratio = (
        df_donut["importance"].sum() / df_importance["importance"].sum()
    )  # 计算子集特征重要性占总重要性的比例
    ax_inset.text(
        0,
        0,
        f"Total importance\nof subset\n{subset_importance_ratio:.2%}",  # 在环圈图中央添加文本
        ha="center",
        va="center",
        fontsize=8,
        linespacing=1.5,
    )  # 设置文本内容、对齐方式、字体大小等
    label_threshold = 2.0  # 设置一个阈值，用于决定标签的显示方式
    y_text_offsets = {"left": 1.4, "right": 1.4}  # 初始化标签在y轴方向的偏移量，用于避免重叠
    for i, p in enumerate(wedges):  # 遍历环圈的每一个扇区
        percent = donut_percentages_raw.iloc[i]  # 获取当前扇区的百分比
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1  # 计算扇区中间的角度
        y = np.sin(np.deg2rad(ang))  # 根据角度计算标签的y坐标
        x = np.cos(np.deg2rad(ang))  # 根据角度计算标签的x坐标
        if percent < label_threshold and percent > 0:  # 如果百分比小于阈值（太小了）
            side = "right" if x > 0 else "left"  # 判断标签在图的左侧还是右侧
            y_pos = y_text_offsets[side]  # 获取当前侧的y轴偏移量
            y_text_offsets[side] += -0.2 if y > 0 else 0.2  # 更新偏移量，避免下一个标签重叠
            connectionstyle = f"angle,angleA=0,angleB={ang}"  # 定义连接线的样式
            ax_inset.annotate(
                f"{percent:.1f}%",
                xy=(x, y),
                xytext=(0.1 * np.sign(x), y_pos),  # 使用annotate添加带连接线的标签
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle),
            )  # 设置箭头和连接线属性
        elif percent > 0:  # 如果百分比较大
            ax_inset.text(
                x * 1.2,
                y * 1.2,
                f"{percent:.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )  # 直接在扇区外侧显示百分比
    ax_inset.legend(
        wedges,
        df_donut["feature"],  # 为环圈图添加图例
        loc="center left",
        bbox_to_anchor=(1, 0.8),  # 设置图例位置在图的右侧
        frameon=False,
        fontsize=12,
    )  # 不显示图例边框，并设置字体大小
output_path = str(OUTPUT_DIR / "feature.jpg")  # 定义输出图片的路径和文件名
plt.savefig(
    output_path, dpi=300, bbox_inches="tight"
)  # 将图表保存为jpg文件，分辨率为300dpi，并自动裁剪边缘空白
plt.close("all")  # Interactive display removed; assets were exported above.
# =============================================================================
# 5. SHAP 分析与绘图
# =============================================================================
print("\n--- 步骤 4: 正在进行 SHAP 分析并绘制图表 ---")  # 打印提示信息
explainer = shap.TreeExplainer(model)  # 创建一个适用于树模型的SHAP解释器
shap_values = explainer(df_features)  # 计算所有数据点中每个特征的SHAP值
plt.figure(figsize=(10, 8))  # 创建一个新的10x8英寸的画布
shap.summary_plot(  # 绘制SHAP摘要图
    shap_values,
    df_features,
    plot_type="dot",
    cmap="RdYlBu",  # 传入SHAP值和特征数据，指定图类型为"dot"，颜色映射为"RdYlBu"
    show=False,
    plot_size=None,  # 不立即显示图形，使用matplotlib的设置来控制大小
)
ax2 = plt.gca()  # 获取当前的坐标轴对象
ax2.set_title("SHAP Feature Importance Summary", fontsize=16)  # 设置图表标题
ax2.set_xlabel("SHAP value (impact on model output)", fontsize=12)  # 设置x轴标签
output_path = str(OUTPUT_DIR / "SHAP_summary_plot.jpg")  # 定义输出图片的路径和文件名
plt.savefig(
    output_path, dpi=300, bbox_inches="tight"
)  # 将图表保存为jpg文件，分辨率为300dpi，并自动裁剪边缘空白
print(f"\n--- SHAP 摘要图已保存到: {output_path} ---")  # 打印保存成功的提示
plt.close("all")  # Interactive display removed; assets were exported above.
# =============================================================================
# 6. GeoShapley 分析与绘图
# =============================================================================
print("\n--- 步骤 5: 正在执行真实的 GeoShapley 分析 ---")  # 打印提示信息
# --- 6.1 准备数据与运行计算 ---
FULL_ANALYSIS = _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1"
background_size = min(len(df_features), 100 if FULL_ANALYSIS else 20)
explain_size = min(len(df_features), len(df_features) if FULL_ANALYSIS else 40)
background_data = df_features.sample(background_size, random_state=42).values
data_to_explain = df_features.sample(explain_size, random_state=43)
geoshap_explainer = GeoShapleyExplainer(
    model.predict, background_data
)  # 初始化GeoShapley解释器，传入模型预测函数和背景数据
print("GeoShapley 计算中，请稍候（使用单核计算）...")  # 提示用户计算正在进行
geoshapley_results = geoshap_explainer.explain(
    data_to_explain, n_jobs=1
)  # 执行GeoShapley计算，n_jobs=1表示使用单核处理

print("GeoShapley 分析完成。")  # 打印计算完成的提示
# --- 6.2 手动处理结果并绘制发散式条形图 ---
print("\n--- 步骤 6: 正在手动处理结果并绘制发散式条形图 ---")  # 打印提示信息
# ---! 如果您的坐标列名称不是 'LAT' 和 'LON', 请在此处修改 !---
coord_columns = ["LAT", "LON"]  # 定义坐标列的名称
non_spatial_features = [
    f for f in feature_names if f not in coord_columns
]  # 获取所有非空间特征的名称列表
mean_primary = pd.Series(
    geoshapley_results.primary.mean(axis=0), index=non_spatial_features
)  # 计算每个非空间特征的主要效应(primary)的平均值
mean_interaction = pd.Series(
    geoshapley_results.geo_intera.mean(axis=0), index=[f"{f} x GEO" for f in non_spatial_features]
)  # 计算每个非空间特征与地理位置交互效应(geo_intera)的平均值
mean_spatial = pd.Series(
    geoshapley_results.geo.mean(), index=["GEO"]
)  # 计算纯地理效应(geo)的平均值
df_plot = pd.concat(
    [mean_primary, mean_interaction, mean_spatial]
).reset_index()  # 将三种效应合并成一个DataFrame
df_plot.columns = ["Variable", "Value"]  # 重命名列为'Variable'和'Value'
vars_to_show = [  # 定义一个列表，指定要在图表中显示的变量及其顺序
    "PM x GEO",
    "TC x GEO",
    "TC",
    "PP x GEO",
    "MC x GEO",
    "MC",
    "WF x GEO",
    "PP",
    "WF",
    "PM",
    "ME x GEO",
    "GEO",
    "ME",
]
df_plot = df_plot[df_plot["Variable"].isin(vars_to_show)]  # 筛选出需要在图表中显示的变量
df_plot["Color"] = [
    "#e69f00" if x >= 0 else "#0072b2" for x in df_plot["Value"]
]  # 根据值的正负为条形图设置不同的颜色
df_plot["Variable"] = pd.Categorical(
    df_plot["Variable"], categories=vars_to_show, ordered=True
)  # 将变量列转换为有序类别，以控制绘图顺序
df_plot = df_plot.sort_values(
    "Variable", ascending=False
)  # 按照指定的类别顺序降序排序（因为barh绘图会反转顺序）
fig3, ax3 = plt.subplots(figsize=(10, 8))  # 创建一个新的10x8英寸的画布
ax3.barh(df_plot["Variable"], df_plot["Value"], color=df_plot["Color"])  # 绘制水平条形图
for _, row in df_plot.iterrows():  # 遍历绘图数据的每一行
    value = row["Value"]  # 获取当前行的值
    ha = "left" if value > 0 else "right"  # 根据值的正负决定文本的水平对齐方式
    offset = 0.002 if value > 0 else -0.002  # 设置一个小的偏移量，避免文本与条形重叠
    ax3.text(
        value + offset, row["Variable"], f"{value:.3f}", ha=ha, va="center", fontsize=9
    )  # 在每个条形旁边标注其数值
ax3.axvline(x=0, color="black", linewidth=0.8)  # 在x=0的位置画一条黑色的垂直线
ax3.set_title("Geoshapley values for XGB", fontsize=16, pad=20)  # 设置图表标题
ax3.set_xlabel("Geoshapley values", fontsize=12)  # 设置x轴标签
ax3.set_ylabel("Variable", fontsize=12)  # 设置y轴标签
ax3.spines["top"].set_visible(False)  # 隐藏图表的顶部边框
ax3.spines["right"].set_visible(False)  # 隐藏图表的右侧边框
current_xlim = ax3.get_xlim()  # 获取当前x轴的范围
max_abs_lim = max(abs(current_xlim[0]), abs(current_xlim[1]))  # 计算x轴范围绝对值的最大值
ax3.set_xlim(-max_abs_lim * 1.2, max_abs_lim * 1.2)  # 设置对称且有留白的x轴范围
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.savefig(
    str(OUTPUT_DIR / "geoshapley_final_plot.jpg"), dpi=300
)  # 将图表保存为jpg文件，分辨率为300dpi
plt.close("all")  # Interactive display removed; assets were exported above.
print("\n--- 所有分析和绘图已成功完成 ---")  # 打印完成提示
# =============================================================================
# 6. 绘制Beeswarm摘要图
# =============================================================================
print("\n--- 步骤 5: 正在使用内置方法绘制Beeswarm摘要图 ---")  # 打印提示信息
# 创建一个新的图窗
plt.figure(figsize=(10, 8))  # 创建一个10x8英寸的新画布
geoshapley_results.summary_plot(  # 调用GeoShapley结果对象的内置绘图方法
    max_display=30,
    include_interaction=True,  # 指定在图中包含交互效应
    cmap="RdYlBu",  # 指定颜色映射为 红-黄-蓝 (Red-Yellow-Blue)
)
# 对图表进行一些美化
ax2 = plt.gca()  # 获取当前的坐标轴对象
ax2.set_title("GeoShapley Value Summary Plot", fontsize=16)  # 设置图表标题
ax2.set_xlabel("GeoShapley value (impact on model prediction)", fontsize=12)  # 设置x轴标签
# 保存图像
# 注意：您的原代码中保存路径是绝对路径，这里为了通用性改成了相对路径
plt.savefig(
    str(OUTPUT_DIR / "geoshapley_beeswarm_plot_colored.jpg"), dpi=300, bbox_inches="tight"
)  # 将图表保存为jpg文件

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

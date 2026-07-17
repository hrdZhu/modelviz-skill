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
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skill_metrics as sm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ====================================== 2. 颜色库 =========================================
# =========================================================================================
selected_palette = 20  # 选择配色方案

color_palettes = {
    1: {
        "Linear Regression": "#4285F4",  # 线性回归模型的颜色
        "Random Forest": "#DB4437",  # 随机森林模型的颜色
        "Gradient Boosting": "#0F9D58",  # 梯度提升模型的颜色
        "Observation": "#000000",  # 观测点/参考点的颜色
        "point_std_color": "#B0B0B0",  # 标准差背景线的颜色
        "point_cor_color": "#D3D3D3",  # 相关系数背景线的颜色
        "point_rms_color": "#808080",  # 均方根误差背景线的颜色
        "plot_bgcolor": "#FFFFFF",  # 背景颜色
        "label_color": "#333333",  # 标签文本的颜色
        "tick_color": "#666666",  # 刻度数值的颜色
    },
    2: {
        "Linear Regression": "#5F6368",
        "Random Forest": "#34A853",
        "Gradient Boosting": "#7E57C2",
        "Observation": "#202124",
        "point_std_color": "#A5D6A7",
        "point_cor_color": "#D1C4E9",
        "point_rms_color": "#90A4AE",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#202124",
        "tick_color": "#5F6368",
    },
    3: {
        "Linear Regression": "#F4B400",
        "Random Forest": "#E67C73",
        "Gradient Boosting": "#B1624E",
        "Observation": "#4E342E",
        "point_std_color": "#FFCC80",
        "point_cor_color": "#FFE0B2",
        "point_rms_color": "#A1887F",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#5D4037",
        "tick_color": "#795548",
    },
    4: {
        "Linear Regression": "#1A73E8",
        "Random Forest": "#00BFA5",
        "Gradient Boosting": "#64B5F6",
        "Observation": "#004D40",
        "point_std_color": "#4DD0E1",
        "point_cor_color": "#B3E5FC",
        "point_rms_color": "#4DB6AC",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#01579B",
        "tick_color": "#00796B",
    },
    5: {
        "Linear Regression": "#8AB4F8",
        "Random Forest": "#F28B82",
        "Gradient Boosting": "#A5D6A7",
        "Observation": "#5F6368",
        "point_std_color": "#FFCDD2",
        "point_cor_color": "#E1F5FE",
        "point_rms_color": "#C8E6C9",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#4A4A4A",
        "tick_color": "#7B7B7B",
    },
    6: {
        "Linear Regression": "#E53935",
        "Random Forest": "#1E88E5",
        "Gradient Boosting": "#FFC107",
        "Observation": "#000000",
        "point_std_color": "#90CAF9",
        "point_cor_color": "#FFF59D",
        "point_rms_color": "#EF9A9A",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#333333",
        "tick_color": "#666666",
    },
    7: {
        "Linear Regression": "#03A9F4",
        "Random Forest": "#00BCD4",
        "Gradient Boosting": "#009688",
        "Observation": "#004D40",
        "point_std_color": "#4DD0E1",
        "point_cor_color": "#80DEEA",
        "point_rms_color": "#26A69A",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#006064",
        "tick_color": "#00838F",
    },
    8: {
        "Linear Regression": "#795548",
        "Random Forest": "#8D6E63",
        "Gradient Boosting": "#A1887F",
        "Observation": "#3E2723",
        "point_std_color": "#BCAAA4",
        "point_cor_color": "#D7CCC8",
        "point_rms_color": "#A1887F",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#3E2723",
        "tick_color": "#5D4037",
    },
    9: {
        "Linear Regression": "#F44336",
        "Random Forest": "#00E676",
        "Gradient Boosting": "#2979FF",
        "Observation": "#000000",
        "point_std_color": "#81D4FA",
        "point_cor_color": "#B9F6CA",
        "point_rms_color": "#FF8A80",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#333333",
        "tick_color": "#666666",
    },
    10: {
        "Linear Regression": "#E91E63",
        "Random Forest": "#9C27B0",
        "Gradient Boosting": "#673AB7",
        "Observation": "#311B92",
        "point_std_color": "#CE93D8",
        "point_cor_color": "#E1BEE7",
        "point_rms_color": "#F48FB1",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#4A148C",
        "tick_color": "#6A1B9A",
    },
    11: {
        "Linear Regression": "#4CAF50",
        "Random Forest": "#8BC34A",
        "Gradient Boosting": "#388E3C",
        "Observation": "#1B5E20",
        "point_std_color": "#AED581",
        "point_cor_color": "#C8E6C9",
        "point_rms_color": "#A5D6A7",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#1B5E20",
        "tick_color": "#2E7D32",
    },
    12: {
        "Linear Regression": "#9E9E9E",
        "Random Forest": "#757575",
        "Gradient Boosting": "#616161",
        "Observation": "#000000",
        "point_std_color": "#E0E0E0",
        "point_cor_color": "#F5F5F5",
        "point_rms_color": "#BDBDBD",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#212121",
        "tick_color": "#424242",
    },
    13: {
        "Linear Regression": "#FF9800",
        "Random Forest": "#FF5722",
        "Gradient Boosting": "#F4511E",
        "Observation": "#BF360C",
        "point_std_color": "#FFCC80",
        "point_cor_color": "#FFD180",
        "point_rms_color": "#FFAB91",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#E65100",
        "tick_color": "#F57C00",
    },
    14: {
        "Linear Regression": "#3F51B5",
        "Random Forest": "#2196F3",
        "Gradient Boosting": "#5C6BC0",
        "Observation": "#1A237E",
        "point_std_color": "#9FA8DA",
        "point_cor_color": "#C5CAE9",
        "point_rms_color": "#90CAF9",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#1A237E",
        "tick_color": "#283593",
    },
    15: {
        "Linear Regression": "#D4A5A5",
        "Random Forest": "#A5C6D4",
        "Gradient Boosting": "#D4CFA5",
        "Observation": "#594F4F",
        "point_std_color": "#CFD8DC",
        "point_cor_color": "#F0EAD6",
        "point_rms_color": "#FFEBEE",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#594F4F",
        "tick_color": "#8C7B7B",
    },
    16: {
        "Linear Regression": "#AED581",
        "Random Forest": "#4DB6AC",
        "Gradient Boosting": "#FFD54F",
        "Observation": "#33691E",
        "point_std_color": "#FFF176",
        "point_cor_color": "#DCEDC8",
        "point_rms_color": "#80CBC4",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#33691E",
        "tick_color": "#558B2F",
    },
    17: {
        "Linear Regression": "#0D47A1",
        "Random Forest": "#1976D2",
        "Gradient Boosting": "#42A5F5",
        "Observation": "#002171",
        "point_std_color": "#64B5F6",
        "point_cor_color": "#90CAF9",
        "point_rms_color": "#BBDEFB",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#0D47A1",
        "tick_color": "#1565C0",
    },
    18: {
        "Linear Regression": "#311B92",
        "Random Forest": "#4527A0",
        "Gradient Boosting": "#512DA8",
        "Observation": "#21005d",
        "point_std_color": "#9575CD",
        "point_cor_color": "#B39DDB",
        "point_rms_color": "#D1C4E9",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#311B92",
        "tick_color": "#4527A0",
    },
    19: {
        "Linear Regression": "#FBC02D",
        "Random Forest": "#D32F2F",
        "Gradient Boosting": "#43A047",
        "Observation": "#2E7D32",
        "point_std_color": "#FF8A65",
        "point_cor_color": "#FFF59D",
        "point_rms_color": "#A5D6A7",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#BF360C",
        "tick_color": "#F9A825",
    },
    20: {
        "Linear Regression": "#CE93D8",
        "Random Forest": "#90CAF9",
        "Gradient Boosting": "#A5D6A7",
        "Observation": "#455A64",
        "point_std_color": "#B0BEC5",
        "point_cor_color": "#CFD8DC",
        "point_rms_color": "#E0E0E0",
        "plot_bgcolor": "#FFFFFF",
        "label_color": "#37474F",
        "tick_color": "#546E7A",
    },
}

selected_colors = color_palettes[selected_palette]  # 从颜色库中获取指定索引的配色方案


# =========================================================================================
# ======================================3.绘图函数=========================================
# =========================================================================================
def plot_taylor_diagram(model_stats, sdev_obs, colors):
    fig = plt.figure(figsize=(10, 8))

    fig.patch.set_facecolor("#FFFFFF")  # 图形的背景颜色
    ax = plt.gca()  # 获取当前的坐标轴对象
    ax.set_facecolor("#FFFFFF")  # 坐标轴区域的背景颜色

    # 使用skill_metrics绘制泰勒图的背景框架
    sm.taylor_diagram(
        np.array([sdev_obs]),  # 标准差
        np.array([0.0]),  # 均方根误差
        np.array([1.0]),  # 相关系数
        colSTD=colors.get("point_std_color", "#606060"),  # 标准差线的颜色
        colCOR=colors.get("point_cor_color", "#808080"),  # 相关系数线的颜色
        colRMS=colors.get("point_rms_color", "#404040"),  # 均方根误差线的颜色
        styleSTD="--",  # 标准差线的样式
        styleCOR="-",  # 相关系数线的样式
        styleRMS=":",  # 均方根误差线的样式
        alpha=1,  # 线的透明度
    )

    for text in ax.texts:
        content = text.get_text()

        try:
            # 尝试将文本转换为浮点数
            value = float(content)
            # 如果转换成功，说明是数值标注
            # 如果是相关系数的数值
            if 0.0 <= value <= 1.0:
                # 调整位置（间隔）
                x, y = text.get_position()
                new_x = x * 0.97
                new_y = y * 0.97
                text.set_position((new_x, new_y))
                # 设置样式
                text.set_fontsize(11)
                text.set_fontweight("bold")
                text.set_color(colors.get("tick_color", "#666666"))
            # 如果是RMSD的数值
            else:
                text.set_fontsize(11)
                text.set_fontweight("bold")
                text.set_color(colors.get("tick_color", "#666666"))

        except ValueError:
            text.set_fontsize(14)  # 字体大小
            text.set_fontweight("bold")  # 字体粗细
            text.set_color(colors.get("label_color", "#333333"))  # 字体颜色

    # 手动设置Y轴的标签文本
    ax.set_ylabel(
        "Standard Deviation",
        fontsize=14,
        fontweight="bold",
        color=colors.get("label_color", "#333333"),
    )
    ax.set_xlabel(
        "Standard Deviation",
        fontsize=14,
        fontweight="bold",
        color=colors.get("label_color", "#333333"),
    )

    # 修改X轴和Y轴的刻度数值标签
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(11)  # 大小
        label.set_fontweight("bold")  # 粗细
        label.set_color(colors.get("tick_color", "#666666"))  # 颜色

    # 绘制各个模型的散点
    for name, group in model_stats.items():  # 遍历存储模型性能的字典
        sdev, ccoef = group["stats"]  # 获取该模型的标准差和相关系数
        style = group["style"]  # 该模型的绘图样式

        # 根据标准差和相关系数计算在极坐标系中的位置
        x = sdev * ccoef  # 该点的x坐标
        y = sdev * np.sin(np.arccos(ccoef))  # y坐标

        plt.scatter(
            x,
            y,
            s=style.get("s", 80),  # 点的大小
            c=style.get("color", "black"),  # 点的颜色
            marker=style.get("marker", "o"),  # 点的形状
            edgecolors=style.get("edgecolors", "none"),  # 点的边缘颜色
            linewidths=1.5,  # 点的边缘线宽
            zorder=10,
        )

    # 绘制观测参考点
    plt.scatter(
        sdev_obs,
        0,
        s=150,
        c=colors["Observation"],
        marker="*",
        zorder=10,
        clip_on=False,
        label="Observation",
    )

    # 设置标题
    plt.title(
        "Model Performance Evaluation",
        fontsize=20,
        pad=20,
        color=colors.get("label_color", "#333333"),
    )

    # 使用从颜色库和样式传入的颜色和形状动态创建图例
    legend_handles = [  # 创建一个列表，用于存放图例的元素
        mlines.Line2D(
            [],
            [],
            color=colors["Linear Regression"],
            marker="o",
            linestyle="None",
            markersize=8,
            label="Linear Regression",
        ),  # 创建线性回归的图例
        mlines.Line2D(
            [],
            [],
            color=colors["Random Forest"],
            marker="s",
            linestyle="None",
            markersize=8,
            label="Random Forest",
        ),  # 创建随机森林的图例
        mlines.Line2D(
            [],
            [],
            color=colors["Gradient Boosting"],
            marker="^",
            linestyle="None",
            markersize=8,
            label="Gradient Boosting",
        ),  # 创建梯度提升的图例
        mlines.Line2D(
            [],
            [],
            color=colors["Observation"],
            marker="*",
            linestyle="None",
            markersize=10,
            label="Observation",
        ),  # 创建观测点的图例
    ]
    # 设置图例
    legend = fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        numpoints=1,
        fontsize=12,
    )
    for text in legend.get_texts():  # 遍历图例中的所有文本
        text.set_color(colors.get("label_color", "#333333"))  # 设置图例文本的颜色

    ax.tick_params(axis="x", length=0)  # 去掉x轴的刻度线
    ax.tick_params(axis="y", length=0)  # 去掉y轴的刻度线

    # 设置刻度标签颜色
    for label in ax.get_xticklabels() + ax.get_yticklabels():  # 遍历所有刻度标签
        label.set_color(colors.get("tick_color", "#666666"))  # 设置刻度标签的颜色
    # 保存图片
    plt.savefig(
        str(OUTPUT_DIR / f"taylor_diagram_{selected_palette}.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(str(OUTPUT_DIR / f"taylor_diagram_{selected_palette}.pdf"), bbox_inches="tight")


# =========================================================================================
# ======================================4.数据的读取与处理=========================================
# =========================================================================================

file_path = str(DATA_DIR / "data.xlsx")  # 定义Excel文件所在的路径

data_df = pd.read_excel(file_path)  # 读取数据

X = data_df.iloc[:, :-1].values  # 提取特征数据
y = data_df.iloc[:, -1].values  # 提取目标变量数据

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# =========================================================================================
# ======================================5.模型的构建=========================================
# =========================================================================================
# 设置要进行网格搜索的模型和参数范围
param_grids = {
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [10, 20]},
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
    },
}

# 存储最佳模型
best_models = {"Linear Regression": LinearRegression()}

# 执行网格搜索
for name, grid_info in param_grids.items():  # 遍历每个模型配置
    print(f"\n--- 正在为模型 '{name}' 进行网格搜索 ---")

    grid_search = GridSearchCV(
        estimator=grid_info["model"],
        param_grid=grid_info["params"],
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
    )
    # 执行网格搜索
    grid_search.fit(X_train_scaled, y_train)
    print(f"'{name}' 找到的最佳超参数: {grid_search.best_params_}")

    # 保存最佳模型
    best_models[name] = grid_search.best_estimator_

# =========================================================================================
# ======================================6.绘图=========================================
# =========================================================================================
# 用于存储每个模型的性能统计数据
model_performance_stats = {}
# 计算标准差
observation_sdev = np.std(y_val)
# 评估每个模型
for name, model in best_models.items():  # 遍历存储了最佳模型的字典
    print(f"\n正在评估模型: {name}...")
    # 线性回归模型需要单独训练
    if name == "Linear Regression":  # 判断是否为线性回归模型
        model.fit(X_train_scaled, y_train)  # 如果是，则在标准化的训练数据上进行训练
    # 进行预测
    predictions = model.predict(X_val_scaled)  # 使用训练好的模型对标准化的验证集进行预测
    # 计算泰勒图所需的两个指标：标准差和相关系数
    pred_sdev = np.std(predictions)  # 计算模型预测值的标准差
    corr = np.corrcoef(y_val, predictions)[0, 1]  # 计算预测值与真实值之间的相关系数

    # 计算预测值和观测值的离差（减去各自的均值）
    pred_anomalies = predictions - np.mean(predictions)
    obs_anomalies = y_val - np.mean(y_val)
    # 计算中心化均方根误差
    rmsd = np.sqrt(np.mean((pred_anomalies - obs_anomalies) ** 2))
    # 打印出模型的各项性能指标
    print(f"标准差: {pred_sdev:.3f}")
    print(f"相关系数: {corr:.3f}")
    print(f"中心化均方根误差: {rmsd:.3f}")
    # 存储统计数据
    model_performance_stats[name] = {  # 将计算出的统计数据存入字典
        "stats": (np.array([pred_sdev]), np.array([corr])),  # 存储标准差和相关系数
    }
# 为每个模型设定绘图样式
model_performance_stats["Linear Regression"]["style"] = {
    "color": selected_colors["Linear Regression"],
    "marker": "o",
    "s": 80,
}
model_performance_stats["Random Forest"]["style"] = {
    "color": selected_colors["Random Forest"],
    "marker": "s",
    "s": 80,
}
model_performance_stats["Gradient Boosting"]["style"] = {
    "color": selected_colors["Gradient Boosting"],
    "marker": "^",
    "s": 80,
}

print("\n所有模型评估完毕，开始绘图...")
# 调用绘图函数
plot_taylor_diagram(model_performance_stats, observation_sdev, selected_colors)

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

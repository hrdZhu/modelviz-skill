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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,  # 正常显示负号 (避免显示为方框)
    }
)
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# ====================================== 2. 颜色库 =========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: {
        "RS": "#2ca02c",
        "ZB": "#415a99",
        "WZ": "#87cefa",
        "Other": "#d62728",
        "fit_line": "#2f2f2f",
        "cross_hair": "darkgray",
        "confidence_fill": "#d3d3d3",
    },
    2: {
        "RS": "#1f77b4",
        "ZB": "#ff7f0e",
        "WZ": "#2ca02c",
        "Other": "#d62728",
        "fit_line": "#2f2f2f",
        "cross_hair": "darkgray",
        "confidence_fill": "#d3d3d3",
    },
    3: {
        "RS": "#e4572e",
        "ZB": "#f3a712",
        "WZ": "#a8c686",
        "Other": "#662e9b",
        "fit_line": "#54494b",
        "cross_hair": "#7d7071",
        "confidence_fill": "#e0dcd3",
    },
    4: {
        "RS": "#003f5c",
        "ZB": "#58508d",
        "WZ": "#bc5090",
        "Other": "#ffa600",
        "fit_line": "#4a4a4a",
        "cross_hair": "#8c8c8c",
        "confidence_fill": "#e0e0e0",
    },
    5: {
        "RS": "#4878d0",
        "ZB": "#ee854a",
        "WZ": "#6acc64",
        "Other": "#d65f5f",
        "fit_line": "#404040",
        "cross_hair": "#999999",
        "confidence_fill": "#e8e8e8",
    },
    6: {
        "RS": "#f94144",
        "ZB": "#f8961e",
        "WZ": "#90be6d",
        "Other": "#277da1",
        "fit_line": "#3a3a3a",
        "cross_hair": "gray",
        "confidence_fill": "#f0f0f0",
    },
    7: {
        "RS": "#a2d2ff",
        "ZB": "#ffafcc",
        "WZ": "#bde0fe",
        "Other": "#ffc8dd",
        "fit_line": "#6b705c",
        "cross_hair": "#a5a58d",
        "confidence_fill": "#f5f3f4",
    },
    8: {
        "RS": "#2d6a4f",
        "ZB": "#40916c",
        "WZ": "#74c69d",
        "Other": "#b22222",
        "fit_line": "#212529",
        "cross_hair": "#8d8d8d",
        "confidence_fill": "#e9f5db",
    },
    9: {
        "RS": "#e07a5f",
        "ZB": "#3d405b",
        "WZ": "#81b29a",
        "Other": "#f2cc8f",
        "fit_line": "#3d405b",
        "cross_hair": "#8e8e8e",
        "confidence_fill": "#f4f1de",
    },
    10: {
        "RS": "#0077b6",
        "ZB": "#00b4d8",
        "WZ": "#90e0ef",
        "Other": "#fca311",
        "fit_line": "#03045e",
        "cross_hair": "#adb5bd",
        "confidence_fill": "#caf0f8",
    },
    11: {
        "RS": "#432371",
        "ZB": "#712B75",
        "WZ": "#C74B50",
        "Other": "#FA9905",
        "fit_line": "#3d3d3d",
        "cross_hair": "#989898",
        "confidence_fill": "#f3e9dc",
    },
    12: {
        "RS": "#005f73",
        "ZB": "#9b2226",
        "WZ": "#0a9396",
        "Other": "#ee9b00",
        "fit_line": "#262626",
        "cross_hair": "#7a7a7a",
        "confidence_fill": "#e9ecef",
    },
    13: {
        "RS": "#440154",
        "ZB": "#31688e",
        "WZ": "#35b779",
        "Other": "#fde725",
        "fit_line": "#2d2d2d",
        "cross_hair": "gray",
        "confidence_fill": "#f0f0f0",
    },
    14: {
        "RS": "#0d0887",
        "ZB": "#7e03a8",
        "WZ": "#cc4778",
        "Other": "#f89540",
        "fit_line": "#2b2b2b",
        "cross_hair": "#7c7c7c",
        "confidence_fill": "#eeeeee",
    },
    15: {
        "RS": "#0b2e59",
        "ZB": "#235a82",
        "WZ": "#5386a6",
        "Other": "#f06543",
        "fit_line": "#212121",
        "cross_hair": "#8f8f8f",
        "confidence_fill": "#dfe7ed",
    },
    16: {
        "RS": "#a44200",
        "ZB": "#d58936",
        "WZ": "#6e0e0a",
        "Other": "#efca08",
        "fit_line": "#4d2d1d",
        "cross_hair": "#796459",
        "confidence_fill": "#fdf8e1",
    },
    17: {
        "RS": "#636efa",
        "ZB": "#ef553b",
        "WZ": "#00cc96",
        "Other": "#ab63fa",
        "fit_line": "#3d3d3d",
        "cross_hair": "#989898",
        "confidence_fill": "#f0f0f0",
    },
    18: {
        "RS": "#3c096c",
        "ZB": "#7b2cbf",
        "WZ": "#c75298",
        "Other": "#5a189a",
        "fit_line": "#240046",
        "cross_hair": "#8d8d8d",
        "confidence_fill": "#e0c3fc",
    },
    19: {
        "RS": "#00f5d4",
        "ZB": "#00bbf9",
        "WZ": "#f15bb5",
        "Other": "#fee440",
        "fit_line": "#f8f9fa",
        "cross_hair": "#adb5bd",
        "confidence_fill": "#343a40",
    },
    20: {
        "RS": "#4285F4",
        "ZB": "#DB4437",
        "WZ": "#0F9D58",
        "Other": "#F4B400",
        "fit_line": "#343434",
        "cross_hair": "#808080",
        "confidence_fill": "#EAEAEA",
    },
}


# =========================================================================================
# ====================================== 3. 绘图函数 =========================================
# =========================================================================================
def create_parity_plot(
    ax, y_true, y_pred, point_types, r2, title, colors, prediction_interval_factor
):
    # 每个数据点类别在图中的样式
    style_map = {
        "RS": {"marker": "o", "color": colors["RS"], "label": "RS"},  # RS类别的样式
        "ZB": {"marker": "D", "color": colors["ZB"], "label": "ZB"},  # ZB类别的样式
        "WZ": {"marker": "D", "color": colors["WZ"], "label": "WZ"},  # WZ类别的样式
        "Other": {"marker": "^", "color": colors["Other"], "label": "Other"},  # ther类别的样式
    }
    # 遍历每个类别，并按类别绘制散点图
    for cat in style_map.keys():
        # 创建一个布尔掩码，用于选择当前类别的数据点
        mask = point_types == cat
        # 绘制当前类别的散点图
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            marker=style_map[cat]["marker"],  # 点的形状
            color=style_map[cat]["color"],  # 点的填充颜色
            edgecolor="black",  # 点的边缘颜色
            linewidth=0.8,  # 点的边缘线宽
            label=style_map[cat]["label"],  # 该类别在图例中的标签
            zorder=3,
        )

    # 将x轴设置为对数尺度
    ax.set_xscale("log")
    # 将y轴设置为对数尺度
    ax.set_yscale("log")

    # 坐标轴的显示范围
    lims = [10**2, 10**6]
    # x轴的范围
    ax.set_xlim(lims)
    # y轴的范围
    ax.set_ylim(lims)

    # 绘制完美预测的参考线
    ax.plot(lims, lims, linestyle="--", color=colors["fit_line"], linewidth=2, zorder=1)

    # 绘制一个表示置信区间的填充区域
    upper_bound = [l * prediction_interval_factor for l in lims]
    lower_bound = [l / prediction_interval_factor for l in lims]
    ax.fill_between(
        lims, lower_bound, upper_bound, color=colors["confidence_fill"], alpha=0.5, zorder=0
    )

    # 绘制水平和垂直的辅助虚线
    threshold = 5000  # 辅助线的阈值位置
    ax.axhline(
        y=threshold, linestyle="--", color=colors["cross_hair"], linewidth=1.5, dashes=(5, 5)
    )  # 绘制水平虚线
    ax.axvline(
        x=threshold, linestyle="--", color=colors["cross_hair"], linewidth=1.5, dashes=(5, 5)
    )  # 绘制垂直虚线

    ax.set_xlabel(
        r"$\kappa(300\mathrm{K}) \, [\mathrm{Wm}^{-1}\mathrm{K}^{-1}]$", fontsize=16
    )  # x轴标题
    ax.set_ylabel(
        r"$\kappa^{\mathrm{ML}}(300\mathrm{K}) \, [\mathrm{Wm}^{-1}\mathrm{K}^{-1}]$", fontsize=16
    )  # y轴标题
    ax.set_title(title, fontsize=16)  # 图标题

    # 添加图例
    ax.legend(loc="upper left", frameon=False, fontsize=12, handletextpad=0.2)

    # 添加评估指标结果到图上
    metrics_text = f"$R^2$: {r2:.2f}"  # 格式化指标文本
    ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=12, va="bottom", ha="right")

    for spine in ax.spines.values():  # 遍历所有图框
        spine.set_linewidth(1.2)  # 设置图框粗细
    ax.tick_params(
        axis="both", which="both", direction="in", top=True, right=True, labelsize=12
    )  # 设置刻度线向内，上下右侧都显示
    ax.tick_params(axis="both", which="major", length=6, width=1.2)  # 设置主刻度线的长度和宽度
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)  # 设置次刻度线的长度和宽度


# =========================================================================================
# ======================================4.数据的读取与处理=========================================
# =========================================================================================
# 选择配色方案
selected_scheme = 19
# 提取配色
selected_colors = COLOR_SCHEMES[selected_scheme]
# 输入数据
file_path = str(DATA_DIR / "simulation_data.xlsx")
# 读取数据
df = pd.read_excel(file_path)

# 提取特征
X = df.iloc[:, 0:3].values
# 提取目标
y = df.iloc[:, 3].values
# 提取类别列
point_types = df.iloc[:, 4].values

# 划分数据
X_train, X_test, y_train, y_test, types_train, types_test = train_test_split(
    X, y, point_types, test_size=0.2, random_state=42
)

# =========================================================================================
# ====================================== 3. 绘图函数 =========================================
# =========================================================================================
# 超参数网格
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30],
}
# 实例化一个随机森林回归器模型
rf = RandomForestRegressor(random_state=42)
# 进行网格搜索和交叉验证
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, np.log10(y_train))
# 获取最佳模型
best_model = grid_search.best_estimator_
print(f"最佳参数: {grid_search.best_params_}")

# 使用最佳模型对训练集特征进行预测
log_pred_train = best_model.predict(X_train)
y_pred_train = 10**log_pred_train
# 使用最佳模型对测试集特征进行预测
log_pred_test = best_model.predict(X_test)
y_pred_test = 10**log_pred_test


# 性能评估
train_r2 = r2_score(np.log10(y_train), log_pred_train)
test_r2 = r2_score(np.log10(y_test), log_pred_test)
print(f"训练集性能: R^2 = {train_r2:.2f}")
print(f"训练集性能: R^2 = {train_r2:.2f}")

# 计算训练集在log10尺度上的残差
log_residuals_train = np.log10(y_train) - log_pred_train

# 计算残差的标准差
std_dev_log_residuals = np.std(log_residuals_train)

# 计算95%预测区间的乘法因子
prediction_interval_factor = 10 ** (1.96 * std_dev_log_residuals)


# =========================================================================================
# ====================================== 3. 绘图函数 =========================================
# =========================================================================================
# 训练结果
fig_train, ax_train = plt.subplots(figsize=(6.5, 6))
create_parity_plot(
    ax_train,
    y_train,
    y_pred_train,
    types_train,
    train_r2,
    "Training Set",
    selected_colors,
    prediction_interval_factor,
)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"train_{selected_scheme}.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(OUTPUT_DIR / f"train_{selected_scheme}.pdf"), bbox_inches="tight")

# 验证结果
fig_val, ax_val = plt.subplots(figsize=(6.5, 6))

create_parity_plot(
    ax_val,
    y_test,
    y_pred_test,
    types_test,
    test_r2,
    "Validation Set",
    selected_colors,
    prediction_interval_factor,
)
plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"validation_{selected_scheme}.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(OUTPUT_DIR / f"validation_{selected_scheme}.pdf"), bbox_inches="tight")


# 组合图
fig_split, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
create_parity_plot(
    axes[0],
    y_train,
    y_pred_train,
    types_train,
    train_r2,
    "Training Set",
    selected_colors,
    prediction_interval_factor,
)
create_parity_plot(
    axes[1],
    y_test,
    y_pred_test,
    types_test,
    test_r2,
    "Validation Set",
    selected_colors,
    prediction_interval_factor,
)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / f"{selected_scheme}.png"), dpi=300, bbox_inches="tight")
plt.savefig(str(OUTPUT_DIR / f"{selected_scheme}.pdf"), bbox_inches="tight")

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

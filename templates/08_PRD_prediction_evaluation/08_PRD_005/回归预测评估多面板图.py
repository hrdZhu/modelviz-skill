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
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
# =========================================================================================
# ====================================== 2.颜色库 =========================================
# =========================================================================================
COLOR_SCHEMES = {
    1: ["RdYlBu_r", "Blues", "Reds", "#40E0D0"],
    2: ["viridis", "Greens", "YlOrBr", "#FF6347"],
    3: ["plasma", "Purples", "Oranges", "#00FF7F"],
    4: ["inferno", "YlOrRd", "GnBu", "#ADFF2F"],
    5: ["magma", "PuRd", "YlGn", "#00CED1"],
    6: ["cividis", "YlOrBr", "Blues", "#FFD700"],
    7: ["Spectral_r", "RdYlGn", "RdBu", "#BA55D3"],
    8: ["coolwarm", "cool", "autumn", "#32CD32"],
    9: ["YlGnBu", "YlGn", "BuPu", "#FF4500"],
    10: ["RdYlGn_r", "Greys", "YlOrRd", "#1E90FF"],
    11: ["nipy_spectral", "GnBu", "PuRd", "#7FFF00"],
    12: ["autumn_r", "YlOrRd", "winter", "#FF1493"],
    13: ["summer_r", "YlGn", "spring", "#DAA520"],
    14: ["magma_r", "binary", "bone", "#8A2BE2"],
    15: ["PuBuGn", "PuBu", "BuGn", "#20B2AA"],
    16: ["terrain", "ocean", "gist_earth", "#DC143C"],
    17: ["cubehelix", "Wistia", "hot", "#7B68EE"],
    18: ["gnuplot2", "rainbow", "nipy_spectral", "#F08080"],
    19: ["jet", "cool", "hot", "#00FA9A"],
    20: ["hot", "copper", "pink", "#4682B4"],
}
# =========================================================================================
# ======================================3.形状标记库=======================================
# =========================================================================================
MARKER_LIB = {
    1: "o",
    2: r"$\heartsuit$",
    3: r"$\diamondsuit$",
    4: r"$\triangle$",
    5: r"$\clubsuit$",
    6: r"$\spadesuit$",
    7: r"$\star$",
    8: r"$\alpha$",
    9: r"$\beta$",
    10: r"$\gamma$",
    11: r"$\delta$",
    12: r"$\epsilon$",
    13: r"$\zeta$",
    14: r"$\theta$",
    15: r"$\lambda$",
    16: r"$\mu$",
    17: r"$\pi$",
    18: r"$\sigma$",
    19: r"$\Phi$",
    20: r"$\Omega$",
}


# =========================================================================================
# ======================================4.绘制渐变直方图的函数=======================================
# =========================================================================================
def draw_gradient_hist(ax, data, bins=30, orientation="vertical", cmap_name="Blues"):
    n, bins_edges = np.histogram(
        data, bins=bins, density=True
    )  # 对数据进行直方图统计并计算概率密度
    cm = plt.get_cmap(cmap_name)  # 获取颜色映射对象
    for i in range(len(n)):  # 遍历每一个柱子
        if n[i] > 0:  # 如果该柱子的高度大于0则进行绘制
            left, right = bins_edges[i], bins_edges[i + 1]  # 获取当前柱子的左右边界坐标
            if orientation == "vertical":  # 如果是垂直
                grad = np.linspace(0.2, 0.8, 100).reshape(100, 1)  # 创建渐变数组
                # 渐变图像
                ax.imshow(
                    grad,
                    extent=[left, right, 0, n[i]],  # 图像填充的范围左, 右， 底, 柱子高度
                    aspect="auto",  # 自动调整图像的纵横比以填充指定的范围
                    cmap=cm,  # 颜色映射
                    origin="lower",  # 原点位于左下角
                    zorder=1,
                )
                # 边框
                rect = plt.Rectangle(
                    (left, 0),  # 左下角起点坐标
                    right - left,  # 矩形的宽度
                    n[i],  # 设置矩形的高度
                    edgecolor="black",  # 矩形边框颜色
                    fill=False,  # 不填充颜色
                    linewidth=0.8,  # 边框线条的宽度
                    zorder=2,
                )
            else:  # 如果是水平
                grad = np.linspace(0.2, 0.8, 100).reshape(1, 100)  # 创建渐变数组
                ax.imshow(
                    grad,
                    extent=[0, n[i], left, right],
                    aspect="auto",
                    cmap=cm,
                    origin="lower",
                    zorder=1,
                )
                rect = plt.Rectangle(
                    (0, left),
                    n[i],
                    right - left,
                    edgecolor="black",
                    fill=False,
                    linewidth=0.8,
                    zorder=2,
                )
            ax.add_patch(rect)  # 边框添加到指定的坐标轴中


# =========================================================================================
# ======================================6.主图绘制函数=======================================
# =========================================================================================
def plot_academic_evaluation(
    y_true, y_pred, label_id, model_real_name, save_path_base, color_list, marker_cfg
):
    main_cmap = color_list[0]  # 主图的散点颜色映射
    marg_x_cmap = color_list[1]  # 横向边际直方图的颜色映射
    marg_y_cmap = color_list[2]  # 纵向边际直方图的颜色映射
    line_color = color_list[3]  # 参考线的颜色

    fig = plt.figure(figsize=(8, 9), dpi=100)  # 初始化画布

    gs_outer = gridspec.GridSpec(
        2, 1, height_ratios=[6, 1], hspace=0.04
    )  # 定义外部布局上下两部分，主图与残差图
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        2,  # 2行
        2,  # 2列
        subplot_spec=gs_outer[0],  # 放置在外部网格的第一行区域内
        width_ratios=[7, 1],  # 左侧主图与右侧边际图
        height_ratios=[1, 7],  # 上方边际图与下方主图
        wspace=0,  # 子图之间的水平间距
        hspace=0,
    )  # 子图之间的垂直间距

    ax_marg_x = fig.add_subplot(gs_inner[0, 0])  # 创建顶部的横向边际分布图
    ax_joint = fig.add_subplot(gs_inner[1, 0])  # 创建中央的散点回归图
    ax_marg_y = fig.add_subplot(
        gs_inner[1, 1], sharey=ax_joint
    )  # 创建右侧的纵向边际分布图，并共享Y轴
    ax_resid = fig.add_subplot(gs_outer[1, 0])  # 创建底部的残差分布图

    fig.canvas.draw()  # 执行初步渲染以确定各组件的几何位置
    pos_joint = ax_joint.get_position()  # 获取散点回归图实际坐标
    pos_resid = ax_resid.get_position()  # 获取残差图在画布上的实际坐标
    ax_resid.set_position(
        [pos_joint.x0, pos_resid.y0, pos_joint.width, pos_resid.height]
    )  # 调整残差图宽度使其与上方主图对齐

    error = y_true - y_pred  # 计算真实值与预测值之间的残差
    r2 = r2_score(y_true, y_pred)  # 计算R1
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # 计算RMSE
    mae = mean_absolute_error(y_true, y_pred)  # 计算MAE
    n_samples = len(y_true)  # 样本总数

    # 循环处理主图和残差图的样式
    for ax in [ax_joint, ax_resid]:
        for spine in ax.spines.values():  # 遍历坐标轴的四个边框
            spine.set_linewidth(2)  # 设置边框线宽
        ax.tick_params(width=2, length=4, labelsize=18)  # 设置刻度线的粗细、长度及标签字体大小

    # 处理边际分布图的样式
    for ax in [ax_marg_x, ax_marg_y]:
        ax.axis("off")  # 关闭所有坐标轴线和标签

    # 散点图绘制
    ax_joint.scatter(
        y_pred,  # X轴
        y_true,  # Y轴
        c=y_true,  # 散点的映射颜色随真实值的数值变化
        cmap=main_cmap,  # 颜色映射方案
        marker=marker_cfg,  # 散点的形状样式
        edgecolor="black",  # 外边框线
        linewidth=0.5,  # 外边框的线条粗细
        s=40,  # 散点的大小
        alpha=1,  # 透明度
        zorder=10,
    )

    max_val = max(y_true.max(), y_pred.max()) * 1.1  # 坐标轴的最大值
    ax_joint.set_xlim(0, max_val)  # X轴的范围
    ax_joint.set_ylim(0, max_val)  # Y轴的范围
    # 绘制参考线
    ax_joint.plot(
        [0, max_val],  # 起止点的横坐标
        [0, max_val],  # 起止点的纵坐标
        color=line_color,  # 参考线的颜色
        linestyle="--",  # 线型
        linewidth=1.5,  # 线条粗细
        zorder=5,
    )
    # 去掉x轴数值标注
    ax_joint.tick_params(labelbottom=False, labelleft=True)

    # 添加文本标注
    ax_joint.text(
        0.05,
        0.93,
        f"Model: {model_real_name}",
        transform=ax_joint.transAxes,
        fontweight="bold",
        fontsize=18,
    )  # 模型名称
    ax_joint.text(0.05, 0.86, f"$R^2$={r2:.4f}", transform=ax_joint.transAxes, fontsize=18)  # R2
    ax_joint.text(
        0.05, 0.79, f"RMSE={rmse:.4f} MPa", transform=ax_joint.transAxes, fontsize=18
    )  # RMSE
    ax_joint.text(
        0.05, 0.72, f"MAE={mae:.4f} MPa", transform=ax_joint.transAxes, fontsize=18
    )  # MAE
    ax_joint.text(
        0.05, 0.65, f"N={n_samples}", transform=ax_joint.transAxes, fontsize=18
    )  # 样本数量
    # 纵轴标题
    ax_joint.set_ylabel(f"$p_{{{label_id}}}$ Actual Value (MPa)", fontsize=24)
    # 子图编号
    ax_joint.text(
        -0.15,
        1.1,
        f"{chr(96 + int(label_id))}",
        transform=ax_joint.transAxes,
        fontsize=28,
        fontweight="bold",
    )

    # 边缘直方图绘制
    # 顶部
    draw_gradient_hist(
        ax_marg_x,  # 在顶部的横向边际坐标轴上绘图
        y_true,  # 绘图数据
        bins=30,  # 柱子数量
        orientation="vertical",  # 柱子的方向
        cmap_name=marg_x_cmap,
    )  # 颜色映射方案
    # 右侧
    draw_gradient_hist(ax_marg_y, y_true, bins=30, orientation="horizontal", cmap_name=marg_y_cmap)
    # 用于绘制核密度估计曲线
    xx = np.linspace(0, max_val, 200)
    # 在顶部图绘制KDE概率密度曲线
    ax_marg_x.plot(
        xx,  # 以生成的连续数值作为横坐标
        gaussian_kde(y_true)(
            xx
        ),  # 使用高斯核密度估计函数计算 y_true 数据在对应点的概率密度值作为纵坐标
        color="#E67E22",  # 曲线颜色
        linewidth=1.2,  # 线条宽度
        zorder=5,
    )
    # 在右侧图绘制KDE概率密度曲线
    ax_marg_y.plot(gaussian_kde(y_true)(xx), xx, color="#1ABC9C", linewidth=1.2, zorder=5)
    ax_marg_x.set_xlim(0, max_val)  # 顶部图的X轴范围
    ax_marg_y.set_ylim(0, max_val)  # 右侧图的Y轴范围

    # 残差图绘制
    ax_resid.scatter(
        y_pred,  # X轴
        error,  # Y轴
        c=np.abs(error),  # 散点颜色映射，依据残差的绝对值大小来确定
        cmap=main_cmap,  # 颜色映射方案
        marker=marker_cfg,  # 散点形状样式
        vmin=0,  # 颜色映射的最小值
        vmax=np.max(np.abs(error)),  # 设置颜色映射的最大值
        edgecolor="black",  # 外边框
        linewidth=0.5,  # 散点外边框的线条粗细
        s=30,  # 散点大小
        alpha=1,  # 透明度
        zorder=10,
    )
    # y=0 的基准线
    ax_resid.axhline(0, color="black", linestyle="--", linewidth=1, zorder=5)

    ax_resid.set_xlim(0, max_val)  # 残差图的X轴范围
    fig.canvas.draw()  # 再次执行渲染以确定刻度位置
    yticks = ax_resid.get_yticks()  # 获取残差图当前的Y轴刻度值
    for y_t in yticks:  # 遍历刻度值
        if y_t != 0:  # 排除0刻度线
            # 绘制淡灰色的残差参考网格线
            ax_resid.axhline(
                y_t,  # 纵坐标值
                color="gray",  # 线条颜色
                linestyle="--",  # 线型
                linewidth=0.8,  # 宽度
                alpha=0.3,  # 透明度
                zorder=1,
            )
    # 残差图横轴标题
    ax_resid.set_xlabel(f"$p_{{{label_id}}}$ Approximate Value (MPa)", fontsize=20)
    # 残差图纵轴标题
    ax_resid.set_ylabel("Error", fontsize=20)
    # 保存
    plt.savefig(save_path_base + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(save_path_base + ".pdf", bbox_inches="tight")
    plt.close()


# =========================================================================================
# ======================================6.模型训练与调优函数=======================================
# =========================================================================================
def train_best_models(X_train, y_train):
    # 模型列表及超参数搜索空间
    model_configs = [
        ("1", Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
        ("2", SVR(), {"C": [1, 10, 100], "gamma": ["scale"]}),
        ("3", RandomForestRegressor(random_state=42), {"n_estimators": [100, 200]}),
        (
            "4",
            GradientBoostingRegressor(random_state=42),
            {"learning_rate": [0.01, 0.1], "n_estimators": [100]},
        ),
        ("5", KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
        ("6", DecisionTreeRegressor(random_state=42), {"max_depth": [5, 10, None]}),
    ]
    best_models = {}  # 用于存储训练好的最佳模型
    cv = KFold(n_splits=3, shuffle=True, random_state=42)  # 交叉验证方案
    # 遍历模型配置
    for name, model, params in model_configs:
        print(f"正在调优模型 p_{name} ({model.__class__.__name__})")
        # 初始化网格搜索
        grid = GridSearchCV(model, params, cv=cv, scoring="r2", n_jobs=-1)
        # 执行搜索
        grid.fit(X_train, y_train)
        print(f"模型 p_{name} ({model.__class__.__name__}) 的最佳参数为: {grid.best_params_}")
        # 保存最佳模型
        best_models[name] = (grid.best_estimator_, model.__class__.__name__)
    return best_models


# =========================================================================================
# ======================================7.图片拼接函数=======================================
# =========================================================================================
def stitch_images_grid(image_paths, n_cols, output_filename_base):
    if not image_paths:
        return  # 如果路径列表为空则直接返回
    images = [Image.open(path) for path in image_paths]  # 打开路径列表中的所有图片文件
    img_width, img_height = images[0].size  # 获取第一张图片的宽度和高度作为标准尺寸
    n_rows = (len(images) + n_cols - 1) // n_cols  # 根据总图数和列数计算所需的行数
    composite_image = Image.new(
        "RGB", (n_cols * img_width, n_rows * img_height), color="white"
    )  # 创建一个白色底图
    for i, img in enumerate(images):  # 遍历所有读取的图片
        row, col = i // n_cols, i % n_cols  # 计算当前图片在大图中的行列索引
        composite_image.paste(img, (col * img_width, row * img_height))  # 将当前图片粘贴到指定位置
        img.close()
    composite_image.save(str(OUTPUT_DIR / f"{output_filename_base}.png"))
    composite_image.save(str(OUTPUT_DIR / f"{output_filename_base}.pdf"))


# =========================================================================================
# ======================================8.执行部分=======================================
# =========================================================================================
if __name__ == "__main__":
    scheme_index = 1  # 颜色方案
    MARKER_index = 12  # 标记方案
    # 获取当前选定的配色和标记
    current_color_list = COLOR_SCHEMES.get(scheme_index, COLOR_SCHEMES[1])
    current_marker = MARKER_LIB.get(MARKER_index, "o")

    save_dir = str(OUTPUT_DIR)  # 绘图结果保存
    df = pd.read_excel(str(DATA_DIR / "data.xlsx"))  # 读取数据
    X = df.iloc[:, :-1].values  # 特征
    y = df.iloc[:, -1].values  # 目标
    model_save_dir = str(OUTPUT_DIR)  # 模型存储路径
    result_save_dir = str(OUTPUT_DIR)  # 预测结果存储路径
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 标准化处理
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 调用函数开始模型训练和调优
    trained_models = train_best_models(X_train_scaled, y_train)

    train_paths, test_paths = [], []  # 用于记录图片路径
    # 遍历每个训练好的模型，执行预测、保存模型、保存结果、绘图
    for name, (model, m_type) in trained_models.items():
        model_filename = f"model_p{name}_{m_type}.pkl"  # 模型保存的文件名
        # 保存模型
        joblib.dump(model, os.path.join(model_save_dir, model_filename))
        # 使用训练好的模型对测试集进行预测
        y_test_pred = model.predict(X_test_scaled)
        test_error = y_test - y_test_pred  # 残差
        # 构建测试集预测结果的DataFrame
        df_test_res = pd.DataFrame(
            {
                "Actual": y_test,  # 真值
                "Predicted": y_test_pred,  # 预测值
                "Error": test_error,  # 误差
            }
        )
        # 保存
        df_test_res.to_excel(
            os.path.join(result_save_dir, f"test_results_p{name}.xlsx"), index=False
        )

        # 测试集图保存的基本路径
        p_test = os.path.join(save_dir, f"test_p{name}_{scheme_index}_{MARKER_index}")
        # 调用绘图函数
        plot_academic_evaluation(
            y_test, y_test_pred, name, m_type, p_test, current_color_list, current_marker
        )
        test_paths.append(p_test + ".png")  # 记录图片路径

        # 对训练集进行预测
        y_train_pred = model.predict(X_train_scaled)
        train_error = y_train - y_train_pred  # 误差
        # 训练集结果
        df_train_res = pd.DataFrame(
            {"Actual": y_train, "Predicted": y_train_pred, "Error": train_error}
        )
        # 保存
        df_train_res.to_excel(
            os.path.join(result_save_dir, f"train_results_p{name}.xlsx"), index=False
        )

        # 训练集图保存路径
        p_train = os.path.join(save_dir, f"train_p{name}_{scheme_index}_{MARKER_index}")
        # 调用绘图函数
        plot_academic_evaluation(
            y_train, y_train_pred, name, m_type, p_train, current_color_list, current_marker
        )
        train_paths.append(p_train + ".png")  # 记录路径

    # 组合图拼接
    stitch_images_grid(
        test_paths,
        n_cols=3,
        output_filename_base=f"Final_Validation_Combined_{scheme_index}_{MARKER_index}",
    )
    stitch_images_grid(
        train_paths,
        n_cols=3,
        output_filename_base=f"Final_Training_Combined_{scheme_index}_{MARKER_index}",
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

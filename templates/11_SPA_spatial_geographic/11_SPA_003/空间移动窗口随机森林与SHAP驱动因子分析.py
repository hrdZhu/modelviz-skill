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
# ====================================== 1. 环境设置 =======================================
# =========================================================================================
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

# =========================================================================================
# ======================================2.基础参数配置 =======================================
# =========================================================================================
output_dir = str(OUTPUT_DIR)  # 输出路径
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
target_var_name = "JS"  # 目标变量
feature_var_names = ["QW", "RZ"]  # 特征变量
# 将特征变量名映射到特定的数字标签，用于最后的结果
feature_numerical_labels = {
    "QW": 1,
    "RZ": 2,
}
# 量的有效数据范围
valid_ranges = {
    "JS": (0, 100),
    "QW": (-100, 100),
    "RZ": (0, 1000),
}
# 最少有效时间序列长度
min_observations_required = 10
# 每个变量对应的栅格数据文件所在的文件夹路径
variable_folder_paths = {
    "JS": str(DATA_DIR),
    "QW": str(DATA_DIR),
    "RZ": str(DATA_DIR),
}
# 定义随机森林的超参数搜索空间
param_grid = {"n_estimators": [5, 10], "max_depth": [1]}
# 定义随机森林回归模型的参数
rf_params = {"random_state": 0, "n_jobs": 1}

excel_output_dir = os.path.join(output_dir, "excel_output")  # Excel输出目录
os.makedirs(excel_output_dir, exist_ok=True)  # 创建Excel输出目录
# 主要是怕数据太多一个装不下
MAX_ROWS_PER_EXCEL = 1000000  # 设置每个Excel文件的最大行数
current_excel_file_index = 0  # 当前Excel文件索引
excel_data = []  # 用于存储当前批次的Excel数据
WINDOW_SIZE = 5  # 窗口设置


# =========================================================================================
# ====================================== 3. 读取栅格数据的函数=======================================
# =========================================================================================
def read_raster(path):
    with rasterio.open(path) as src:  # 使用rasterio打开指定路径的栅格文件
        data = src.read(1).astype(np.float32)  # 读取第一个波段的数据，并将其数据类型转换为float32
        nodata = src.nodata  # 获取栅格文件定义的NoData值
        if nodata is not None:  # 如果文件定义了NoData值
            data[data == nodata] = np.nan  # 将数组中等于NoData值的所有像元替换为NumPy的NaN
        return data, src.meta  # 返回读取的数据数组和栅格文件的元数据信息


# =========================================================================================
# ====================================== 4. 写入栅格数据的函数=======================================
# =========================================================================================
def write_raster(path, array, meta):
    meta_copy = meta.copy()  # 复制元数据字典，以避免修改原始元数据
    # 更新元数据
    meta_copy.update(
        {
            "count": 1,  # 输出栅格的波段数为1
            "dtype": "float32",  # 输出栅格的数据类型
            "nodata": np.nan,  # 输出栅格的NoData值
            "compress": "lzw",  # 输出栅格的压缩方式
        }
    )
    with rasterio.open(
        path, "w", **meta_copy
    ) as dst:  # 以写入模式('w')打开指定路径的栅格文件，并传入更新后的元数据
        dst.write(
            array.astype(np.float32), 1
        )  # 将数据数组转换为float32类型后写入栅格文件的第一个波段


def save_array_preview(
    array,
    path,
    title,
    cmap="tab10",
    vmin=None,
    vmax=None,
    cbar_label="value",
    cbar_ticks=None,
    cbar_ticklabels=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(preview)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    if valid.any():
        masked = np.ma.masked_invalid(preview)
        if cbar_ticks is not None and cbar_ticklabels is not None:
            from matplotlib.colors import BoundaryNorm, ListedColormap

            base_cmap = plt.get_cmap(cmap)
            colors = base_cmap(np.linspace(0, 1, len(cbar_ticklabels)))
            cmap_obj = ListedColormap(colors)
            bounds = [cbar_ticks[0] - 0.5] + [
                (left + right) / 2 for left, right in zip(cbar_ticks[:-1], cbar_ticks[1:])
            ] + [cbar_ticks[-1] + 0.5]
            norm = BoundaryNorm(bounds, cmap_obj.N)
            im = ax.imshow(masked, cmap=cmap_obj, norm=norm)
        else:
            im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label(cbar_label)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)
    else:
        ax.text(0.5, 0.5, "No valid raster cells", ha="center", va="center", fontsize=12)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =========================================================================================
# ====================================== 5.查找特定变量相关文件的函数=======================================
# =========================================================================================
def find_variable_files(var_name_to_match, specific_variable_folder_path):
    matched = {}  # 用于存储匹配到的年份及其对应的文件路径
    # 正则表达式，用于匹配文件名格式：年份 + 变量名 + ".tif"或".tiff"
    pattern = re.compile(r"(\d{4})" + re.escape(var_name_to_match) + r"\.(tif|tiff)", re.IGNORECASE)
    for root, _, files in os.walk(specific_variable_folder_path):  # 遍历指定文件夹及其所有子文件夹
        for file in files:  # 遍历当前文件夹下的所有文件
            m = pattern.fullmatch(file)  # 匹配当前文件名
            if m:  # 如果文件名与模式完全匹配
                year = m.group(1)  # 提取正则表达式捕获的第一个组，即四位数字年份
                matched[year] = os.path.join(root, file)  # 将年份和对应的完整文件路径存入字典
    return matched  # 返回包含年份和文件路径的字典


# =========================================================================================
# =========================6.定义加载变量栅格数据堆栈并进行过滤的函数=======================================
# =========================================================================================
def load_variable_stack(var_name, paths_config, range_dict):
    if var_name not in paths_config:  # 检查变量名是否存在于路径配置字典中
        raise FileNotFoundError(f"未配置变量 '{var_name}' 路径")
    files_map = find_variable_files(
        var_name, paths_config[var_name]
    )  # 查找该变量对应的所有栅格文件
    if not files_map:  # 如果没有找到文件
        raise FileNotFoundError(
            f"在路径 '{paths_config[var_name]}' 中没有找到变量 '{var_name}' 的文件。"
        )
    dates = sorted(files_map.keys())  # 获取所有文件的年份并进行排序
    stack = []  # 用于存储读取的各年份栅格数据数组
    meta0 = None  # 用于存储第一个读取的栅格文件的元数据作为参考，用于数据结果使用
    vmin, vmax = range_dict[var_name]  # 获取该变量的有效值范围 (最小值, 最大值)
    for d in tqdm(dates, desc=f"加载 {var_name}"):  # 遍历排序后的年份，使用tqdm显示加载进度
        arr, meta = read_raster(files_map[d])  # 读取对应年份的栅格数据和元数据
        invalid = (arr < vmin) | (arr > vmax)  # 创建一个布尔掩码，标记出超出有效值范围的像元
        arr[invalid] = np.nan  # 将无效像元的值设为NaN
        if meta0 is None:  # 如果是第一次读取文件
            meta0 = meta  # 保存当前文件的元数据作为参考元数据
        stack.append(arr)  # 将处理后的数据数组添加到堆栈列表中
    if not stack:  # 如果没有加载到任何数据
        raise ValueError(f"'{var_name}' 没有加载到任何数据")  # 抛出值错误
    return (
        np.stack(stack),
        dates,
        meta0,
    )  # 将列表中的所有数组堆叠成一个NumPy数组，并返回该数组、年份列表和参考元数据


# =========================================================================================
# =========================7. 提取窗口数据的函数=======================================
# =========================================================================================
def extract_window(data, row, col, window_radius, padding_value_unused):
    height, width = data.shape  # 获取栅格数据的高度和宽度
    # 计算窗口的边界
    row_start = max(0, row - window_radius)  # 窗口起始行索引
    row_end = min(height, row + window_radius + 1)  # 窗口结束行索引
    col_start = max(0, col - window_radius)  # 窗口起始列索引
    col_end = min(width, col + window_radius + 1)  # 窗口结束列索引
    # 提取窗口区域的数据
    window = data[row_start:row_end, col_start:col_end]
    return window  # 返回提取的窗口数据


# =========================================================================================
# =========================8. 定义处理栅格数据中单行的函数======================================
# =========================================================================================
def process_row(args):
    (
        i,  # 当前正在处理的行索引
        H,  # 栅格数据的总高度，行数
        W,  # 栅格数据的总宽度，列数
        target,  # 目标变量的名称
        feats,  # 特征变量名称的列表
        data_dict,  # 所有栅格数据
        min_samples_for_window_model,  # 最少有效样本数
        model_params_dict,  # 随机森林的基础固定参数
        feature_labels_param,  # 特征名到数字标签的映射字典
        param_grid,  # 超参数范围
        current_window_radius,  # 滑动窗口的半径
        current_padding_value,  # 边缘填充值
        min_valid_points_for_center_target_ts,  # 中心像素点在时间序列上所需的最少有效观测点数
    ) = args  # 解包赋值

    y_stack = data_dict[target]  # 提取目标变量数据堆栈
    X_stacks = {f: data_dict[f] for f in feats}  # 提取所有特征变量数据堆栈

    r2_r = np.full(W, np.nan, dtype=np.float32)  # 当前行R2分数的数组
    rmse_r = np.full(W, np.nan, dtype=np.float32)  # 当前行RMSE的数组

    shap_values_r = {
        f: np.full(W, np.nan, dtype=np.float32) for f in feats
    }  # 存储当前行各特征SHAP值的字典
    shap_values_abs_r = {
        f: np.full(W, np.nan, dtype=np.float32) for f in feats
    }  # 存储当前行各特征绝对SHAP值的字典

    max_shap_feature_label_r = np.full(W, np.nan, dtype=np.float32)  # 存储最大SHAP特征标签的数组

    model_data = []  # 收集当前行的模型参数和性能指标

    for j in range(W):  # 遍历当前行中的每一列
        y_center_pixel_timeseries = y_stack[:, i, j]  # 提取中心像元处目标变量的时间序列数据
        if np.sum(~np.isnan(y_center_pixel_timeseries)) < min_valid_points_for_center_target_ts:
            # 如果有效数据点数量不足，则跳过此像元
            continue  # 继续处理当前行的下一个像元
        # 提取以(i,j)为中心，在每个时间步t上的空间窗口数据，然后展平
        y_pixel_windows_at_t = [
            extract_window(
                y_stack[t, :, :], i, j, current_window_radius, current_padding_value
            ).flatten()
            for t in range(y_stack.shape[0])
        ]

        X_pixel_windows_at_t_by_feature = {}  # 用于存储各特征变量的窗口数据
        for f_name in feats:  # 遍历每个特征变量
            X_stack_f = X_stacks[f_name]  # 获取当前特征变量的数据堆栈
            X_pixel_windows_at_t_by_feature[f_name] = [  # 对当前特征提取时间序列窗口数据
                extract_window(
                    X_stack_f[t, :, :], i, j, current_window_radius, current_padding_value
                ).flatten()
                for t in range(X_stack_f.shape[0])
            ]
        y_pixel = np.array(
            y_pixel_windows_at_t
        ).flatten()  # 将目标变量窗口数据列表转换为单个一维数组

        X_pixel_list_flattened_time_series = []  # 初始化列表用于存储展平后的各特征时间序列窗口数据
        for f_name in feats:  # 遍历每个特征
            # 将每个特征的时间序列窗口数据转换为二维数组再展平为一维数组
            X_pixel_list_flattened_time_series.append(
                np.array(X_pixel_windows_at_t_by_feature[f_name]).flatten()
            )

        if not X_pixel_list_flattened_time_series:  # 如果没有特征数据
            continue  # 跳过当前像元

        # 将所有特征变量的展平后的一维数组沿列方向堆叠，形成一个二维数组
        X_pixel = np.stack(X_pixel_list_flattened_time_series, axis=1)

        # 创建有效数据掩码：目标变量非NaN且所有特征变量在同一时空点上均非NaN
        valid_mask = ~np.isnan(y_pixel) & ~np.isnan(X_pixel).any(axis=1)

        # 检查展平后的时空窗口数据中有效观测点的数量是否少于模型训练要求的最小样本数
        if valid_mask.sum() < min_samples_for_window_model:
            continue  # 跳过当前像元，处理下一个

        yv = y_pixel[valid_mask]  # 提取有效的目标变量序列
        Xv = X_pixel[valid_mask]  # 提取有效的特征变量序列

        # 如果目标变量或任何一个特征变量在有效序列中是常数，标准差为0
        if np.nanstd(yv) == 0 or any(np.nanstd(Xv[:, k]) == 0 for k in range(Xv.shape[1])):
            continue

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(Xv, yv, test_size=0.3, random_state=0)
        if len(X_train) < 2 or len(X_val) < 1:  # 训练集中至少有2个样本
            continue

        # 进行超参数搜索
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=0, n_jobs=1),
            param_grid,  # 超参数网格
            cv=2,  # 交叉验证的折数
            scoring="r2",  # 评估指标
            n_jobs=1,
        )  # 并行数
        grid_search.fit(X_train, y_train)  # 运行网格搜索

        best_model = grid_search.best_estimator_  # 获取最佳参数组合的模型
        best_params = best_model.get_params()  # 获取最佳模型的参数

        pred = best_model.predict(X_val)  # 在验证集上进行预测
        r2_current = r2_score(y_val, pred)  # 计算R2
        rmse_current = np.sqrt(mean_squared_error(y_val, pred))  # 计算RMSE

        r2_r[j] = r2_current  # 存储R2
        rmse_r[j] = rmse_current  # 存储RMSE

        explainer = shap.TreeExplainer(best_model)  # 创建SHAP的TreeExplainer对象
        # 对所有有效数据点计算 SHAP 值
        shap_values_pixel_all_points = explainer.shap_values(Xv)

        # 计算每个特征在所有有效点上的平均SHAP值和平均绝对SHAP值
        mean_shap_for_pixel_model = np.nanmean(shap_values_pixel_all_points, axis=0)
        abs_shap_values_all_points = np.abs(shap_values_pixel_all_points)
        mean_abs_shap_for_pixel_model = np.nanmean(abs_shap_values_all_points, axis=0)

        # 获取每个特征的平均SHAP值
        feature_shap_values = {
            feats[idx]: mean_shap_for_pixel_model[idx] for idx in range(len(feats))
        }

        for idx, f_name in enumerate(feats):  # 遍历特征变量名称
            if idx < len(mean_shap_for_pixel_model):  # 确保索引在有效范围内
                shap_values_r[f_name][j] = mean_shap_for_pixel_model[idx]  # 存储该特征的平均SHAP值
                shap_values_abs_r[f_name][j] = mean_abs_shap_for_pixel_model[
                    idx
                ]  # 存储平均绝对SHAP值

        # 确定平均绝对SHAP值最大的特征并标记其数字标签
        if not np.all(np.isnan(mean_abs_shap_for_pixel_model)) and np.any(
            mean_abs_shap_for_pixel_model > 0
        ):  # 确保有有效的SHAP值
            max_shap_idx = np.nanargmax(
                mean_abs_shap_for_pixel_model
            )  # 找到绝对SHAP值最大特征的索引
            dominant_feature_name = feats[max_shap_idx]  # 获取该特征的名称
            if dominant_feature_name in feature_labels_param:  # 如果该特征在预定义的标签字典中
                max_shap_feature_label_r[j] = feature_labels_param[
                    dominant_feature_name
                ]  # 存储其数字标签
        # 将当前像元的模型信息和结果添加到列表
        model_data.append(
            {
                "row": i,
                "col": j,
                "n_estimators": best_params.get("n_estimators", np.nan),
                "max_depth": best_params.get("max_depth", np.nan),
                "r2": r2_current,
                "rmse": rmse_current,
                **feature_shap_values,
            }
        )
    return i, r2_r, rmse_r, shap_values_r, shap_values_abs_r, max_shap_feature_label_r, model_data


# =========================================================================================
# =========================9. excel文件保存函数=====================================
# =========================================================================================
def save_to_excel(data, filename):
    output_dir_excel = os.path.dirname(filename)  # 获取文件名所在目录
    if not os.path.exists(output_dir_excel):  # 如果目录不存在
        os.makedirs(output_dir_excel, exist_ok=True)  # 创建目录
    df = pd.DataFrame(data)  # 将数据列表转换为pandas DataFrame
    df.to_excel(filename, index=False, engine="openpyxl")  # 保存到Excel文件


# =========================================================================================
# =========================10主运行函数=====================================
# =========================================================================================
def run_parallel():
    global current_excel_file_index, excel_data  # 声明使用全局变量

    vars_all = [target_var_name] + feature_var_names  # 创建包含目标变量和所有特征变量的列表
    raw_data = {}  # 用于存储每个变量原始加载的栅格数据堆栈
    dates_map = {}  # 用于存储每个变量对应的时间序列（年份）
    meta_ref = None  # 初始化参考元数据变量，用于后续栅格写入

    for v_name in vars_all:  # 遍历所有需要加载的变量
        data, dates, meta = load_variable_stack(
            v_name, variable_folder_paths, valid_ranges
        )  # 加载该变量的数据堆栈
        raw_data[v_name] = data  # 加载的数据
        dates_map[v_name] = dates  # 年份列表
        if v_name == target_var_name:  # 如果当前变量是目标变量
            meta_ref = meta  # 获取目标变量的元数据作为参考元数据

    if meta_ref is None and feature_var_names:  # 如果目标变量没有元数据
        # 尝试从第一个特征变量获取元数据
        _, _, meta_ref = load_variable_stack(
            feature_var_names[0], variable_folder_paths, valid_ranges
        )
    elif meta_ref is None:  # 如果没有任何元数据可以获取
        print("无法获取任何栅格文件的元数据")
        return
    # 查找所有变量共有的日期
    common_dates = set(dates_map[vars_all[0]])  # 从第一个变量的日期开始
    for v_name in vars_all[1:]:  # 与后续变量的日期取交集
        common_dates &= set(dates_map[v_name])
    common_dates_sorted = sorted(list(common_dates))  # 排序得到共有的日期列表

    if not common_dates_sorted:  # 如果没有共同日期
        print("在所有变量之间没有找到共同的日期")
        return

    # 检查共同日期数量是否满足配置中要求的最小时间序列长度
    if len(common_dates_sorted) < min_observations_required:
        print(
            f"共同日期数量 ({len(common_dates_sorted)}) 少于模型要求的最小时间序列长度 ({min_observations_required})"
        )
        return

    # --- 根据共同日期对齐数据 ---
    aligned_data = {}  # 存储对齐后的数据
    for v_name in vars_all:  # 遍历所有变量
        indices = [
            dates_map[v_name].index(d) for d in common_dates_sorted
        ]  # 获取共同日期在当前变量日期列表中的索引
        aligned_data[v_name] = raw_data[v_name][indices]  # 根据索引提取对齐的数据

    T, H, W = aligned_data[target_var_name].shape  # 获取对齐后目标变量数据的维度 (时间, 高度, 宽度)
    meta_ref["height"], meta_ref["width"] = H, W  # 更新参考元数据中的高度和宽度

    # process_row 函数的参数列表
    tasks = [
        (
            r_idx,  # 行索引
            H,  # 行数
            W,  # 列数
            target_var_name,  # 目标变量的名称
            feature_var_names,  # 特征变量名称列表
            aligned_data,  # 所有数据
            max(
                len(feature_var_names) + 2, min_observations_required
            ),  # 窗口模型训练所需的最小样本数
            rf_params,  # 随机森林的固定参数
            feature_numerical_labels,  # 特征名对应的数字标签字典
            param_grid,  # 超参数范围
            WINDOW_SIZE,  # 滑动窗口的半径
            0,  # 边缘填充值
            min_observations_required,  # 中心像元时间序列所需的最小有效观测点数
        )
        for r_idx in range(H)
    ]

    num_workers = min(
        multiprocessing.cpu_count(), H, 8 if H > 16 else H if H > 0 else 1
    )  # 确定并行工作进程的数量
    print(f"使用 {num_workers} 个进程并行计算...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:  # 创建进程池执行器
        # 使用 executor.map 并行处理所有任务
        results_list = list(
            tqdm(executor.map(process_row, tasks), total=H, desc="逐行随机森林、SHAP与最大特征分析")
        )

    # 初始化用于存储整张图结果的Numpy数组
    r2_map = np.full((H, W), np.nan, dtype=np.float32)  # R2图
    rmse_map = np.full((H, W), np.nan, dtype=np.float32)  # RMSE图
    shap_maps = {
        f: np.full((H, W), np.nan, dtype=np.float32) for f in feature_var_names
    }  # 每个特征的SHAP值图
    shap_maps_abs = {
        f: np.full((H, W), np.nan, dtype=np.float32) for f in feature_var_names
    }  # 每个特征的绝对SHAP值图
    max_shap_label_map = np.full((H, W), np.nan, dtype=np.float32)  # 最大SHAP特征标签图

    for (
        r_idx_res,
        r2_row_data,
        rmse_row_data,
        shap_row_data,
        shap_row_data_abs,
        max_shap_label_row_data,
        model_data_list_for_row,
    ) in results_list:
        if r_idx_res is None:  # 如果某一行处理彻底失败
            continue
        r2_map[r_idx_res, :] = r2_row_data  # 将当前行的R2数据填入R2图
        rmse_map[r_idx_res, :] = rmse_row_data  # 将当前行的RMSE数据填入RMSE图
        for f_name in feature_var_names:  # 遍历特征名称
            if f_name in shap_row_data:  # 检查该特征的SHAP值是否存在于结果中
                shap_maps[f_name][r_idx_res, :] = shap_row_data[f_name]  # 填入SHAP值
            if f_name in shap_row_data_abs:  # 检查该特征的绝对SHAP值是否存在
                shap_maps_abs[f_name][r_idx_res, :] = shap_row_data_abs[f_name]  # 填入绝对SHAP值
        max_shap_label_map[r_idx_res, :] = (
            max_shap_label_row_data  # 将当前行的最大SHAP特征标签数据填入对应图
        )
        excel_data.extend(
            model_data_list_for_row
        )  # 将当前行处理的所有像元的模型数据追加到excel_data列表

        # 如果数据行数达到每个Excel文件的最大行数
        while len(excel_data) >= MAX_ROWS_PER_EXCEL:
            excel_filename_part = os.path.join(
                excel_output_dir, f"model_data_{current_excel_file_index}.xlsx"
            )  # 构建文件名
            save_to_excel(
                excel_data[:MAX_ROWS_PER_EXCEL], excel_filename_part
            )  # 保存当前批次的数据
            excel_data = excel_data[MAX_ROWS_PER_EXCEL:]  # 移除已保存的数据
            current_excel_file_index += 1  # 文件索引加1

    if excel_data:  # 如果处理完所有行后仍有剩余数据
        excel_filename_final = os.path.join(
            excel_output_dir, f"model_data_{current_excel_file_index}.xlsx"
        )  # 构建最终文件名
        save_to_excel(excel_data, excel_filename_final)  # 保存剩余数据

    # 保存栅格结果
    write_raster(
        os.path.join(output_dir, f"{target_var_name}_RandomForest_r2.tif"), r2_map, meta_ref
    )
    write_raster(
        os.path.join(output_dir, f"{target_var_name}_RandomForest_rmse.tif"), rmse_map, meta_ref
    )

    for f_name in feature_var_names:  # 遍历每个特征
        # 保存该特征的SHAP值图
        write_raster(
            os.path.join(output_dir, f"{target_var_name}_RandomForest_shap_{f_name}.tif"),
            shap_maps[f_name],
            meta_ref,
        )
        # 保存该特征的绝对SHAP值图
        write_raster(
            os.path.join(output_dir, f"{target_var_name}_RandomForest_shap_abs_{f_name}.tif"),
            shap_maps_abs[f_name],
            meta_ref,
        )
    max_shap_label_filename = (
        f"{target_var_name}_RandomForest_max_shap_feature_label.tif"  # 最大SHAP特征标签图文件名
    )
    write_raster(
        os.path.join(output_dir, max_shap_label_filename), max_shap_label_map, meta_ref
    )  # 保存
    label_items = sorted(feature_numerical_labels.items(), key=lambda item: item[1])
    save_array_preview(
        max_shap_label_map,
        os.path.join(output_dir, "preview.png"),
        "Dominant SHAP feature map",
        cmap="tab10",
        vmin=0.5,
        vmax=max(feature_numerical_labels.values()) + 0.5,
        cbar_label="dominant feature",
        cbar_ticks=[label for _, label in label_items],
        cbar_ticklabels=[name for name, _ in label_items],
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # 防止冻结问题
    run_parallel()  # 调用主运行函数

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

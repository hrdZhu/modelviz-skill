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

import logging  # 导入logging模块，用于记录程序运行时的信息
import multiprocessing  # 导入multiprocessing模块，用于支持多进程并行计算
import os  # 导入os模块，用于处理文件和目录路径
import warnings  # 导入warnings模块，用于处理警告信息

import numpy as np  # 导入numpy库，用于进行数值计算，特别是数组操作
import pandas as pd  # 导入pandas库，用于数据分析和处理，特别是DataFrame操作
import pingouin as pg  # 导入pingouin库，用于统计分析，如偏相关计算
from joblib import Parallel, delayed  # 从joblib库导入Parallel和delayed，用于并行处理
from osgeo import gdal  # 导入gdal库，用于处理地理空间数据
from tqdm import tqdm  # 导入tqdm库，用于显示进度条

# 定义日志文件名
log_filename = str(OUTPUT_DIR / "partial_corr_analysis_final.log")
# 配置日志记录器
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,  # 配置日志记录的基本设置，filename指定日志文件名，level指定记录的最低级别为INFO
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",  # 日志格式字符串，定义了日志条目的显示方式
    filemode="w",
)  # 文件写入模式：覆盖，'w'表示每次运行都会覆盖旧的日志文件


def find_min_shape_and_geotransform(
    image_files_list,
):  # 定义函数，用于找到所有输入影像的最小行列数，并获取第一个有效影像的地理变换和投影信息
    """
    找到所有输入影像的最小行列数，并获取第一个有效影像的地理变换和投影信息。
    """
    min_rows, min_cols = float("inf"), float("inf")  # 初始化最小行数和列数为正无穷大
    geotransform, projection = None, None  # 初始化地理变换和投影信息为None
    if not image_files_list:  # 检查影像文件列表是否为空
        logging.error("影像文件列表为空。")  # 如果为空，记录错误日志
        raise ValueError("影像文件列表为空。")  # 并抛出ValueError异常
    first_valid_ds_found = False  # 初始化标志，表示是否已找到第一个有效的GDAL数据集
    for (
        files
    ) in image_files_list:  # 遍历影像文件列表中的每个子列表（每个子列表代表一个变量的影像序列）
        if not files:  # 检查子列表是否为空
            logging.warning("一个影像文件子列表为空，已跳过。")  # 如果为空，记录警告日志并跳过
            continue  # 继续下一个子列表
        for f in files:  # 遍历子列表中的每个文件路径
            try:  # 尝试打开和处理影像文件
                ds = gdal.Open(f)  # 使用gdal打开影像文件
                if ds is None:  # 检查影像是否成功打开
                    logging.error(f"无法打开影像文件: {f}")  # 如果打开失败，记录错误日志
                    continue  # 继续下一个文件
                rows, cols = ds.RasterYSize, ds.RasterXSize  # 获取影像的行数和列数
                min_rows = min(min_rows, rows)  # 更新最小行数
                min_cols = min(min_cols, cols)  # 更新最小列数
                if not first_valid_ds_found:  # 如果还没有找到有效的地理变换和投影信息
                    current_geotransform = ds.GetGeoTransform()  # 获取当前影像的地理变换参数
                    current_projection = ds.GetProjection()  # 获取当前影像的投影信息
                    if current_geotransform and current_projection:  # 如果两者都成功获取
                        geotransform = current_geotransform  # 将当前地理变换赋给全局变量
                        projection = current_projection  # 将当前投影信息赋给全局变量
                        first_valid_ds_found = True  # 设置标志为已找到
                ds = None  # 关闭GDAL数据集，释放资源
            except Exception as e:  # 捕获处理文件时可能发生的任何异常
                logging.error(f"处理文件 {f} 时出错: {e}")  # 记录错误日志
                continue  # 继续下一个文件
    if min_rows == float("inf") or min_cols == float("inf"):  # 检查是否成功确定了最小尺寸
        logging.error("未能从任何影像文件中确定最小尺寸。")  # 如果没有，记录错误日志
        raise ValueError("未能从任何影像文件中确定最小尺寸。")  # 并抛出ValueError异常
    if geotransform is None or projection is None:  # 检查是否成功获取了地理变换或投影参数
        logging.error(
            "未能从任何影像文件中获取有效的地理变换或投影参数。"
        )  # 如果没有，记录错误日志
        raise ValueError(
            "未能从任何影像文件中获取有效的地理变换或投影参数。"
        )  # 并抛出ValueError异常
    return (
        min_rows,
        min_cols,
        geotransform,
        projection,
    )  # 返回最小行数、最小列数、地理变换和投影信息


def apply_value_range_filter(
    data_stack, valid_min, valid_max, variable_name=""
):  # 定义函数，用于将指定范围之外的值替换为np.nan
    """
    将指定范围之外的值替换为np.nan。
    """
    if data_stack is None or data_stack.size == 0:  # 检查输入数据栈是否为None或为空
        return data_stack  # 如果是，直接返回
    original_nan_count = np.isnan(data_stack).sum()  # 计算过滤前NaN值的数量
    filtered_stack = data_stack.copy()  # 创建数据栈的副本进行修改，避免修改原始数据
    if valid_min is not None:  # 如果指定了有效最小值
        condition_min = filtered_stack < valid_min  # 创建一个布尔掩码，标记小于最小值的值
        filtered_stack[condition_min] = np.nan  # 将这些值设为NaN
        logging.info(
            f"变量 {variable_name}: {np.sum(condition_min)} 个值小于允许的最小值 {valid_min}，已设为NaN。"
        )  # 记录日志信息
    if valid_max is not None:  # 如果指定了有效最大值
        condition_max = filtered_stack > valid_max  # 创建一个布尔掩码，标记大于最大值的值
        filtered_stack[condition_max] = np.nan  # 将这些值设为NaN
        logging.info(
            f"变量 {variable_name}: {np.sum(condition_max)} 个值大于允许的最大值 {valid_max}，已设为NaN。"
        )  # 记录日志信息
    final_nan_count = np.isnan(filtered_stack).sum()  # 计算过滤后NaN值的数量
    logging.info(
        f"变量 {variable_name}: 值范围过滤后，新增NaN数量: {final_nan_count - original_nan_count}"
    )  # 记录新增NaN的数量
    return filtered_stack  # 返回过滤后的数据栈


def save_array_preview(
    array,
    path,
    title,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    cbar_label="partial r",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preview = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(preview)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    if valid.any():
        masked = np.ma.masked_invalid(preview)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label(cbar_label)
    else:
        ax.text(0.5, 0.5, "No valid raster cells", ha="center", va="center", fontsize=12)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def choose_valid_read_window(raster_path, rows, cols, full_rows, full_cols, valid_range):
    ds = gdal.Open(raster_path)
    if ds is None:
        return 0, 0

    max_row_offset = max(0, full_rows - rows)
    max_col_offset = max(0, full_cols - cols)
    row_candidates = np.linspace(0, max_row_offset, num=min(7, max_row_offset + 1), dtype=int)
    col_candidates = np.linspace(0, max_col_offset, num=min(7, max_col_offset + 1), dtype=int)
    candidates = {(int(r), int(c)) for r in row_candidates for c in col_candidates}
    candidates.add((max_row_offset // 2, max_col_offset // 2))

    valid_min, valid_max = valid_range
    best_score = (-1, -1.0)
    best_offset = (0, 0)
    for row_offset, col_offset in candidates:
        arr = ds.GetRasterBand(1).ReadAsArray(col_offset, row_offset, cols, rows)
        if arr is None:
            continue
        arr = arr.astype(np.float32)
        valid = np.isfinite(arr)
        if valid_min is not None:
            valid &= arr >= valid_min
        if valid_max is not None:
            valid &= arr <= valid_max
        valid_count = int(valid.sum())
        spread = float(np.nanstd(arr[valid])) if valid_count else 0.0
        score = (valid_count, spread)
        if score > best_score:
            best_score = score
            best_offset = (int(row_offset), int(col_offset))

    ds = None
    return best_offset


def shift_geotransform(geotransform, row_offset, col_offset):
    gt = list(geotransform)
    gt[0] = gt[0] + col_offset * gt[1] + row_offset * gt[2]
    gt[3] = gt[3] + col_offset * gt[4] + row_offset * gt[5]
    return tuple(gt)


def process_pixel_windowed(
    center_i, center_j, X_stacks, Y_stack, num_variables, window_size, img_rows, img_cols
):  # 定义函数，处理单个中心像元周围窗口的数据以计算偏相关
    """
    处理单个中心像元，使用其周围窗口的数据计算偏相关系数和 p-value。
    如果中心像元的Y值在其整个时间序列上都为NaN，或者所有非NaN值都相同，则跳过处理。
    """
    partial_corr_row = np.full(
        num_variables, np.nan, dtype=np.float32
    )  # 初始化存储偏相关系数的数组，填充为NaN
    pvalue_row = np.full(num_variables, np.nan, dtype=np.float32)  # 初始化存储p值的数组，填充为NaN

    center_y_timeseries = Y_stack[:, center_i, center_j]  # 提取中心像元Y的时间序列数据

    # 核心判断条件：如果中心像元Y的时间序列中的所有值都是NaN，或者所有非NaN值都是相同的数值，则跳过。
    non_nan_y_values = center_y_timeseries[
        ~np.isnan(center_y_timeseries)
    ]  # 获取Y时间序列中所有非NaN的值
    if (
        len(np.unique(non_nan_y_values)) <= 1
    ):  # 如果非NaN值的唯一值数量小于等于1（即全为NaN或所有非NaN值相同）
        return None  # 则跳过该像元的处理，返回None
    if len(non_nan_y_values) < 20:  # 如果中心Y时间序列的有效值（非NaN）数量小于20
        # logging.debug(f"像元 ({center_i}, {center_j}): 中心Y时间序列有效值数量 {len(non_nan_y_values)} < 20，跳过。") # 可选的调试日志，记录跳过原因
        return None  # 则跳过该像元的处理，返回None

    half_window = window_size // 2  # 计算窗口半径 (整数除法)
    r_start, r_end = (
        center_i - half_window,
        center_i + half_window + 1,
    )  # 计算窗口的起始和结束行索引
    c_start, c_end = (
        center_j - half_window,
        center_j + half_window + 1,
    )  # 计算窗口的起始和结束列索引

    collected_rows_for_df = []  # 初始化一个列表，用于收集构建DataFrame的行数据
    num_time_steps = Y_stack.shape[0]  # 获取时间序列的长度 (时间步数)

    for t in range(num_time_steps):  # 遍历每个时间步
        for r_idx in range(r_start, r_end):  # 遍历窗口内的每一行
            for c_idx in range(c_start, c_end):  # 遍历窗口内的每一列
                if (
                    r_idx < 0 or r_idx >= img_rows or c_idx < 0 or c_idx >= img_cols
                ):  # 检查当前像元是否在影像边界之外
                    y_value = np.nan  # 如果在边界外，Y值设为NaN
                    x_values = [np.nan] * num_variables  # X变量值也设为NaN列表
                else:  # 如果在边界内
                    y_value = Y_stack[t, r_idx, c_idx]  # 获取当前时间步、当前窗口位置的Y值
                    x_values = [
                        X_stacks[k][t, r_idx, c_idx] for k in range(num_variables)
                    ]  # 获取所有X变量在当前时间步、当前窗口位置的值
                current_data_point = {"Y": y_value}  # 创建一个字典存储当前数据点，首先放入Y值
                for k in range(num_variables):  # 遍历所有X变量
                    current_data_point[f"X{k + 1}"] = x_values[
                        k
                    ]  # 将X变量的值放入字典，键名为 'X1', 'X2', ...
                collected_rows_for_df.append(current_data_point)  # 将当前数据点字典添加到列表中

    if (
        not collected_rows_for_df
    ):  # 如果没有收集到任何数据行 (理论上不太可能发生，除非窗口大小为0或时间步为0)
        return None  # 返回None

    df = pd.DataFrame(collected_rows_for_df)  # 使用收集到的数据行创建pandas DataFrame
    df.dropna(
        how="any", inplace=True
    )  # 删除DataFrame中任何包含NaN值的行 (how='any'表示只要有一个NaN就删除整行)，inplace=True表示直接修改原DataFrame

    # 理论上偏相关至少需要的样本数 (变量数+Y+1) 再加1，pingouin文档建议n > p (p为变量总数)
    min_samples_needed_stats = (
        num_variables + 2
    )  # 计算统计学上偏相关所需的最小样本数 (Y变量, 当前X变量, 加上其余作为协变量的X变量，至少需要这么多列 + 1个观测)
    min_threshold_empirical = max(
        min_samples_needed_stats, 10
    )  # 设定一个经验性的最小样本阈值，取理论值和10之间的较大者

    if df.shape[0] < min_threshold_empirical:  # 如果有效样本数（删除NaN行后）小于经验阈值
        return None  # 则跳过该像元的处理，返回None

    # 检查是否有任何列的标准差为0 (即所有值都相同)，这会导致相关性计算问题
    if (
        df.std(ddof=0) == 0
    ).any():  # 计算每一列的标准差(ddof=0表示总体标准差)，如果任何一列的标准差为0
        return None  # 则跳过该像元的处理，返回None

    # 协方差矩阵和条件数检查
    try:  # 尝试计算协方差矩阵并检查其条件数
        cols_for_cov_matrix = ["Y"] + [
            f"X{k + 1}" for k in range(num_variables)
        ]  # 构建用于计算协方差矩阵的列名列表
        actual_cols_in_df = [
            col for col in cols_for_cov_matrix if col in df.columns
        ]  # 确保这些列实际存在于DataFrame中 (可能因全NaN或std=0被移除)
        if len(actual_cols_in_df) < 2:  # 如果实际存在的列少于2个，无法计算协方差/相关性
            return None  # 返回None
        cov_matrix_df = df[actual_cols_in_df].cov()  # 计算这些列的协方差矩阵
        if (
            cov_matrix_df.isnull().values.any()
            or cov_matrix_df.shape[0] != len(actual_cols_in_df)
            or cov_matrix_df.shape[1] != len(actual_cols_in_df)
        ):  # 如果协方差矩阵计算失败，返回NaN，或形状不正确
            return None  # 返回None
        if (
            cov_matrix_df.shape[0] >= 2 and cov_matrix_df.shape[1] >= 2
        ):  # 仅当矩阵至少为2x2时计算条件数
            cond_number = np.linalg.cond(cov_matrix_df.values)  # 计算协方差矩阵的条件数
            if np.isinf(cond_number) or cond_number > 1e12:  # 如果条件数为无穷大或非常大 (矩阵病态)
                return None  # 返回None
        elif len(actual_cols_in_df) < 2:  # 再次确认，理论上已被前一个if覆盖 (如果列数少于2)
            return None  # 返回None
    except (
        np.linalg.LinAlgError,
        ValueError,
    ):  # 捕获计算协方差矩阵或条件数时可能发生的线性代数错误或值错误
        return None  # 返回None

    for k in range(num_variables):  # 遍历每一个自变量X
        current_x_var = f"X{k + 1}"  # 当前要计算偏相关的X变量名
        if (
            current_x_var not in df.columns
        ):  # 如果当前X变量因为之前的处理（如全是NaN或标准差为0）而已不在DataFrame中
            partial_corr_row[k] = np.nan  # 将该变量的偏相关系数设为NaN
            pvalue_row[k] = np.nan  # 将该变量的p值设为NaN
            continue  # 继续处理下一个X变量

        # 确定协变量列表 (除了当前X变量之外的其他X变量)
        covariates = [
            f"X{j + 1}" for j in range(num_variables) if j != k and f"X{j + 1}" in df.columns
        ]  # 构建协变量列表，即其他存在于DataFrame中的X变量

        if "Y" not in df.columns:  # Y变量不应在此阶段丢失，但作为安全检查
            partial_corr_row[k] = np.nan  # 如果Y变量丢失，则该X变量的偏相关系数设为NaN
            pvalue_row[k] = np.nan  # p值也设为NaN
            continue  # 继续下一个X变量

        r, pval = np.nan, np.nan  # 初始化当前X变量的偏相关系数r和p值为NaN
        try:  # 尝试计算相关性
            if not covariates:  # 如果没有协变量 (即只有一个X变量或者其他X变量都无效)
                if (
                    len(df[current_x_var]) >= 2 and len(df["Y"]) >= 2
                ):  # 确保当前X变量和Y变量都有至少2个数据点才能计算相关性
                    result = pg.corr(x=df[current_x_var], y=df["Y"])  # 计算简单皮尔逊相关系数
                    r = result["r"].iloc[0]  # 提取相关系数r
                    pval = result["p-val"].iloc[0]  # 提取p值
            else:  # 如果有协变量
                # 检查协变量数据是否有效（不为空且标准差不为零）
                if not (
                    df[covariates].empty or (df[covariates].std(ddof=0) == 0).any()
                ):  # 确保协变量DataFrame不为空，并且所有协变量的标准差都不为0
                    with (
                        warnings.catch_warnings()
                    ):  # 使用warnings模块捕获并暂时忽略pingouin可能产生的警告 (如UserWarning)
                        warnings.simplefilter("ignore")  # 设置警告过滤器为"ignore"
                        # 为偏相关准备数据，确保选取的列都存在且再次去除NaN
                        analysis_cols = [
                            current_x_var,
                            "Y",
                        ] + covariates  # 包含当前X, Y, 和所有协变量的列名列表
                        temp_df_for_pcorr = df[
                            analysis_cols
                        ].dropna()  # 从原始DataFrame中选取这些列，并再次删除包含NaN的行
                        # 偏相关需要的样本数 > 变量数 (X, Y, 加上所有协变量)
                        if temp_df_for_pcorr.shape[0] > len(
                            analysis_cols
                        ):  # 确保用于偏相关计算的样本数严格大于变量总数 (这是pingouin的要求)
                            result = pg.partial_corr(
                                data=temp_df_for_pcorr, x=current_x_var, y="Y", covar=covariates
                            )  # 计算偏相关系数
                            r = result["r"].iloc[0]  # 提取偏相关系数r
                            pval = result["p-val"].iloc[0]  # 提取p值
        except (np.linalg.LinAlgError, ValueError, AttributeError, KeyError):
            # logging.warning(f"像元 ({center_i}, {center_j}): 变量 {current_x_var} 相关性计算出错: {e}") # 可选的警告日志，记录相关性计算错误信息
            pass  # 如果发生错误，r 和 pval 将保持为 np.nan

        # 对r和pval进行最终检查和清理
        if not np.isfinite(r):  # 检查r是否为有限数值 (不是NaN或inf)
            r = np.nan  # 如果不是，则设为NaN
        else:  # 如果是有限数值
            r = np.clip(
                r, -1.0, 1.0
            )  # 将r的值限制在[-1.0, 1.0]的范围内，防止因浮点精度问题超出理论范围

        if not np.isfinite(pval):  # 检查pval是否为有限数值
            pval = np.nan  # 如果不是，则设为NaN

        partial_corr_row[k] = r  # 将计算得到的r值存入结果数组
        pvalue_row[k] = pval  # 将计算得到的pval值存入结果数组

    return partial_corr_row, pvalue_row  # 返回该中心像元所有X变量的偏相关系数和p值数组


def main():  # 定义主函数
    # ***** 配置参数 *****
    FULL_ANALYSIS = _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1"
    WINDOW_SIZE = 3  # 定义滑动窗口的大小 (例如5x5) - 移到函数顶部方便修改
    logging.info(
        f"----- 开始执行偏相关分析 (窗口大小: {WINDOW_SIZE}x{WINDOW_SIZE}) -----"
    )  # 记录程序开始执行的日志信息

    X_folders = [  # 定义存储自变量影像数据的文件夹路径列表
        str(DATA_DIR / "HSL"),
        str(DATA_DIR / "JS"),
        str(DATA_DIR / "NIR"),
        # r'E:\YZQ\shiyan\5.19\qw', # 以下是被注释掉的文件夹路径示例
        # r'E:\YZQ\shiyan\5.19\rz',
        # r'E:\YZQ\shiyan\5.19\SOF',
        # r'E:\YZQ\shiyan\5.19\SOT-180',
        # r'E:\YZQ\shiyan\5.19\XSDL',
    ]
    Y_folder = str(DATA_DIR / "GST")
    output_folder = str(OUTPUT_DIR)  # 定义存储输出结果的文件夹路径
    os.makedirs(
        output_folder, exist_ok=True
    )  # 创建输出文件夹，如果文件夹已存在则不报错 (exist_ok=True)
    variable_value_ranges = {  # 定义每个变量的有效值范围，用于数据预处理过滤
        # 'ST': (-50, 50), # 示例：ST变量的有效值范围是-50到50
        # 'TRHSL': (0, 50),
        # 'js': (0, 500),
        # 'qw': (-50, 50),
        # 'rz': (0, 500),
        # 'SOF': (180, 366),
        # 'SOT-180': (1, 180),
        # 'XSDL': (0, 200),
        "HSL": (0, 100),  # HSL变量的有效值范围是0到100
        "JS": (0, 100),  # JS变量的有效值范围是0到100
        "NIR": (0, 1),  # NIR变量的有效值范围是0到1
        "GST": (-50, 50),  # GST变量（因变量）的有效值范围是-50到50
    }
    # *******************

    X_files_list = []  # 初始化一个列表，用于存储每个自变量文件夹中的文件路径列表
    x_folder_basenames = []  # 初始化一个列表，用于存储每个自变量文件夹的基本名称 (用于后续命名和日志)
    for folder in X_folders:  # 遍历每个自变量文件夹路径
        try:  # 尝试读取文件夹内容
            files = sorted(
                [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".tif")]
            )  # 获取文件夹下所有以.tif结尾的文件，并按名称排序
            if not files:
                raise ValueError(
                    f"文件夹 {folder} 中没有找到 .tif 文件。"
                )  # 如果没有找到.tif文件，则抛出ValueError
            X_files_list.append(files)  # 将找到的文件路径列表添加到X_files_list中
            x_folder_basenames.append(
                os.path.basename(folder)
            )  # 将文件夹的基本名称添加到x_folder_basenames中
        except Exception as e:  # 捕获读取文件夹时可能发生的任何异常
            logging.error(
                f"读取文件夹 {folder} 失败: {e}", exc_info=True
            )  # 记录错误日志，exc_info=True会记录堆栈跟踪信息
            print(
                f"错误: 读取文件夹 {folder} 失败。详情见日志 {log_filename}"
            )  # 在控制台打印错误信息
            return  # 终止程序执行

    try:  # 尝试读取因变量文件夹内容
        Y_files = sorted(
            [os.path.join(Y_folder, f) for f in os.listdir(Y_folder) if f.endswith(".tif")]
        )  # 获取因变量文件夹下所有以.tif结尾的文件，并按名称排序
        if not Y_files:
            raise ValueError(
                f"文件夹 {Y_folder} 中没有找到 .tif 文件。"
            )  # 如果没有找到.tif文件，则抛出ValueError
        y_folder_basename = os.path.basename(Y_folder)  # 获取因变量文件夹的基本名称
    except Exception as e:  # 捕获读取文件夹时可能发生的任何异常
        logging.error(f"读取文件夹 {Y_folder} 失败: {e}", exc_info=True)  # 记录错误日志
        print(
            f"错误: 读取文件夹 {Y_folder} 失败。详情见日志 {log_filename}"
        )  # 在控制台打印错误信息
        return  # 终止程序执行

    num_y_files = len(Y_files)  # 获取因变量影像文件的数量
    for i, X_files in enumerate(X_files_list):  # 遍历每个自变量的文件列表
        if len(X_files) != num_y_files:  # 检查当前自变量组的文件数量是否与因变量的文件数量相同
            error_msg = (
                f"自变量组 {X_folders[i]} ({len(X_files)}个文件) "  # 构建错误信息字符串
                f"与因变量 {Y_folder} ({num_y_files}个文件) 的影像数量不匹配。"
            )
            logging.error(error_msg)
            print(f"错误: {error_msg}")
            return  # 记录错误日志，打印错误信息，并终止程序
    if num_y_files == 0:  # 检查因变量影像列表是否为空
        logging.error("因变量影像列表为空。")
        print("错误: 因变量影像列表为空。")
        return  # 记录错误日志，打印错误信息，并终止程序

    logging.info("确定影像的最小范围和地理信息...")  # 记录日志信息
    try:  # 尝试调用find_min_shape_and_geotransform函数
        min_rows, min_cols, geotransform, projection = find_min_shape_and_geotransform(
            X_files_list + [Y_files]
        )  # 将所有自变量文件列表和因变量文件列表合并后传入
    except ValueError as e:  # 捕获可能由该函数抛出的ValueError
        logging.error(f"确定影像范围和地理信息时出错: {e}", exc_info=True)
        print(f"错误: {e}")
        return  # 记录错误日志，打印错误信息，并终止程序
    logging.info(
        f"所有影像将按最小范围读取：{min_rows}行, {min_cols}列"
    )  # 记录最终确定的影像读取范围
    row_offset = 0
    col_offset = 0
    if not FULL_ANALYSIS:
        full_rows, full_cols = min_rows, min_cols
        min_rows = min(full_rows, 40)
        min_cols = min(full_cols, 40)
        row_offset, col_offset = choose_valid_read_window(
            Y_files[0],
            min_rows,
            min_cols,
            full_rows,
            full_cols,
            variable_value_ranges.get(y_folder_basename, (None, None)),
        )
        geotransform = shift_geotransform(geotransform, row_offset, col_offset)
        logging.info(
            f"默认示例裁剪为：{min_rows}行, {min_cols}列，读取偏移 row={row_offset}, col={col_offset}"
        )

    num_variables = len(X_folders)  # 获取自变量的数量 (即X文件夹的数量)
    partial_correlation_array = np.full(
        (num_variables, min_rows, min_cols), np.nan, dtype=np.float32
    )  # 初始化存储所有像元偏相关系数的3D NumPy数组，填充为NaN
    pvalue_array = np.full(
        (num_variables, min_rows, min_cols), np.nan, dtype=np.float32
    )  # 初始化存储所有像元p值的3D NumPy数组，填充为NaN

    logging.info("读取数据并裁剪到最小范围...")  # 记录日志信息
    X_stacks = []  # 初始化一个列表，用于存储每个自变量的时间序列数据栈 (3D NumPy数组)
    for i, X_files in enumerate(
        tqdm(X_files_list, desc="读取自变量影像")
    ):  # 遍历每个自变量的文件列表，使用tqdm显示进度条
        stack = []  # 初始化一个临时列表，用于存储当前自变量的各个时间点的影像数据
        var_basename = x_folder_basenames[i]  # 获取当前自变量的文件夹基本名称
        valid_range = variable_value_ranges.get(
            var_basename, (None, None)
        )  # 从variable_value_ranges字典中获取当前变量的有效值范围，如果未定义则不限制 (None, None)
        logging.info(
            f"自变量 {var_basename}: 使用值范围 {valid_range}"
        )  # 记录当前自变量使用的值范围
        for f_idx, f_path in enumerate(X_files):  # 遍历当前自变量的每个影像文件路径
            try:  # 尝试打开和读取影像
                ds = gdal.Open(f_path)  # 使用gdal打开影像文件
                if ds is None:  # 如果打开失败
                    logging.error(f"无法打开影像文件: {f_path}")  # 记录错误日志
                    stack.append(
                        np.full((min_rows, min_cols), np.nan, dtype=np.float32)
                    )  # 添加一个全为NaN的数组以保持堆栈形状一致性
                    continue  # 继续下一个文件
                array = (
                    ds.GetRasterBand(1)
                    .ReadAsArray(col_offset, row_offset, min_cols, min_rows)
                    .astype(np.float32)
                )  # 读取影像的第一个波段，并裁剪到min_cols, min_rows，转换为float32类型
                ds = None  # 关闭GDAL数据集
                # 应用值范围过滤器
                array_filtered = apply_value_range_filter(
                    array,
                    valid_range[0],
                    valid_range[1],  # 调用函数过滤值
                    f"{var_basename}_file{f_idx}",
                )  # 传入变量名和文件名用于日志
                stack.append(array_filtered)  # 将过滤后的数组添加到临时列表中
            except Exception as e:  # 捕获读取或处理影像时可能发生的任何异常
                logging.error(f"读取或处理影像 {f_path} 失败: {e}", exc_info=True)  # 记录错误日志
                stack.append(
                    np.full((min_rows, min_cols), np.nan, dtype=np.float32)
                )  # 发生错误时添加一个全为NaN的数组
        current_x_stack = np.array(
            stack
        )  # 将临时列表转换为NumPy数组，形成当前自变量的时间序列数据栈
        logging.info(  # 记录当前自变量数据栈的信息
            f"自变量 {var_basename} stack: 形状 {current_x_stack.shape}, NaN 百分比: {np.isnan(current_x_stack).sum() / current_x_stack.size * 100:.2f}% (值范围过滤后)"
        )
        X_stacks.append(current_x_stack)  # 将当前自变量的数据栈添加到X_stacks列表中

    Y_stack_list = []  # 初始化一个列表，用于存储因变量的时间序列数据
    y_valid_range = variable_value_ranges.get(
        y_folder_basename, (None, None)
    )  # 获取因变量的有效值范围
    logging.info(
        f"因变量 {y_folder_basename}: 使用值范围 {y_valid_range}"
    )  # 记录因变量使用的值范围
    for f_idx, f_path in enumerate(
        tqdm(Y_files, desc="读取因变量影像")
    ):  # 遍历因变量的每个影像文件路径，使用tqdm显示进度条
        try:  # 尝试打开和读取影像
            ds = gdal.Open(f_path)  # 使用gdal打开影像文件
            if ds is None:  # 如果打开失败
                logging.error(f"无法打开影像文件: {f_path}")  # 记录错误日志
                Y_stack_list.append(
                    np.full((min_rows, min_cols), np.nan, dtype=np.float32)
                )  # 添加一个全为NaN的数组
                continue  # 继续下一个文件
            array = (
                ds.GetRasterBand(1)
                .ReadAsArray(col_offset, row_offset, min_cols, min_rows)
                .astype(np.float32)
            )  # 读取并裁剪影像
            ds = None  # 关闭GDAL数据集
            array_filtered = apply_value_range_filter(
                array,
                y_valid_range[0],
                y_valid_range[1],  # 调用函数过滤值
                f"{y_folder_basename}_file{f_idx}",
            )  # 传入变量名和文件名用于日志
            Y_stack_list.append(array_filtered)  # 将过滤后的数组添加到列表中
        except Exception as e:  # 捕获读取或处理影像时可能发生的任何异常
            logging.error(f"读取或处理影像 {f_path} 失败: {e}", exc_info=True)  # 记录错误日志
            Y_stack_list.append(
                np.full((min_rows, min_cols), np.nan, dtype=np.float32)
            )  # 发生错误时添加一个全为NaN的数组
    Y_stack = np.array(Y_stack_list)  # 将列表转换为NumPy数组，形成因变量的时间序列数据栈
    logging.info(  # 记录因变量数据栈的信息
        f"因变量 {y_folder_basename} stack: 形状 {Y_stack.shape}, NaN 百分比: {np.isnan(Y_stack).sum() / Y_stack.size * 100:.2f}% (值范围过滤后)"
    )

    if (
        any(x_s.size == 0 for x_s in X_stacks) or Y_stack.size == 0
    ):  # 检查是否有任何一个X数据栈或Y数据栈为空 (size为0)
        logging.error("一个或多个影像堆栈为空。")
        print(f"错误: 一个或多个影像堆栈为空。请检查日志 {log_filename}")
        return  # 记录错误，打印信息，终止
    if not all(
        x_s.shape[0] == Y_stack.shape[0] for x_s in X_stacks
    ):  # 检查所有X数据栈的时间序列长度 (shape[0]) 是否与Y数据栈的时间序列长度一致
        logging.error("自变量和因变量的时间序列长度不匹配。")
        print(f"错误: 自变量和因变量的时间序列长度不匹配。请检查日志 {log_filename}")
        return  # 记录错误，打印信息，终止

    num_cores = multiprocessing.cpu_count()  # 获取系统的CPU核心数
    # 调整n_jobs的逻辑，确保至少为1，并且在多核情况下留出核心 (例如留出2个核心给系统或其他任务)
    if not FULL_ANALYSIS:
        n_jobs = 1
    elif num_cores > 2:  # 如果核心数大于2
        n_jobs = num_cores - 2  # 使用核心数减2作为并行作业数
    elif num_cores == 2:  # 如果核心数为2
        n_jobs = 1  # 使用1个核心
    else:  # num_cores == 1 (如果核心数为1)
        n_jobs = 1  # 使用1个核心
    logging.info(
        f"系统CPU核心数: {num_cores}。将使用 {n_jobs} 个核心进行并行处理。"
    )  # 记录使用的并行核心数

    pixel_indices = [
        (r, c) for r in range(min_rows) for c in range(min_cols)
    ]  # 创建一个包含所有像元索引 (行, 列) 的列表
    logging.info("开始并行处理像元...")  # 记录日志信息

    if n_jobs == 1:  # 如果并行作业数为1 (即单核处理)
        logging.info("以单核模式运行。")  # 记录以单核模式运行
        results = []  # 初始化结果列表
        for r_idx, c_idx in tqdm(
            pixel_indices, desc="处理像元窗口 (单核)"
        ):  # 遍历所有像元索引，使用tqdm显示进度
            results.append(
                process_pixel_windowed(
                    r_idx, c_idx, X_stacks, Y_stack, num_variables, WINDOW_SIZE, min_rows, min_cols
                )
            )  # 调用处理函数并添加结果
    else:  # 如果并行作业数大于1 (多核处理)
        results = Parallel(
            n_jobs=n_jobs, backend="threading", verbose=0
        )(  # 使用joblib的Parallel进行并行处理，backend="threading"表示使用线程并行，verbose=0减少并行时的输出信息
            delayed(process_pixel_windowed)(
                r, c, X_stacks, Y_stack, num_variables, WINDOW_SIZE, min_rows, min_cols
            )  # delayed包装了要并行执行的函数及其参数
            for r, c in tqdm(
                pixel_indices, desc="处理像元窗口 (并行)"
            )  # 遍历所有像元索引，使用tqdm显示进度
        )

    logging.info("并行处理完成。将结果填充到数组中...")  # 记录日志信息
    valid_results_count = 0  # 初始化有效结果计数器 (即至少计算出一个非NaN相关系数的像元数量)
    for idx, result in enumerate(results):  # 遍历并行处理返回的结果列表
        if result is not None:  # 如果当前像元的结果不是None (即像元被成功处理了)
            r, c = pixel_indices[idx]  # 获取该结果对应的像元行、列索引
            partial_corr_pixel_results, pvalue_pixel_results = (
                result  # 解包结果为偏相关系数数组和p值数组
            )
            partial_correlation_array[:, r, c] = (
                partial_corr_pixel_results  # 将该像元的偏相关系数结果填充到总的偏相关系数数组中
            )
            pvalue_array[:, r, c] = pvalue_pixel_results  # 将该像元的p值结果填充到总的p值数组中
            if not np.isnan(
                partial_corr_pixel_results
            ).all():  # 如果该像元的偏相关系数结果不全是NaN
                valid_results_count += 1  # 有效结果计数器加1
    logging.info(
        f"总像元数: {len(pixel_indices)}, 计算得到至少一个有效相关系数的像元数: {valid_results_count}"
    )  # 记录总像元数和有效结果的像元数

    logging.info("保存结果...")  # 记录日志信息
    driver = gdal.GetDriverByName("GTiff")  # 获取GDAL的GeoTIFF驱动
    variable_names_for_output = [
        os.path.basename(folder) for folder in X_folders
    ]  # 获取用于输出文件名的变量名列表 (来自X文件夹的基本名称)

    for k in range(num_variables):  # 遍历每个自变量
        var_name = variable_names_for_output[k]  # 获取当前自变量的名称
        # 保存偏相关系数
        corr_output_filename = f"partial_correlation_{var_name}_win{WINDOW_SIZE}.tif"  # 构建偏相关系数输出文件名，包含变量名和窗口大小
        corr_output_path = os.path.join(
            output_folder, corr_output_filename
        )  # 构建完整的输出文件路径
        try:  # 尝试创建并写入GeoTIFF文件
            corr_ds = driver.Create(
                corr_output_path, min_cols, min_rows, 1, gdal.GDT_Float32
            )  # 创建GeoTIFF文件 (路径, 列数, 行数, 波段数, 数据类型)
            if corr_ds is None:
                logging.error(f"无法创建输出影像文件: {corr_output_path}")
                continue  # 如果创建失败，记录错误并跳过当前变量
            corr_ds.GetRasterBand(1).WriteArray(
                partial_correlation_array[k]
            )  # 将当前变量的偏相关系数二维数组写入第一个波段
            corr_ds.SetGeoTransform(geotransform)
            corr_ds.SetProjection(projection)
            corr_ds.FlushCache()
            corr_ds = None  # 设置地理变换、投影信息，刷新缓存，关闭文件
            logging.info(f"已保存: {corr_output_path}")  # 记录保存成功信息
        except Exception as e:  # 捕获保存文件时可能发生的任何异常
            logging.error(
                f"保存偏相关影像 {corr_output_path} 时出错: {e}", exc_info=True
            )  # 记录错误日志

        # 保存P值
        pvalue_output_filename = f"pvalue_{var_name}_win{WINDOW_SIZE}.tif"  # 构建P值输出文件名
        pvalue_output_path = os.path.join(
            output_folder, pvalue_output_filename
        )  # 构建完整的P值输出文件路径
        try:  # 尝试创建并写入GeoTIFF文件
            pvalue_ds = driver.Create(
                pvalue_output_path, min_cols, min_rows, 1, gdal.GDT_Float32
            )  # 创建GeoTIFF文件
            if pvalue_ds is None:
                logging.error(f"无法创建输出影像文件: {pvalue_output_path}")
                continue  # 如果创建失败，记录错误并跳过
            pvalue_ds.GetRasterBand(1).WriteArray(
                pvalue_array[k]
            )  # 将当前变量的P值二维数组写入第一个波段
            pvalue_ds.SetGeoTransform(geotransform)
            pvalue_ds.SetProjection(projection)
            pvalue_ds.FlushCache()
            pvalue_ds = None  # 设置地理信息，刷新缓存，关闭文件
            logging.info(f"已保存: {pvalue_output_path}")  # 记录保存成功信息
        except Exception as e:  # 捕获保存文件时可能发生的任何异常
            logging.error(
                f"保存P值影像 {pvalue_output_path} 时出错: {e}", exc_info=True
            )  # 记录错误日志

    if num_variables:
        save_array_preview(
            partial_correlation_array[0],
            os.path.join(output_folder, "preview.png"),
            f"Partial correlation: {variable_names_for_output[0]} vs {y_folder_basename}",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            cbar_label="partial r",
        )

    logging.info(
        f"----- 基于 {WINDOW_SIZE}x{WINDOW_SIZE} 窗口的偏相关性分析完成。结果保存在: {output_folder} -----"
    )  # 记录程序执行完成的日志信息
    print(
        f"处理完成。结果已保存到 {output_folder}。详情请查看日志文件: {log_filename}"
    )  # 在控制台打印完成信息和日志文件位置


if __name__ == "__main__":  # Python的入口点检查，确保当脚本被直接执行时运行以下代码
    multiprocessing.freeze_support()  # 对于在Windows上使用joblib或multiprocessing进行打包或冻结的应用程序，建议调用此函数以避免问题
    main()  # 调用主函数开始执行程序

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

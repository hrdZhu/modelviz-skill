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

# -*- coding: utf-8 -*-
import glob
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import optimize
from scipy.stats import linregress
from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["SimHei", "Times New Roman"]  # 'SimHei' 用于中文显示
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
matplotlib.use(
    "Agg"
)  # 设置 Matplotlib 的后端为 'Agg'，这意味着图像将被保存到文件中，而不是在屏幕上显示。


# HANTS 函数
def HANTS(y, t, Nh, delta, low, high, min_data, suppression_flag="both"):
    """
    使用 HANTS 算法重建时间序列。
    参数:
        y: 时间序列数据 (numpy 数组)，包含待处理的像素值序列，可以包含 NaN (缺失值)。
        t: 对应的时间点 (numpy 数组)，通常是时间序列数据的索引或日期对应的数值表示。
        Nh: 谐波数量 (整数)，决定了用于拟合季节性变化的傅里叶级数项的数量。
        delta: 异常值剔除阈值 (浮点数)，用于判断残差是否超出范围，从而识别异常值。
        low: 数据最小值 (浮点数)，用于初步筛选低于此值的无效数据。
        high: 数据最大值 (浮点数)，用于初步筛选高于此值的无效数据。
        min_data: 最小有效数据点数 (整数)，如果有效数据点少于此数量，则不进行拟合，直接返回 NaN 序列。
        suppression_flag: 异常值排除标志 (字符串)，控制异常值剔除的方向，
                          可选值："high" (只剔除高于拟合曲线的异常值),
                          "low" (只剔除低于拟合曲线的异常值),
                          "both" (剔除高于和低于拟合曲线的异常值),
                          "none" (不根据残差剔除异常值)。默认为 "both"。
    返回值:
        重建的时间序列 (numpy 数组)，长度与输入 y 相同，包含填补缺失值和异常值后的结果。
        如果拟合失败或有效数据不足，返回一个全是 NaN 的数组。
    """
    y = np.array(
        y, dtype=np.float32
    )  # 将输入的时间序列数据转换为 NumPy 数组，并确保数据类型为 float32。
    t = np.array(t)  # 将时间点转换为 NumPy 数组。
    # 应用数据合理范围筛选 (在提取数据时已做，但HANTS内部再次检查更安全)
    # 注意：这里的low/high检查是在HANTS内部对输入y进行的，不会改变外部的original_full_y
    y[(y < low) | (y > high)] = np.nan  # 将超出 [low, high] 范围的值标记为 NaN（缺失值）。
    valid_indices = ~np.isnan(y)  # 获取非 NaN 值的布尔索引。
    y_valid = y[valid_indices]  # 提取有效的（非 NaN）y 值。
    t_valid = t[valid_indices]  # 提取有效的（对应非 NaN 值的）时间点。
    if len(y_valid) < min_data:  # 如果有效数据点的数量少于设定的最小数据点数。
        return np.full_like(
            y, np.nan, dtype=np.float32
        )  # 则返回一个与输入 y 形状相同，但所有值都为 NaN 的数组。

    # 定义谐波函数：用于拟合时间序列的傅里叶级数模型
    def harmonic_function(t_points, *params):
        # t_points 可以是部分时间点 (用于拟合) 或全部时间点 (用于重建)
        A0 = params[0]  # 获取傅里叶级数的常数项 A0。
        # 检查 params 长度是否与预期的 1 + 2*Nh 相符
        if len(params) != (1 + 2 * Nh):  # 如果参数数量不正确。
            # print(f"警告: 拟合参数数量不正确: {len(params)} != {1 + 2 * Nh}")
            # 返回 NaN 数组以指示拟合模型参数错误
            return np.full_like(t_points, np.nan, dtype=np.float32)  # 返回一个全是 NaN 的数组。
        Ak = params[1 : Nh + 1]  # 获取余弦项的系数 Ak。
        Bk = params[Nh + 1 :]  # 获取正弦项的系数 Bk。
        # 周期使用原始时间序列的长度
        period = len(y)  # 使用输入时间序列 y 的总长度作为周期。
        # 如果 period 是 0 或小于 0 (不应该发生)，避免除以零
        if period <= 0:  # 如果周期无效。
            # print("警告: 时间序列长度为0或负数，无法计算谐波。")
            return np.full_like(t_points, np.nan, dtype=np.float32)  # 返回一个全是 NaN 的数组。
        # 避免在计算过程中出现 NaN 或 Inf
        try:
            t_points_reshaped = np.array(t_points).reshape(
                -1, 1
            )  # 将时间点重塑为 (N, 1) 的二维数组。
            k_reshaped = (
                np.arange(Nh).reshape(1, -1) + 1
            )  # 创建谐波次数 k 的数组，从 1 到 Nh，并重塑为 (1, Nh)。
            Ak_reshaped = np.array(Ak).reshape(1, -1)  # 将 Ak 重塑为 (1, Nh)。
            Bk_reshaped = np.array(Bk).reshape(1, -1)  # 将 Bk 重塑为 (1, Nh)。
            cos_terms = Ak_reshaped * np.cos(
                2 * np.pi * k_reshaped * t_points_reshaped / period
            )  # 计算余弦项。
            sin_terms = Bk_reshaped * np.sin(
                2 * np.pi * k_reshaped * t_points_reshaped / period
            )  # 计算正弦项。
            harmonic_sum = A0 + np.sum(cos_terms + sin_terms, axis=1)  # 计算傅里叶级数的和。
            if np.any(np.isnan(harmonic_sum)) or np.any(
                np.isinf(harmonic_sum)
            ):  # 如果计算结果包含 NaN 或 Inf。
                return np.full_like(t_points, np.nan, dtype=np.float32)  # 返回一个全是 NaN 的数组。
            return harmonic_sum  # 返回计算出的谐波函数值。
        except Exception as e:  # 捕获计算过程中可能发生的异常。
            return np.full_like(
                t_points, np.nan, dtype=np.float32
            )  # 发生异常时返回全是 NaN 的数组。

    # 初始参数估计
    mean_val = (
        np.nanmean(y_valid) if np.sum(~np.isnan(y_valid)) > 0 else 0.0
    )  # 计算有效数据的平均值作为 A0 的初始估计，如果无有效数据则为 0.0。
    initial_params = [mean_val] + [0.0] * (
        2 * Nh
    )  # 构造初始参数列表，A0 为平均值，其他 Ak 和 Bk 为 0。
    # 迭代剔除异常值
    max_iterations = 10  # 设置最大迭代次数。
    current_y_full = np.copy(y)  # 复制原始 y 数据，用于迭代过程中的修改。
    current_t_full = np.copy(t)  # 复制原始 t 数据。
    final_popt = None  # 初始化最终的拟合参数为 None。
    success_in_iteration = False  # 标志位，表示当前迭代是否成功拟合。
    for iteration in range(max_iterations):  # 进行迭代。
        valid_indices_current = ~np.isnan(current_y_full)  # 获取当前有效数据的索引。
        current_y_valid = current_y_full[valid_indices_current]  # 提取当前有效 y 值。
        current_t_valid = current_t_full[valid_indices_current]  # 提取当前有效时间点。
        if len(current_y_valid) < min_data:  # 如果当前有效数据点少于最小数据点数。
            break  # 停止迭代。
        try:
            if len(current_y_valid) != len(current_t_valid):  # 如果有效 y 值和时间点数量不匹配。
                break  # 停止迭代。
            p0_current = (
                initial_params if final_popt is None else final_popt
            )  # 设置当前拟合的初始参数：第一次迭代使用初始参数，之后使用上一次的拟合结果。
            if (
                p0_current is None or np.any(np.isnan(p0_current)) or np.any(np.isinf(p0_current))
            ):  # 如果初始参数无效。
                return np.full_like(y, np.nan, dtype=np.float32)  # 返回一个全是 NaN 的数组。
            param_scales = np.ones(1 + 2 * Nh, dtype=float)  # 初始化参数缩放因子为 1。
            if len(current_y_valid) > 0:  # 如果有有效数据。
                data_range = np.max(current_y_valid) - np.min(current_y_valid)  # 计算数据范围。
                if data_range > 1e-9:  # 如果数据范围足够大。
                    param_scales[0] = data_range  # 将 A0 的缩放因子设置为数据范围。
                else:
                    param_scales[0] = (
                        np.abs(mean_val) if np.abs(mean_val) > 1e-9 else 1.0
                    )  # 否则根据平均值设置。
            if Nh > 0 and len(current_y_valid) > 0 and data_range > 1e-9:  # 如果有谐波且数据有效。
                param_scales[1:] = data_range / (2 * Nh)  # 设置 Ak 和 Bk 的缩放因子。
            try:
                popt, pcov = optimize.curve_fit(
                    harmonic_function,
                    current_t_valid,
                    current_y_valid,
                    p0=p0_current,
                    maxfev=10000,
                    method="lm",
                    ftol=1e-4,
                    xtol=1e-4,
                    check_finite=True,
                )  # 使用 curve_fit 进行曲线拟合。
                if (
                    popt is None or np.any(np.isnan(popt)) or np.isinf(popt).any()
                ):  # 如果拟合结果参数无效。
                    popt = None  # 重置 popt。
                    raise RuntimeError("Optimizer returned NaN/Inf parameters")  # 抛出运行时错误。
                final_popt = popt  # 更新最终拟合参数。
                success_in_iteration = True  # 标记本次迭代成功。
            except (RuntimeError, ValueError, TypeError) as e:  # 捕获拟合过程中可能发生的错误。
                if iteration == 0 and final_popt is None:  # 如果是第一次迭代且没有成功拟合。
                    success_in_iteration = False  # 标记失败。
                break  # 停止迭代。
            if np.any(np.isnan(popt)) or np.any(np.isinf(popt)):  # 如果拟合参数包含 NaN 或 Inf。
                break  # 停止迭代。
            final_popt = popt  # 更新最终拟合参数。
            success_in_iteration = True  # 标记本次迭代成功。
        except (RuntimeError, ValueError, TypeError) as e:  # 捕获其他可能发生的错误。
            break  # 停止迭代。
        try:
            fitted_values = harmonic_function(
                current_t_valid, *final_popt
            )  # 使用拟合参数计算拟合值。
            if (
                fitted_values is None
                or np.any(np.isnan(fitted_values))
                or np.any(np.isinf(fitted_values))
            ):  # 如果拟合值无效。
                break  # 停止迭代。
        except (RuntimeError, ValueError, TypeError) as e:  # 捕获计算拟合值可能发生的错误。
            break  # 停止迭代。
        residuals = current_y_valid - fitted_values  # 计算残差（原始值与拟合值之差）。
        outlier_relative_indices = np.zeros_like(
            residuals, dtype=bool
        )  # 初始化异常值索引为全 False。

        if suppression_flag == "both":  # 如果剔除高于和低于拟合曲线的异常值。
            outlier_relative_indices = np.abs(residuals) > delta  # 绝对残差大于 delta 的为异常值。
        elif suppression_flag == "high":  # 如果只剔除高于拟合曲线的异常值。
            outlier_relative_indices = residuals > delta  # 残差大于 delta 的为异常值。
        elif suppression_flag == "low":  # 如果只剔除低于拟合曲线的异常值。
            outlier_relative_indices = residuals < -delta  # 残差小于 -delta 的为异常值。
        elif suppression_flag == "none":  # 如果不剔除异常值。
            pass  # 不做任何操作。

        if suppression_flag == "none":  # 如果不剔除异常值。
            break  # 直接停止迭代。

        if not np.any(outlier_relative_indices):  # 如果没有检测到异常值。
            break  # 停止迭代。

        outlier_original_indices_in_t_valid = np.where(outlier_relative_indices)[
            0
        ]  # 获取在 current_t_valid 中的异常值相对索引。
        outlier_original_t_values = current_t_valid[
            outlier_original_indices_in_t_valid
        ]  # 获取异常值对应的时间点。
        valid_indices_to_mask = outlier_original_t_values.astype(
            int
        )  # 将异常值时间点转换为整数索引。
        valid_indices_to_mask = valid_indices_to_mask[
            (valid_indices_to_mask >= 0) & (valid_indices_to_mask < len(current_y_full))
        ]  # 确保索引在有效范围内。
        current_y_full[valid_indices_to_mask] = np.nan  # 将异常值标记为 NaN。

    if final_popt is None or not success_in_iteration:  # 如果最终没有成功拟合参数。
        return np.full_like(y, np.nan, dtype=np.float32)  # 返回一个全是 NaN 的数组。

    try:
        y_reconstructed = harmonic_function(
            t, *final_popt
        )  # 使用最终拟合参数，在所有时间点上重建时间序列。
        if (
            y_reconstructed is None
            or np.any(np.isnan(y_reconstructed))
            or np.any(np.isinf(y_reconstructed))
        ):  # 如果重建结果无效。
            return np.full_like(y, np.nan, dtype=np.float32)  # 返回一个全是 NaN 的数组。
    except (RuntimeError, ValueError, TypeError) as e:  # 捕获重建过程中可能发生的错误。
        return np.full_like(y, np.nan, dtype=np.float32)  # 发生异常时返回全是 NaN 的数组。

    y_reconstructed = np.clip(y_reconstructed, low, high)  # 将重建值裁剪到 [low, high] 范围内。
    return y_reconstructed.astype(np.float32)  # 返回重建后的时间序列，数据类型为 float32。


def calculate_reconstruction_metrics(
    original_full_y=None,
    reconstructed_y=None,
    masked_indices=None,
    original_compare=None,
    reconstructed_compare=None,
):
    """
    计算时间序列重建的评估指标，包括相关系数、R²、RMSE、MAE 和粗糙度。

    参数:
        original_full_y: 原始完整的 Y 时间序列（用于在 masked_indices 处提取原始值）。
        reconstructed_y: 重建后的 Y 时间序列（用于在 masked_indices 处提取重建值）。
        masked_indices: 被人工遮蔽的点的索引。
        original_compare: 直接提供的原始值数组（用于直接比较）。
        reconstructed_compare: 直接提供的重建值数组（用于直接比较）。

    返回值:
        包含各项评估指标的字典。
    """
    metrics = {  # 初始化指标字典，所有值设为 NaN。
        "Correlation": np.nan,
        "R2": np.nan,
        "RMSE": np.nan,
        "MAE": np.nan,
        "Roughness": np.nan,
    }

    if (
        original_compare is not None and reconstructed_compare is not None
    ):  # 如果直接提供了比较数组。
        original_compare_subset = original_compare  # 直接使用提供的原始值。
        reconstructed_compare_subset = reconstructed_compare  # 直接使用提供的重建值。
        recon_series_for_roughness = None  # 粗糙度计算不使用此路径。
    elif (
        original_full_y is not None and reconstructed_y is not None and masked_indices is not None
    ):  # 如果提供了完整序列、重建序列和遮蔽索引。
        if (
            reconstructed_y is None
            or np.all(np.isnan(reconstructed_y))
            or len(original_full_y) != len(reconstructed_y)
        ):
            return metrics  # 如果重建序列无效，返回初始指标。
        original_at_masked = original_full_y[masked_indices]  # 提取原始序列在遮蔽点的值。
        reconstructed_at_masked = reconstructed_y[masked_indices]  # 提取重建序列在遮蔽点的值。
        valid_comparison_mask = np.isfinite(original_at_masked) & np.isfinite(
            reconstructed_at_masked
        )  # 创建有效比较点的掩码。
        original_compare_subset = original_at_masked[
            valid_comparison_mask
        ]  # 提取有效的原始比较值。
        reconstructed_compare_subset = reconstructed_at_masked[
            valid_comparison_mask
        ]  # 提取有效的重建比较值。
        recon_series_for_roughness = reconstructed_y  # 使用整个重建序列计算粗糙度。
    else:
        return metrics  # 如果输入参数不足，返回初始指标。

    if (
        original_compare_subset.size > 0
        and reconstructed_compare_subset.size > 0
        and original_compare_subset.shape == reconstructed_compare_subset.shape
    ):  # 如果有足够的有效比较数据。
        if len(original_compare_subset) >= 2:  # 如果至少有两个点用于计算相关性。
            try:
                if np.all(
                    original_compare_subset == original_compare_subset[0]
                ):  # 如果原始比较子集所有值都相同。
                    if np.all(
                        reconstructed_compare_subset == original_compare_subset[0]
                    ):  # 如果重建比较子集也与原始子集所有值相同。
                        metrics["Correlation"] = 1.0  # 相关系数为 1。
                    else:
                        metrics["Correlation"] = np.nan  # 否则为 NaN。
                else:
                    corr_matrix = np.corrcoef(
                        original_compare_subset.astype(float),
                        reconstructed_compare_subset.astype(float),
                    )  # 计算相关系数矩阵。
                    metrics["Correlation"] = np.clip(
                        corr_matrix[0, 1], -1.0, 1.0
                    )  # 提取相关系数并裁剪到 [-1, 1] 范围。
            except Exception as e:
                metrics["Correlation"] = np.nan  # 发生异常时相关系数为 NaN。

            try:
                ssr = np.sum(
                    (original_compare_subset - reconstructed_compare_subset) ** 2
                )  # 计算残差平方和（SSR）。
                if len(original_compare_subset) > 0:  # 如果有原始比较数据。
                    original_compare_mean = np.mean(
                        original_compare_subset
                    )  # 计算原始比较数据的平均值。
                    if np.isfinite(original_compare_mean):  # 如果平均值是有限的。
                        sst = np.sum(
                            (original_compare_subset - original_compare_mean) ** 2
                        )  # 计算总平方和（SST）。
                        if sst > 1e-9:  # 如果 SST 足够大。
                            metrics["R2"] = 1 - (ssr / sst)  # 计算 R²。
                        elif sst <= 1e-9:  # 如果 SST 非常小。
                            if ssr <= 1e-9:  # 如果 SSR 也非常小。
                                metrics["R2"] = 1.0  # R² 为 1。
                            else:
                                metrics["R2"] = np.nan  # 否则为 NaN。
                    else:
                        metrics["R2"] = np.nan  # 如果原始平均值无效，R² 为 NaN。
                else:
                    metrics["R2"] = np.nan  # 如果没有原始比较数据，R² 为 NaN。
            except Exception as e:
                metrics["R2"] = np.nan  # 发生异常时 R² 为 NaN。

        if len(original_compare_subset) >= 1:  # 如果至少有一个点用于计算 RMSE 和 MAE。
            try:
                metrics["RMSE"] = np.sqrt(
                    np.mean((original_compare_subset - reconstructed_compare_subset) ** 2)
                )  # 计算均方根误差（RMSE）。
            except Exception as e:
                metrics["RMSE"] = np.nan  # 发生异常时 RMSE 为 NaN。
            try:
                metrics["MAE"] = np.mean(
                    np.abs(original_compare_subset - reconstructed_compare_subset)
                )  # 计算平均绝对误差（MAE）。
            except Exception as e:
                metrics["MAE"] = np.nan  # 发生异常时 MAE 为 NaN。
        else:
            pass  # 如果没有足够的点，则跳过 RMSE 和 MAE 计算。

    if recon_series_for_roughness is not None:  # 如果有重建序列用于计算粗糙度。
        if not np.all(np.isnan(recon_series_for_roughness)):  # 如果重建序列不全是 NaN。
            if len(recon_series_for_roughness) >= 3:  # 如果至少有三个点用于计算粗糙度。
                try:
                    metrics["Roughness"] = np.sum(
                        np.diff(recon_series_for_roughness, n=2) ** 2
                    )  # 计算粗糙度。
                except Exception as e:
                    metrics["Roughness"] = np.nan  # 发生异常时粗糙度为 NaN。
            else:
                metrics["Roughness"] = 0.0  # 如果不足三个点，粗糙度为 0。
        else:
            metrics["Roughness"] = np.nan  # 如果全是 NaN，粗糙度为 NaN。
    return metrics  # 返回计算出的评估指标。


def extract_pixel_time_series(input_file_paths, sample_row, sample_col, low, high):
    """
    从一系列栅格文件中提取指定像素的时间序列数据。

    参数:
        input_file_paths: 包含所有时间步栅格文件路径的列表。
        sample_row: 像素的行索引。
        sample_col: 像素的列索引。
        low: 有效数据的最小值阈值。
        high: 有效数据的最大值阈值。

    返回值:
        pixel_time_series: 指定像素的时间序列数据（NumPy 数组），无效值或超出范围的值为 NaN。
        time_points: 对应时间序列数据的时间点索引（NumPy 数组）。
        如果无法打开文件或像素坐标无效，则返回 (None, None)。
    """
    if not input_file_paths:  # 如果输入文件路径列表为空。
        return None, None  # 返回 None。

    ref_dataset = gdal.Open(
        input_file_paths[0], gdal.GA_ReadOnly
    )  # 打开第一个文件作为参考数据集，获取栅格信息。
    if ref_dataset is None:  # 如果无法打开参考文件。
        return None, None  # 返回 None。

    cols = ref_dataset.RasterXSize  # 获取栅格的列数。
    rows = ref_dataset.RasterYSize  # 获取栅格的行数。
    ref_dataset = None  # 关闭参考数据集。

    if (
        sample_row < 0 or sample_row >= rows or sample_col < 0 or sample_col >= cols
    ):  # 如果像素坐标超出栅格范围。
        return None, None  # 返回 None。

    num_timesteps = len(input_file_paths)  # 获取时间步的数量。
    pixel_time_series = np.zeros(num_timesteps, dtype=np.float32)  # 初始化像素时间序列数组为全零。
    pixel_time_series.fill(np.nan)  # 将所有值填充为 NaN。

    for i, ds_path in enumerate(input_file_paths):  # 遍历所有输入文件路径。
        ds = gdal.Open(ds_path, gdal.GA_ReadOnly)  # 打开当前栅格文件。
        if ds:  # 如果文件成功打开。
            try:
                pixel_value = ds.GetRasterBand(1).ReadAsArray(sample_col, sample_row, 1, 1)[
                    0, 0
                ]  # 读取指定像素的值。
                if np.isfinite(pixel_value) and not (
                    pixel_value < low or pixel_value > high
                ):  # 如果像素值是有限的且在有效范围内。
                    pixel_time_series[i] = float(pixel_value)  # 将像素值存储到时间序列中。
                else:
                    pixel_time_series[i] = np.nan  # 否则标记为 NaN。
            except Exception as e:  # 捕获读取过程中可能发生的异常。
                pixel_time_series[i] = np.nan  # 发生异常时标记为 NaN。
            finally:
                ds = None  # 关闭数据集。
        else:
            pixel_time_series[i] = np.nan  # 如果文件无法打开，标记为 NaN。

    time_points = np.arange(num_timesteps)  # 使用时间步索引作为时间点。
    return pixel_time_series, time_points  # 返回像素时间序列和时间点。


# --- 函数：寻找具有完整时间序列数据的随机像素 ---
def find_complete_pixels(input_file_paths, num_pixels_to_find, low, high):
    """
    从栅格图像集合中寻找指定数量的具有完整时间序列数据的随机像素。

    参数:
        input_file_paths: 包含所有时间步栅格文件路径的列表。
        num_pixels_to_find: 希望找到的完整像素的数量。
        low: 有效数据的最小值阈值。
        high: 有效数据的最大值阈值。

    返回值:
        found_pixels: 找到的完整像素的 (row, col) 坐标列表。
        complete_pixel_data: 字典，键为 (row, col) 坐标，值为该像素的完整时间序列数据。
    """
    print(f"\n正在寻找 {num_pixels_to_find} 个具有完整时间序列数据的像素...")  # 打印提示信息。

    if not input_file_paths:  # 如果输入文件路径列表为空。
        print("错误: 输入文件路径列表为空，无法寻找像素。")  # 打印错误信息。
        return [], {}  # 返回空列表和空字典。

    ref_dataset = gdal.Open(input_file_paths[0], gdal.GA_ReadOnly)  # 打开第一个文件作为参考数据集。
    if ref_dataset is None:  # 如果无法打开参考文件。
        print(f"错误: 无法打开参考文件 {input_file_paths[0]}，无法获取栅格信息。")  # 打印错误信息。
        return [], {}  # 返回空列表和空字典。

    cols = ref_dataset.RasterXSize  # 获取栅格列数。
    rows = ref_dataset.RasterYSize  # 获取栅格行数。
    ref_dataset = None  # 关闭参考数据集。

    all_coords = [(r, c) for r in range(rows) for c in range(cols)]  # 生成所有像素坐标的列表。
    random.shuffle(all_coords)  # 随机打乱所有坐标，以便随机选择像素。

    found_pixels = []  # 存储找到的 (row, col) 坐标。
    complete_pixel_data = {}  # 存储找到的完整时间序列数据。

    pbar = tqdm(all_coords, desc="Searching for complete pixels")  # 创建进度条。
    for r, c in pbar:  # 遍历所有随机打乱后的像素坐标。
        if len(found_pixels) >= num_pixels_to_find:  # 如果已经找到足够数量的像素。
            break  # 停止寻找。
        pixel_y, _ = extract_pixel_time_series(
            input_file_paths, r, c, low, high
        )  # 提取当前像素的时间序列。
        if pixel_y is not None and not np.any(
            np.isnan(pixel_y)
        ):  # 如果像素时间序列不为 None 且不包含 NaN（即是完整的）。
            found_pixels.append((r, c))  # 将该像素坐标添加到找到的列表中。
            complete_pixel_data[(r, c)] = pixel_y  # 存储该像素的完整时间序列数据。
            pbar.set_description(
                f"Found {len(found_pixels)}/{num_pixels_to_find} complete pixels"
            )  # 更新进度条描述。
    pbar.close()  # 关闭进度条。

    print(f"完成寻找像素。共找到 {len(found_pixels)} 个符合条件的像素。")  # 打印找到的像素数量。
    if len(found_pixels) < num_pixels_to_find:  # 如果找到的像素数量不足。
        print(
            f"警告: 未能找到足够的 ({num_pixels_to_find}) 具有完整时间序列的像素。将使用找到的 {len(found_pixels)} 个像素进行测试。"
        )  # 打印警告。

    return found_pixels, complete_pixel_data  # 返回找到的像素列表和其数据。


def create_masked_series(original_full_y, num_points_to_mask):
    """
    在原始时间序列中随机制造缺失值（NaN）。

    参数:
        original_full_y: 原始完整的 Y 时间序列。
        num_points_to_mask: 要制造缺失点的数量。

    返回值:
        masked_y: 制造缺失值后的时间序列。
        masked_indices: 被遮蔽的点的索引。
    """
    if num_points_to_mask <= 0:  # 如果要遮蔽的点数小于等于 0。
        return np.copy(original_full_y), np.array(
            [], dtype=int
        )  # 返回原始序列和空索引，表示不制造缺失。

    series_length = len(original_full_y)  # 获取时间序列的长度。
    if num_points_to_mask >= series_length:  # 如果要遮蔽的点数大于等于序列长度。
        masked_y = np.full_like(original_full_y, np.nan)  # 创建一个全是 NaN 的序列。
        masked_indices = np.arange(series_length, dtype=int)  # 所有点的索引都被遮蔽。
        return masked_y, masked_indices  # 返回全是 NaN 的序列和所有索引。

    masked_y = np.copy(original_full_y)  # 复制原始序列。
    masked_indices = np.random.choice(
        series_length, size=num_points_to_mask, replace=False
    )  # 随机选择不重复的索引作为遮蔽点。
    masked_y[masked_indices] = np.nan  # 将这些索引处的原始值替换为 NaN。
    masked_indices = np.sort(masked_indices)  # 对遮蔽的索引进行排序。
    return masked_y, masked_indices  # 返回制造缺失值后的序列和被遮蔽的索引。


input_folder = str(DATA_DIR)  # 替换为您的输入文件夹路径。
output_folder = str(OUTPUT_DIR)  # 设置图表和结果输出文件夹。
num_pixels_to_test = 3  # 设置要测试的像素数量。
num_points_to_mask_per_pixel = 5  # 设置每个像素要人为制造的缺失点数量。
global_low = -40  # 设置全局数据下限。
global_high = 40  # 设置全局数据上限。

# 定义不同的 HANTS 参数组合进行测试。
test_params_list = [
    {
        "Nh": 2,
        "delta": 5.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "both",
        "label": "P1_Nh2_d5_bth",
    },
    {
        "Nh": 4,
        "delta": 6.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "both",
        "label": "P2_Nh4_d6_bth",
    },
    {
        "Nh": 4,
        "delta": 10.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "both",
        "label": "P3_Nh4_d10_bth",
    },
    {
        "Nh": 4,
        "delta": 3.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "both",
        "label": "P4_Nh4_d3_bth",
    },
    {
        "Nh": 6,
        "delta": 6.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "both",
        "label": "P5_Nh6_d6_bth",
    },
    {
        "Nh": 4,
        "delta": 6.0,
        "low": global_low,
        "high": global_high,
        "min_data": 4,
        "suppression_flag": "both",
        "label": "P6_Nh4_d6_min4",
    },
    {
        "Nh": 4,
        "delta": 6.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "high",
        "label": "P7_Nh4_d6_high",
    },
    {
        "Nh": 4,
        "delta": 6.0,
        "low": global_low,
        "high": global_high,
        "min_data": 7,
        "suppression_flag": "none",
        "label": "P8_Nh4_d6_none",
    },
]

if not os.path.exists(output_folder):  # 如果输出文件夹不存在。
    os.makedirs(output_folder)  # 创建输出文件夹。

print(f"\n正在扫描输入文件夹: {input_folder}")  # 打印扫描文件夹信息。
tif_files = sorted(
    glob.glob(os.path.join(input_folder, "**", "*.tif"), recursive=True)
)  # 查找输入文件夹中所有 .tif 文件并排序。

if not tif_files:  # 如果没有找到 .tif 文件。
    print(f"错误: 在 {input_folder} 中没有找到 .tif 文件。请检查路径。")  # 打印错误信息。
    exit()  # 退出程序。

time_file_map = {}  # 创建一个字典，用于存储时间字符串和文件路径的映射。
input_file_paths = []  # 存储排序后的文件路径。
time_strings = []  # 存储排序后的时间字符串。

print("正在解析文件名和时间信息...")  # 打印解析信息。
for tif_file in tif_files:  # 遍历所有 .tif 文件。
    filename = os.path.basename(tif_file)  # 获取文件名。
    if (
        len(filename) >= 10 and filename[4] == "_" and filename[7] == "_"
    ):  # 根据特定命名格式（例如：YYYY_MM_DD_...）解析时间字符串。
        time_str = filename[:10]  # 提取时间字符串。
        time_file_map[time_str] = tif_file  # 存储映射关系。
    else:
        pass  # 如果不符合格式则跳过。

if not time_file_map:  # 如果没有找到符合命名格式的 TIF 文件。
    print("错误: 没有找到符合命名格式的 TIF 文件。")  # 打印错误信息。
    exit()  # 退出程序。

sorted_times = sorted(time_file_map.keys())  # 对时间字符串进行排序。
input_file_paths = [time_file_map[t] for t in sorted_times]  # 根据排序后的时间获取文件路径。
time_points = np.arange(len(input_file_paths))  # 创建时间点（索引）。
time_strings = sorted_times  # 存储排序后的时间字符串。

print(f"\n共找到 {len(input_file_paths)} 个时间步文件。")  # 打印找到的时间步文件数量。
print(f"准备测试 {num_pixels_to_test} 个随机选取的具有完整时间序列的像素。")  # 打印测试信息。
print(f"将在每个像素中人为制造 {num_points_to_mask_per_pixel} 个缺失点。")  # 打印缺失点信息。
print(f"测试 {len(test_params_list)} 组参数组合。")  # 打印参数组合数量。

found_pixels_list, original_full_pixel_data = find_complete_pixels(
    input_file_paths, num_pixels_to_test, global_low, global_high
)  # 寻找具有完整时间序列的像素。

if not found_pixels_list:  # 如果没有找到任何符合条件的像素。
    print(
        "错误: 未找到任何符合条件的完整像素，无法进行测试。请检查输入数据、low/high 范围和文件格式。"
    )  # 打印错误信息。
    exit()  # 退出程序。

individual_metrics_list = []  # 存储单个像素的评估指标。
reconstructed_results = {}  # 存储重建结果。
masked_series_info = {}  # 存储遮蔽序列的信息。

print("\n正在进行 HANTS 重建测试...")  # 打印 HANTS 重建测试信息。
for pixel_row, pixel_col in tqdm(
    found_pixels_list, desc="Processing pixels"
):  # 遍历找到的每个像素，并显示进度条。
    original_full_y = original_full_pixel_data[(pixel_row, pixel_col)]  # 获取原始完整时间序列。
    masked_y, masked_indices = create_masked_series(
        original_full_y, num_points_to_mask_per_pixel
    )  # 创建带缺失值的序列。
    reconstructed_results[(pixel_row, pixel_col)] = {}  # 初始化当前像素的重建结果字典。
    masked_series_info[(pixel_row, pixel_col)] = {
        "masked_y": masked_y,
        "masked_indices": masked_indices,
    }  # 存储遮蔽序列信息。

    for params in test_params_list:  # 遍历每个参数组合。
        param_label = params.get(
            "label",
            f"Nh{params['Nh']}_d{params['delta']}_min{params['min_data']}_supp{params['suppression_flag']}",
        )  # 获取或生成参数标签。
        param_label = param_label.replace(" ", "_").replace(
            ":", "_"
        )  # 替换标签中的特殊字符，使其适合作为文件名。
        params["label"] = param_label  # 更新参数字典中的标签。

        reconstructed_y = HANTS(
            masked_y.copy(),
            time_points.copy(),
            params["Nh"],
            params["delta"],
            params["low"],
            params["high"],
            params["min_data"],
            params["suppression_flag"],
        )  # 调用 HANTS 函数进行重建。

        reconstructed_results[(pixel_row, pixel_col)][param_label] = (
            reconstructed_y  # 存储重建结果。
        )

        metrics = calculate_reconstruction_metrics(
            original_full_y=original_full_y,
            reconstructed_y=reconstructed_y,
            masked_indices=masked_indices,
        )  # 计算重建指标。

        individual_metrics_list.append(
            {  # 将单个像素的指标添加到列表中。
                "Pixel": f"({pixel_row}, {pixel_col})",
                "Params": param_label,
                "Masked_Points_Count": len(masked_indices),
                "Successfully_Compared_Count": np.sum(
                    np.isfinite(original_full_y[masked_indices])
                    & np.isfinite(reconstructed_y[masked_indices])
                ),
                **metrics,
            }
        )

# 4. 绘制每个像素的对比图并保存
print("\n正在生成并保存单个像素的对比图...")  # 打印绘图信息。
for pixel_row, pixel_col in tqdm(
    found_pixels_list, desc="Plotting pixels"
):  # 遍历每个像素，并显示进度条。
    original_full_y = original_full_pixel_data[(pixel_row, pixel_col)]  # 获取原始完整时间序列。
    masked_y = masked_series_info[(pixel_row, pixel_col)]["masked_y"]  # 获取带缺失值的序列。
    masked_indices = masked_series_info[(pixel_row, pixel_col)][
        "masked_indices"
    ]  # 获取被遮蔽的索引。
    pixel_reconstructions = reconstructed_results[
        (pixel_row, pixel_col)
    ]  # 获取当前像素的所有重建结果。

    plt.figure(figsize=(18, 8))  # 创建一个新的 Matplotlib 图形。
    plt.plot(
        time_points,
        original_full_y,
        "-",
        color="grey",
        label="Original Full Data (Truth)",
        linewidth=1.5,
        alpha=0.7,
    )  # 绘制原始完整数据。
    plt.plot(
        time_points,
        masked_y,
        ".",
        color="blue",
        label="HANTS Input (with NaNs)",
        markersize=4,
        alpha=0.6,
    )  # 绘制带缺失值的输入数据。
    plt.plot(
        time_points[masked_indices],
        original_full_y[masked_indices],
        "ro",
        label="Artificially Masked Points (Original Value)",
        markersize=8,
        fillstyle="none",
        markeredgewidth=1.5,
    )  # 绘制人工遮蔽点的原始值。

    cmap = matplotlib.colormaps.get_cmap("tab10")  # 获取颜色映射。
    num_params = len(test_params_list)  # 获取参数组合数量。
    colors = cmap(np.arange(num_params))  # 生成颜色列表。
    linestyles = ["-", "--", "-.", ":"]  # 定义线型列表。
    param_styles = {
        params["label"]: (colors[i % num_params], linestyles[i % len(linestyles)])
        for i, params in enumerate(test_params_list)
    }  # 为每个参数组合分配颜色和线型。

    for params in test_params_list:  # 遍历每个参数组合。
        param_label = params["label"]  # 获取参数标签。
        reconstructed_y = pixel_reconstructions.get(param_label)  # 获取对应的重建结果。
        color, linestyle = param_styles[param_label]  # 获取颜色和线型。

        if reconstructed_y is not None and not np.all(
            np.isnan(reconstructed_y)
        ):  # 如果重建结果有效。
            plt.plot(
                time_points,
                reconstructed_y,
                linestyle=linestyle,
                color=color,
                label=f"Recon: {param_label}",
                linewidth=2,
            )  # 绘制重建序列。
            reconstructed_at_masked = reconstructed_y[masked_indices]  # 提取重建序列在遮蔽点的值。
            valid_recon_at_masked_mask = np.isfinite(
                reconstructed_at_masked
            )  # 获取有效重建点的掩码。
            if np.sum(valid_recon_at_masked_mask) > 0:  # 如果有有效的重建点。
                plt.plot(
                    time_points[masked_indices[valid_recon_at_masked_mask]],
                    reconstructed_at_masked[valid_recon_at_masked_mask],
                    "X",
                    color=color,
                    markersize=10,
                    markeredgewidth=1.5,
                    label=f"Recon Value @ Masked ({param_label})",
                )  # 绘制重建值在遮蔽点的位置。

    plt.xlabel("时间点 (索引)", fontsize=14, fontweight="bold")  # 设置 X 轴标签。
    plt.ylabel("值", fontsize=14, fontweight="bold")  # 设置 Y 轴标签。
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.title(
        f"HANTS 重建测试：像素 ({pixel_row}, {pixel_col})\n({len(masked_indices)} 点被遮蔽)",
        fontsize=14,
        fontweight="bold",
    )  # 设置图表标题。

    if (
        time_points is not None and len(time_points) > 0 and len(time_strings) == len(time_points)
    ):  # 如果时间点和时间字符串有效。
        step = max(1, len(time_points) // 15)  # 计算 X 轴刻度的步长，确保不超过 15 个刻度。
        xticks_indices = time_points[::step]  # 获取 X 轴刻度索引。
        xtick_labels = [time_strings[i] for i in xticks_indices]  # 获取 X 轴刻度标签。
        if time_points[-1] not in xticks_indices:  # 如果最后一个时间点不在刻度中。
            xticks_indices = np.append(xticks_indices, time_points[-1])  # 添加最后一个时间点。
            xtick_labels.append(time_strings[-1])  # 添加最后一个时间字符串。
        plt.xticks(xticks_indices, xtick_labels, rotation=45, ha="right")  # 设置 X 轴刻度及其标签。

    plt.legend(loc="best", fontsize="small")  # 显示图例。
    plt.grid(True)  # 显示网格。
    plt.tight_layout()  # 调整布局，防止标签重叠。

    safe_pixel_label = f"{pixel_row}_{pixel_col}"  # 创建安全的文件名标签。
    filename = f"pixel_{safe_pixel_label}_reconstruction_comparison.png"  # 生成文件名。
    filepath = os.path.join(output_folder, filename)  # 拼接文件路径。
    plt.savefig(filepath, dpi=300)  # 保存图形。
    plt.close()  # 关闭图形，释放内存。

# 5. 打印单个像素的评估指标表格
print("\n--- 单个像素重建精度评估指标 (聚焦在人为缺失点) ---")  # 打印标题。
if individual_metrics_list:  # 如果有单个像素的指标。
    individual_metrics_df = pd.DataFrame(individual_metrics_list)  # 创建 Pandas DataFrame。
    individual_metrics_df_sorted_for_print = individual_metrics_df.sort_values(
        by=["Pixel", "RMSE"]
    ).reset_index(drop=True)  # 按像素和 RMSE 排序。
    individual_metrics_df_rounded = individual_metrics_df_sorted_for_print.round(
        3
    )  # 将指标值四舍五入到小数点后三位。
    cols_order = [
        "Pixel",
        "Params",
        "Masked_Points_Count",
        "Successfully_Compared_Count",
        "Correlation",
        "R2",
        "RMSE",
        "MAE",
        "Roughness",
    ]  # 定义列的顺序。
    ordered_cols = [
        col for col in cols_order if col in individual_metrics_df_rounded.columns
    ]  # 确保列存在于 DataFrame 中。
    extra_cols = [
        col for col in individual_metrics_df_rounded.columns if col not in cols_order
    ]  # 获取额外列。
    ordered_cols.extend(extra_cols)  # 将额外列添加到有序列表中。
    print(individual_metrics_df_rounded[ordered_cols].to_string(index=False))  # 打印表格。
    csv_filename = "individual_pixel_reconstruction_metrics.csv"  # CSV 文件名。
    csv_filepath = os.path.join(output_folder, csv_filename)  # CSV 文件路径。
    individual_metrics_df.to_csv(csv_filepath, index=False)  # 保存原始精度的数据到 CSV 文件。
    print(f"单个像素评估指标已保存到: {csv_filepath}")  # 打印保存路径。
else:
    print("没有计算到单个像素的指标。")  # 打印没有指标的提示。

# 6. 计算并打印总体评估指标 (跨所有样本像素的人为缺失点)
print("\n--- 总体重建精度评估指标 (跨所有样本像素的人为缺失点) ---")  # 打印标题。
overall_metrics_list = []  # 存储总体评估指标。
all_comparison_points_per_param = {}  # 存储每个参数组合的所有比较点。
all_reconstructed_series_per_param = {}  # 存储每个参数组合的所有重建序列。

for params in tqdm(
    test_params_list, desc="Aggregating overall metrics"
):  # 遍历每个参数组合，并显示进度条。
    param_label = params["label"]  # 获取参数标签。
    all_comparison_points_per_param[param_label] = {
        "original": [],
        "reconstructed": [],
    }  # 初始化原始和重建比较点列表。
    all_reconstructed_series_per_param[param_label] = []  # 初始化重建序列列表。

    for pixel_row, pixel_col in found_pixels_list:  # 遍历每个找到的像素。
        original_full_y = original_full_pixel_data[(pixel_row, pixel_col)]  # 获取原始完整时间序列。
        masked_indices = masked_series_info[(pixel_row, pixel_col)][
            "masked_indices"
        ]  # 获取被遮蔽的索引。
        reconstructed_y = reconstructed_results[(pixel_row, pixel_col)].get(
            param_label
        )  # 获取对应的重建结果。

        if reconstructed_y is not None:  # 如果重建结果有效。
            original_at_masked = original_full_y[masked_indices]  # 提取原始序列在遮蔽点的值。
            reconstructed_at_masked = reconstructed_y[masked_indices]  # 提取重建序列在遮蔽点的值。
            valid_comparison_mask = np.isfinite(original_at_masked) & np.isfinite(
                reconstructed_at_masked
            )  # 获取有效比较点的掩码。
            all_comparison_points_per_param[param_label]["original"].extend(
                original_at_masked[valid_comparison_mask].tolist()
            )  # 添加有效的原始值。
            all_comparison_points_per_param[param_label]["reconstructed"].extend(
                reconstructed_at_masked[valid_comparison_mask].tolist()
            )  # 添加有效的重建值。

            if not np.all(np.isnan(reconstructed_y)):  # 如果重建序列不全是 NaN。
                all_reconstructed_series_per_param[param_label].append(
                    reconstructed_y
                )  # 添加重建序列。

    combined_original_compare = np.array(
        all_comparison_points_per_param[param_label]["original"]
    )  # 合并所有原始比较点。
    combined_reconstructed_compare = np.array(
        all_comparison_points_per_param[param_label]["reconstructed"]
    )  # 合并所有重建比较点。

    overall_metrics = calculate_reconstruction_metrics(
        original_compare=combined_original_compare,
        reconstructed_compare=combined_reconstructed_compare,
    )  # 计算总体重建指标。

    roughness_values = []  # 存储粗糙度值。
    for recon_series in all_reconstructed_series_per_param[param_label]:  # 遍历每个重建序列。
        if len(recon_series) >= 3:  # 如果序列长度大于等于 3。
            try:
                roughness_values.append(np.sum(np.diff(recon_series, n=2) ** 2))  # 计算粗糙度。
            except Exception as e:
                pass  # 发生异常则跳过。
    overall_roughness = (
        np.nan if not roughness_values else np.mean(roughness_values)
    )  # 计算平均粗糙度。
    overall_metrics["Roughness"] = overall_roughness  # 更新总体指标中的粗糙度。
    overall_metrics["Total_Potential_Masked_Points"] = (
        num_pixels_to_test * num_points_to_mask_per_pixel
    )  # 记录总潜在遮蔽点数。
    overall_metrics["Total_Successfully_Compared_Points"] = len(
        combined_original_compare
    )  # 记录总成功比较点数。
    overall_metrics_list.append(
        {"Params": param_label, **overall_metrics}
    )  # 将总体指标添加到列表中。

# 7. 打印总体评估指标表格
print("\n--- 总体重建精度评估指标 (跨所有样本像素的人为缺失点) ---")  # 打印标题。
overall_metrics_df = None  # 初始化 DataFrame 为 None。
if overall_metrics_list:  # 如果有总体指标。
    overall_metrics_df = pd.DataFrame(overall_metrics_list)  # 创建 Pandas DataFrame。
    overall_metrics_df_sorted_for_print = overall_metrics_df.sort_values(
        by="RMSE", ascending=True
    ).reset_index(drop=True)  # 按 RMSE 排序。
    overall_metrics_df_rounded = overall_metrics_df_sorted_for_print.round(3)  # 四舍五入。
    cols_order = [
        "Params",
        "Total_Potential_Masked_Points",
        "Total_Successfully_Compared_Points",
        "Correlation",
        "R2",
        "RMSE",
        "MAE",
        "Roughness",
    ]  # 定义列的顺序。
    ordered_cols = [
        col for col in cols_order if col in overall_metrics_df_rounded.columns
    ]  # 确保列存在。
    extra_cols = [
        col for col in overall_metrics_df_rounded.columns if col not in cols_order
    ]  # 获取额外列。
    ordered_cols.extend(extra_cols)  # 添加额外列。
    print(overall_metrics_df_rounded[ordered_cols].to_string(index=False))  # 打印表格。
    csv_filename = "overall_reconstruction_metrics.csv"  # CSV 文件名。
    csv_filepath = os.path.join(output_folder, csv_filename)  # CSV 文件路径。
    overall_metrics_df.to_csv(csv_filepath, index=False)  # 保存原始精度的数据到 CSV 文件。
    print(f"总体评估指标已保存到: {csv_filepath}")  # 打印保存路径。
else:
    print("没有计算到总体指标。")  # 打印没有指标的提示。

# --- 绘制总体“原始 vs. 重建”散点图的代码 (包含指标文本和回归线) ---
print("\n正在生成并保存总体 '原始 vs. 重建' 散点图 (包含指标和回归线)...")  # 打印绘图信息。
if overall_metrics_df is None or overall_metrics_df.empty:  # 如果没有总体评估指标。
    print(
        "错误: 未成功计算到总体评估指标，无法在图上显示指标文本或绘制回归线。跳过绘制总体 '原始 vs. 重建' 散点图。"
    )  # 打印错误信息。
else:
    for params in tqdm(
        test_params_list, desc="Plotting overall reconstruction scatter"
    ):  # 遍历每个参数组合，并显示进度条。
        param_label = params["label"]  # 获取参数标签。
        overall_orig_compare = np.array(
            all_comparison_points_per_param[param_label]["original"]
        )  # 获取总体原始比较点。
        overall_recon_compare = np.array(
            all_comparison_points_per_param[param_label]["reconstructed"]
        )  # 获取总体重建比较点。

        if overall_orig_compare.size > 0:  # 如果有数据点进行散点图绘制。
            plt.figure(figsize=(8, 8))  # 创建新的图形。
            plt.scatter(overall_orig_compare, overall_recon_compare, alpha=0.5, s=5)  # 绘制散点图。

            # 确定绘图范围
            min_orig = (
                np.min(overall_orig_compare)
                if overall_orig_compare.size > 0 and np.isfinite(np.min(overall_orig_compare))
                else params.get("low", global_low)
            )
            max_orig = (
                np.max(overall_orig_compare)
                if overall_orig_compare.size > 0 and np.isfinite(np.max(overall_orig_compare))
                else params.get("high", global_high)
            )
            min_recon = (
                np.min(overall_recon_compare)
                if overall_recon_compare.size > 0 and np.isfinite(np.min(overall_recon_compare))
                else params.get("low", global_low)
            )
            max_recon = (
                np.max(overall_recon_compare)
                if overall_recon_compare.size > 0 and np.isfinite(np.max(overall_recon_compare))
                else params.get("high", global_high)
            )
            min_val = min(min_orig, min_recon, params.get("low", global_low))  # 计算最小绘图值。
            max_val = max(max_orig, max_recon, params.get("high", global_high))  # 计算最大绘图值。

            buffer = (
                (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-9 else 1.0
            )  # 计算缓冲区。
            plot_min = min_val - buffer  # 最终绘图最小值。
            plot_max = max_val + buffer  # 最终绘图最大值。

            if plot_max - plot_min < 1e-9:  # 如果范围太小。
                plot_min = min_val - 1 if np.isfinite(min_val) else -10  # 调整最小值。
                plot_max = max_val + 1 if np.isfinite(max_val) else 10  # 调整最大值。
                if plot_max - plot_min < 1e-9:
                    plot_max = plot_min + 2  # 确保范围足够。

            plt.plot(
                [plot_min, plot_max],
                [plot_min, plot_max],
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="y=x",
            )  # 绘制理想的 y=x 线。

            # --- 计算并绘制线性回归线 ---
            if overall_orig_compare.size >= 2:  # 如果至少有两个点用于线性回归。
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        overall_orig_compare, overall_recon_compare
                    )  # 计算线性回归参数。
                    line_x = np.linspace(plot_min, plot_max, 100)  # 生成回归线的 X 值。
                    line_y = slope * line_x + intercept  # 计算回归线的 Y 值。
                    plt.plot(
                        line_x,
                        line_y,
                        color="blue",
                        linestyle="-",
                        linewidth=2,
                        label=f"y={slope:.2f}x + {intercept:.2f}",
                    )  # 绘制线性回归线。
                except Exception as e:
                    pass  # 发生异常则跳过。
            else:
                pass  # 如果点数不足则跳过。

            plt.xlabel("原始数据值", fontsize=14, fontweight="bold")  # 设置 X 轴标签。
            plt.ylabel("重建数据值", fontsize=14, fontweight="bold")  # 设置 Y 轴标签。
            plt.tick_params(axis="both", which="major", labelsize=16)

            num_pixels_contributing = sum(
                1
                for pixel_data_list in all_reconstructed_series_per_param.get(param_label, [])
                if pixel_data_list is not None and not np.all(np.isnan(pixel_data_list))
            )  # 计算贡献的像素数量。
            if num_pixels_contributing == 0:  # 如果没有贡献像素。
                num_pixels_contributing = len(found_pixels_list)  # 使用找到的总像素数。

            plt.title(
                f"总体原始值 vs. 重建值：参数组合 {param_label}\n(在 {num_pixels_contributing} 个像素的人为遮蔽点处进行比较)",
                fontsize=14,
                fontweight="bold",
            )  # 更新图表标题。
            plt.legend(loc="upper left", fontsize=16)  # 显示图例。
            plt.grid(True, linestyle="--", alpha=0.6)  # 显示网格。
            # plt.axis('equal') # 设置 X 和 Y 轴比例相等。
            # 调整轴限以确保所有点可见
            current_xlim = plt.xlim()  # 获取当前 X 轴限。
            current_ylim = plt.ylim()  # 获取当前 Y 轴限。
            final_plot_min = min(plot_min, current_xlim[0], current_ylim[0])  # 计算最终绘图最小值。
            final_plot_max = max(plot_max, current_xlim[1], current_ylim[1])  # 计算最终绘图最大值。
            # plt.xlim([final_plot_min, final_plot_max]) # 设置 X 轴限。
            # plt.ylim([final_plot_min, final_plot_max]) # 设置 Y 轴限。
            param_metrics_row = overall_metrics_df_rounded[
                overall_metrics_df_rounded["Params"] == param_label
            ]  # 获取当前参数组合的评估指标行。
            if not param_metrics_row.empty:  # 如果指标行不为空。
                corr = param_metrics_row["Correlation"].iloc[0]  # 获取相关系数。
                r2 = param_metrics_row["R2"].iloc[0]  # 获取 R²。
                rmse = param_metrics_row["RMSE"].iloc[0]  # 获取 RMSE。
                mae = param_metrics_row["MAE"].iloc[0]  # 获取 MAE。
                roughness = param_metrics_row["Roughness"].iloc[0]  # 获取粗糙度。
                metrics_text = (
                    f"相关系数: {corr:.2f}\n"
                    f"R2: {r2:.2f}\n"
                    f"RMSE: {rmse:.2f}\n"
                    f"MAE: {mae:.2f}\n"
                    f"粗糙度: {roughness:.2f}"
                )  # 格式化指标文本。
                plt.text(
                    0.97,
                    0.03,
                    metrics_text,
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    transform=plt.gca().transAxes,
                    fontsize=14,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8, edgecolor="gray"),
                )  # 在图上添加指标文本框。
            plt.tight_layout()  # 调整布局。
            safe_param_label = param_label.replace(" ", "_").replace(
                ":", "_"
            )  # 创建安全的文件名标签。
            filename = f"overall_reconstruction_scatter_{safe_param_label}.png"  # 生成文件名。
            filepath = os.path.join(output_folder, filename)  # 拼接文件路径。
            plt.savefig(filepath, dpi=300)  # 保存图形。
            plt.close()  # 关闭图形。
        else:
            print(
                f" 警告: 参数组合 {param_label} 没有生成任何有效的 '原始 vs. 重建' 对进行总体散点图绘图，跳过保存。"
            )  # 打印警告。
print("\n所有指定的随机像素重建测试完毕，所有图表和结果表格已保存到指定文件夹。")  # 打印完成信息。

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

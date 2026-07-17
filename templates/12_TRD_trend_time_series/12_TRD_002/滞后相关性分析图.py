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

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# --- Matplotlib 和字体配置 ---
matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# --- 配置信息 ---
excel_file_path = str(DATA_DIR / "滞后性分析.xlsx")
time_column_name = "date"  # 定义 Excel 文件中表示时间/日期的列的名称。请根据您的实际列名进行修改！
gst_column_name = "LST"  # 请根据您的实际列名进行修改！
remote_sensing_columns_map = {  # 定义一个字典，映射原始遥感数据列名到更易读的图表标签。请根据您的实际列名进行修改！
    "TD": "MOD-TD",
    "TN": "MOD-TN",
    "AD": "MYD-AD",
    "AN": "MYD-AN",
}
max_lag_days = (
    30  # 定义最大向前推移的天数（即遥感数据相对于LST最大滞后天数）。例如，分析0到30天的滞后效应。
)
# --- 加载数据 ---
print(f"正在加载数据: {excel_file_path}")

df = pd.read_excel(
    excel_file_path
)  # 使用 pandas 读取指定路径的 Excel 文件到 DataFrame 对象 df 中。
print("数据加载成功！")
# 转换时间列为 datetime 对象
if time_column_name not in df.columns:
    print(f"错误：数据中找不到时间列 '{time_column_name}'。请检查配置。")
    exit()

df[time_column_name] = pd.to_datetime(
    df[time_column_name]
)  # 将时间列（由 time_column_name 指定）的数据转换为 pandas 的 datetime 对象格式。
print(f"时间列 '{time_column_name}' 转换为 datetime 格式。")

# 按日期排序，以确保shift操作正确应用（如果数据尚未排序）
df = df.sort_values(by=time_column_name).reset_index(
    drop=True
)  # 按时间列对 DataFrame 进行升序排序，并重置索引（drop=True 表示不将旧索引作为新列）。
print("数据已按日期排序。")
# 检查关键列是否存在
if gst_column_name not in df.columns:
    print(f"错误：数据中找不到LST列 '{gst_column_name}'。请检查配置。")
    exit()
for rs_col_original in remote_sensing_columns_map.keys():
    if rs_col_original not in df.columns:
        print(f"警告：数据中找不到遥感列 '{rs_col_original}'，将跳过该列的分析。")
# --- 滞后分析与相关性计算 ---
lag_analysis_results = {}  # 初始化一个空字典，用于存储每个遥感数据集在不同滞后天数下的相关系数。键为遥感数据标签，值为包含滞后和相关系数的DataFrame。
optimal_lags_info = {}  # 初始化一个空字典，用于存储每个遥感数据集的最佳滞后天数和对应的最高相关系数。键为遥感数据标签，值为包含'lag'和'correlation'的字典。
print("\n开始滞后相关性分析...")
print(
    f"分析GST({gst_column_name})与遥感数据({list(remote_sensing_columns_map.keys())})，滞后范围 0 到 {max_lag_days} 天（遥感数据滞后于LST）。\n"
)  # 打印一条消息，提示分析开始和分析范围。
for (
    rs_col_original,
    rs_label,
) in (
    remote_sensing_columns_map.items()
):  # 遍历 remote_sensing_columns_map 字典中的每一对（原始列名，图表标签）。
    if rs_col_original not in df.columns:  # 跳过数据中不存在的列
        continue
    print(
        f"处理数据集: {rs_label} vs {gst_column_name}"
    )  # 打印当前正在处理的遥感数据集与LST的组合。
    correlations_for_current_rs = []  # 初始化一个空列表，用于临时存储当前遥感数据集在各个滞后天数下的相关系数。
    best_correlation_so_far = -2  # 初始化迄今为止的最佳相关系数为一个较小的值（小于-1的任何值均可，确保第一个有效相关系数能替换它）。
    optimal_lag_for_current_rs = 0  # 初始化当前遥感数据集的最佳滞后天数为0。
    for lag in range(max_lag_days + 1):  # 内层循环，遍历从0到 max_lag_days（包含）的所有滞后天数。
        # ***核心修改***：使用 shift(-lag) 将遥感数据向上平移 lag 天
        # 这使得原始日期 D 的GST数据与原始日期 D+lag 的遥感数据对齐
        # 从而分析遥感数据滞后于LST lag 天时的相关性
        shifted_rs_data = df[rs_col_original].shift(-lag)
        print(shifted_rs_data.head())
        temp_df_for_corr = pd.DataFrame(
            {  # 创建一个临时的 DataFrame，用于计算相关性。
                "gst": df[gst_column_name],  # LST列，使用原始LST数据（日期 D 的数据）
                "shifted_rs": shifted_rs_data,  # 移位后的遥感数据列（现在包含了原始日期 D+lag 的数据）
            }
        )
        temp_df_for_corr.dropna(
            inplace=True
        )  # 从临时 DataFrame 中移除任何包含 NaN（缺失值）的行。inplace=True 表示直接修改原 DataFrame。这样可以确保只对有完整数据对（LST和移位后遥感数据都存在）的行计算相关性。
        if len(temp_df_for_corr) < 3:  # 检查在移除NaN后，剩余的有效数据对数量是否少于3个。
            correlation = np.nan  # 如果数据点少于3个（通常计算相关性至少需要这么多点），则相关系数设为 NaN（Not a Number，表示无法计算）。
            print(f"  滞后 {lag} 天: 数据点不足 ({len(temp_df_for_corr)} < 3)，无法计算相关性。")
        else:  # 如果有足够的数据点。
            correlation = temp_df_for_corr["gst"].corr(
                temp_df_for_corr["shifted_rs"]
            )  # 计算 'gst' 列和 'shifted_rs' 列之间的皮尔逊相关系数。
            print(
                f"  滞后 {lag} 天: 相关系数 = {correlation:.4f}"
            )  # 可以打印每个滞后的相关性进行详细检查
        correlations_for_current_rs.append(
            {"lag": lag, "correlation": correlation}
        )  # 将当前的滞后天数 (lag) 和计算得到的相关系数 (correlation) 作为一个字典添加到列表中。
        # 注意：这里我们寻找相关系数的绝对值的最大值，因为最高相关性可能体现在强的正相关或强的负相关
        # 如果您只关注正相关，可以将 abs() 去掉
        if pd.notna(correlation) and correlation > best_correlation_so_far:
            best_correlation_so_far = correlation  # 更新最佳相关系数 (保留正负号)
            optimal_lag_for_current_rs = lag  # 更新最佳滞后天数
    lag_analysis_results[rs_label] = pd.DataFrame(
        correlations_for_current_rs
    )  # 将当前遥感数据集的所有滞后分析结果（滞后天数和对应相关系数列表）转换为 DataFrame，并以遥感数据标签为键存入 lag_analysis_results 字典。
    optimal_lags_info[
        rs_label
    ] = {  # 将当前遥感数据集的最佳滞后信息（滞后天数和最高相关系数）存入 optimal_lags_info 字典。
        "lag": optimal_lag_for_current_rs,  # 最佳滞后天数。
        "correlation": best_correlation_so_far
        if best_correlation_so_far > -2
        else np.nan,  # 对应的最高相关系数。如果 best_correlation_so_far 未被更新过（仍为初始值-2），则设为NaN。
    }
    # 打印当前遥感数据集的最佳滞后时间和对应的最大相关系数（保留4位小数）。
    if pd.notna(optimal_lags_info[rs_label]["correlation"]):
        print(
            f"  {rs_label}: 最佳滞后时间 = {optimal_lag_for_current_rs} 天, 最大相关系数 (绝对值) = {abs(optimal_lags_info[rs_label]['correlation']):.4f} (实际值: {optimal_lags_info[rs_label]['correlation']:.4f})\n"
        )
    else:
        print(f"  {rs_label}: 未能确定最佳滞后 (数据不足或无有效相关性)\n")
# --- 输出最佳滞后时间 ---
print("\n--- 各遥感数据相对于LST的最佳滞后时间总结 ---")  # 打印一个总结信息的标题。
for (
    rs_label,
    info,
) in optimal_lags_info.items():  # 遍历 optimal_lags_info 字典中存储的每个遥感数据集的最佳滞后信息。
    if pd.notna(info["correlation"]):  # 检查该数据集的最佳相关系数是否有效（不是 NaN）。
        print(
            f"{rs_label}: 滞后 {info['lag']} 天 (相关系数: {info['correlation']:.4f})"
        )  # 如果有效，打印遥感数据标签、最佳滞后天数和对应的相关系数。
    else:  # 如果相关系数无效。
        print(
            f"{rs_label}: 未能确定最佳滞后 (数据不足或无显著相关性)"
        )  # 打印未能确定最佳滞后的信息。
# --- 绘制相关系数 vs. 滞后天数图 ---
print("\n正在生成图表...")
fig, ax = plt.subplots(
    figsize=(14, 8)
)  # 创建一个新的图形 (fig) 和一个坐标轴对象 (ax)，并设置图形的大小为14x8英寸。
for (
    rs_label,
    corr_data_df,
) in lag_analysis_results.items():  # 遍历 lag_analysis_results 字典中每个遥感数据集的滞后分析结果。
    if (
        not corr_data_df.empty and pd.notna(corr_data_df["correlation"]).any()
    ):  # 仅当有有效相关性数据时才绘图
        ax.plot(
            corr_data_df["lag"],
            corr_data_df["correlation"],
            marker="o",
            linestyle="-",
            label=f"{rs_label} vs {gst_column_name}",
        )  # 在坐标轴上绘制相关系数随滞后天数变化的折线图。marker='o'表示用圆点标记数据点，linestyle='-'表示实线连接，label用于图例。
        opt_info = optimal_lags_info[rs_label]  # 获取当前遥感数据集的最佳滞后信息。
        if pd.notna(opt_info["correlation"]):  # 检查最佳相关系数是否有效。
            # 在图上用一个较大的红色圆点标记出最佳滞后点
            ax.scatter(opt_info["lag"], opt_info["correlation"], s=100, color="red", zorder=5)
            # 添加数值标注
            ax.annotate(
                f"{opt_info['correlation']:.2f}",
                (opt_info["lag"], opt_info["correlation"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )  # 设置标注文本的偏移量、对齐方式、字体大小和粗细。
ax.set_xlabel(
    f"{gst_column_name} 相对于遥感数据超前的天数 (天)", fontsize=14, fontweight="bold"
)  # 设置X轴的标签文本，描述的是LST相对于遥感数据的超前天数（等价于遥感数据滞后于LST的天数）。
ax.set_ylabel("皮尔逊相关系数", fontsize=14, fontweight="bold")  # 设置Y轴的标签文本。
ax.set_title(
    f"{gst_column_name} 与遥感数据滞后相关性分析", fontsize=16, fontweight="bold"
)  # 设置图表的标题。
# 设置X轴主刻度的间隔
if max_lag_days <= 10:
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
elif max_lag_days <= 40:
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
else:
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
# --- 刻度标签和图例美化 ---
# X轴刻度标签
ax.tick_params(axis="x", labelsize=12, labelrotation=0)  # 设置X轴刻度标签的属性。
plt.setp(ax.get_xticklabels(), fontweight="bold")  # 设置X轴刻度标签的字体为粗体。
# Y轴刻度标签
ax.tick_params(axis="y", labelsize=12)  # 设置Y轴刻度标签的属性。
plt.setp(ax.get_yticklabels(), fontweight="bold")  # 设置Y轴刻度标签的字体为粗体。
# 图例
# 确保图例中不包含scatter点重复生成的标签
handles, labels = ax.get_legend_handles_labels()
# 过滤掉重复的标签 (如果scatter自动生成了label的话) 或者手动创建图例handles和labels
# 更简单的做法是在 scatter 中不加 label，只在 plot 中加 label
legend = ax.legend(loc="best", frameon=True, fontsize=11, ncol=1)  # 创建并显示图例。
# 图表网格和边框美化
ax.grid(True, linestyle="--", alpha=0.7)  # 添加虚线网格线。
ax.spines["top"].set_visible(False)  # 隐藏顶部边框。
ax.spines["right"].set_visible(False)  # 隐藏右侧边框。
ax.spines["left"].set_linewidth(1.5)  # 设置左侧边框线宽。
ax.spines["bottom"].set_linewidth(1.5)  # 设置底部边框线宽。
plt.tight_layout()  # 自动调整布局，避免重叠。
print("图表已生成，正在显示...")
plt.close("all")  # Interactive display removed; assets were exported above.
# --- (可选) 保存图形 ---
fig.savefig(
    str(OUTPUT_DIR / "gst_lagged_correlation_analysis.png"), dpi=300
)  # 将图表保存为PNG图片。

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

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

import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import rasterio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 配置参数部分 ---
# 设置输出结果的文件夹路径
output_dir = str(OUTPUT_DIR)
# 确保输出目录存在，如果不存在则自动创建
os.makedirs(output_dir, exist_ok=True)
# 定义目标变量（因变量）的名称
target_var_name = "modlst"
# 定义特征变量（自变量）的名称列表
feature_var_names = ["ndvi", "ndwi", "nsc", "qw"]
# 为每个变量定义有效的数据范围，用于过滤异常值
valid_ranges = {
    "modlst": (-50, 50),
    "ndvi": (-1, 1),
    "ndwi": (-1, 1),
    "nsc": (0, 100),
    "qw": (-50, 50),
}
# 定义执行回归分析所需的最小有效观测次数（年份数）
min_observations_required = 10
# 定义每个变量所在的数据文件夹路径
variable_folder_paths = {
    "modlst": str(DATA_DIR),
    "ndvi": str(DATA_DIR),
    "ndwi": str(DATA_DIR),
    "nsc": str(DATA_DIR),
    "qw": str(DATA_DIR),
}


# --- 栅格数据读写工具函数 ---
# 定义一个函数，用于读取单个栅格文件
def read_raster(path):
    # 使用rasterio打开栅格文件
    with rasterio.open(path) as src:
        # 读取第一个波段的数据，并转换为32位浮点数类型
        data = src.read(1).astype(np.float32)
        # 获取文件的无效值（NoData value）
        nodata = src.nodata
        # 如果文件定义了无效值
        if nodata is not None:
            # 将数组中所有等于无效值的像元替换为NumPy的NaN（非数字），便于后续计算
            data[data == nodata] = np.nan
        # 返回读取到的数据数组和文件的元数据（坐标系、分辨率等）
        return data, src.meta


# 定义一个函数，用于将numpy数组写入到栅格文件中
def write_raster(path, array, meta):
    # 复制一份传入的元数据，以免修改原始元数据
    meta_copy = meta.copy()
    # 更新元数据信息，以适应输出文件的要求
    meta_copy.update(
        {
            "count": 1,  # 波段数为1
            "dtype": "float32",  # 数据类型为32位浮点数
            "nodata": np.nan,  # 使用NaN作为无效值
            "compress": "lzw",  # 使用LZW无损压缩算法，减小文件大小
        }
    )
    # 使用更新后的元数据，以写入模式('w')打开一个新文件
    with rasterio.open(path, "w", **meta_copy) as dst:
        # 将数组数据（同样转换为32位浮点数）写入文件的第一个波段
        dst.write(array.astype(np.float32), 1)


def save_array_preview(
    array,
    path,
    title,
    cmap="viridis",
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


# --- 文件匹配工具函数 ---
# 定义一个函数，用于根据变量名和年份查找对应的栅格文件
def find_variable_files(var_name_to_match, specific_variable_folder_path):
    # 创建一个空字典，用于存放找到的文件路径，以年份为键
    matched = {}
    # 创建一个正则表达式模式，用于匹配 "四位年份" + "变量名" + ".tif或.tiff" 的文件名，忽略大小写
    pattern = re.compile(r"(\d{4})" + re.escape(var_name_to_match) + r"\.(tif|tiff)", re.IGNORECASE)
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, _, files in os.walk(specific_variable_folder_path):
        # 遍历当前目录下的所有文件名
        for file in files:
            # 尝试用正则表达式对完整文件名进行匹配
            m = pattern.fullmatch(file)
            # 如果匹配成功
            if m:
                # 提取第一个捕获组，即四位年份
                date = m.group(1)
                # 将年份和对应的完整文件路径存入字典
                matched[date] = os.path.join(root, file)
    # 返回包含所有匹配文件信息的字典
    return matched


# --- 加载并过滤数据栈函数 ---
# 定义一个函数，用于加载一个变量所有年份的数据，并将其堆叠成一个三维数组
def load_variable_stack(var_name, paths_config, range_dict):
    # 检查变量的路径是否已在配置中定义
    if var_name not in paths_config:
        # 如果未定义，则抛出文件未找到的错误
        raise FileNotFoundError(f"未配置变量 '{var_name}' 路径")
    # 查找该变量对应的所有栅格文件
    files_map = find_variable_files(var_name, paths_config[var_name])
    # 获取所有年份并进行排序，确保数据按时间顺序加载
    dates = sorted(files_map.keys())
    # 创建一个空列表，用于存放每年读取的数据数组
    stack = []
    # 初始化元数据变量为空
    meta0 = None
    # 从范围字典中获取该变量的有效值最小值和最大值
    vmin, vmax = range_dict[var_name]
    # 使用tqdm显示加载进度条
    for d in tqdm(dates, desc=f"加载 {var_name}"):
        # 读取当前年份的栅格文件
        arr, meta = read_raster(files_map[d])
        # 创建一个布尔掩码，标记出所有不在有效值范围内的数据
        invalid = (arr < vmin) | (arr > vmax)
        # 将无效数据点赋值为NaN
        arr[invalid] = np.nan
        # 如果这是读取的第一个文件，保存其元数据作为参考
        if meta0 is None:
            meta0 = meta
        # 将处理后的二维数组添加到列表中
        stack.append(arr)
    # 检查是否成功加载了任何数据
    if not stack:
        # 如果没有加载到数据，则抛出值错误
        raise ValueError(f"'{var_name}' 没有加载到任何数据")
    # 使用numpy.stack将列表中的所有二维数组沿新轴（时间轴）堆叠成一个三维数组
    return np.stack(stack), dates, meta0


# --- 按行处理的函数 (核心计算逻辑) ---
# 定义一个函数，用于处理单行像元的数据回归
def process_row(args):
    # 从传入的参数元组中解包出所有需要的变量
    i, H, W, target, feats, data_dict, min_obs = args
    # 提取目标变量当前行所有年份的数据
    y_stack = data_dict[target][:, i, :]
    # 提取所有特征变量当前行所有年份的数据
    X_stacks = [data_dict[f][:, i, :] for f in feats]
    # 初始化用于存储该行回归结果的数组，默认填充为NaN
    intercept_r = np.full(W, np.nan)  # 截距
    r2_r = np.full(W, np.nan)  # R²
    rmse_r = np.full(W, np.nan)  # RMSE
    coef_r = {f: np.full(W, np.nan) for f in feats}  # 各特征的系数
    # 遍历当前行中的每一个像元
    for j in range(W):
        # 提取当前像元上所有年份的目标变量值
        y = y_stack[:, j]
        # 提取当前像元上所有年份的特征变量值，并堆叠成一个二维数组 (年份数 x 特征数)
        X = np.stack([X_stacks[k][:, j] for k in range(len(feats))], axis=1)
        # 确定哪些年份的数据是有效的（目标变量和所有特征变量都不是NaN）
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        # 如果有效观测点的数量少于要求的最小值，则跳过此像元
        if valid.sum() < min_obs:
            continue
        # 获取有效观测点对应的数据
        yv = y[valid]
        Xv = X[valid]
        # 如果目标变量或任何一个特征变量在所有有效观测点上的值都相同（标准差为0），无法进行回归，跳过
        if np.nanstd(yv) == 0 or any(np.nanstd(Xv[:, k]) == 0 for k in range(Xv.shape[1])):
            continue
        # 尝试执行回归计算，使用try-except以防止因数据问题导致程序崩溃
        try:
            # 创建一个标准化处理器
            scaler = StandardScaler()
            # 对特征变量数据进行标准化（使其均值为0，方差为1）
            Xv_scaled = scaler.fit_transform(Xv)
            # 创建并训练线性回归模型
            model = LinearRegression().fit(Xv_scaled, yv)
            # 使用训练好的模型进行预测
            pred = model.predict(Xv_scaled)
            # 保存模型的截距
            intercept_r[j] = model.intercept_
            # 计算并保存R²分数
            r2_r[j] = r2_score(yv, pred)
            # 计算并保存均方根误差 (RMSE)
            rmse_r[j] = np.sqrt(mean_squared_error(yv, pred))
            # 遍历并保存每个特征变量的回归系数
            for idx, f in enumerate(feats):
                coef_r[f][j] = model.coef_[idx]
        # 如果计算过程中出现任何异常
        except:
            # 跳过当前像元，结果将保持为NaN
            continue
    # 返回当前行的行号以及计算出的所有结果
    return i, intercept_r, r2_r, rmse_r, coef_r


# --- 用于主导因子分析的辅助函数 ---
# 定义一个辅助函数，用于查找主导因子的索引
def find_dominant_index(coef_slice):
    """
    安全地查找主导因子索引的函数。
    对于单个像元的系数数组（切片），如果所有值都是NaN，则返回NaN，
    否则返回绝对值最大的那个系数的索引。
    """
    # 检查系数数组中的所有值是否都是NaN
    if np.all(np.isnan(coef_slice)):
        # 如果是，则返回NaN
        return np.nan
    # 否则，返回数组中绝对值最大元素的索引
    return np.nanargmax(coef_slice)


# --- 主执行函数 ---
# 定义并行计算的主函数
def run_parallel():
    # 将目标变量和特征变量合并到一个列表中
    vars_all = [target_var_name] + feature_var_names
    # 创建一个空字典，用于存放所有变量原始的、未对齐的数据
    raw = {}
    # 创建一个空字典，用于存放每个变量对应的时间序列（年份）
    dates_map = {}
    # 初始化一个变量，用于存储参考的元数据
    meta_ref = None
    # 遍历所有需要处理的变量
    for v in vars_all:
        # 加载每个变量的时间序列数据
        data, dates, meta = load_variable_stack(v, variable_folder_paths, valid_ranges)
        # 将加载的数据存入raw字典
        raw[v] = data
        # 将对应的年份列表存入dates_map字典
        dates_map[v] = dates
        # 如果当前变量是目标变量，将其元数据作为后续输出的参考
        if v == target_var_name:
            meta_ref = meta
    print("正在查找所有输入栅格的最小公共维度...")
    # 首先，以第一个变量的维度作为初始值
    try:
        # 获取第一个变量的数据形状
        first_shape = raw[vars_all[0]].shape
        # 初始化最小高度和宽度
        min_h, min_w = first_shape[1], first_shape[2]
    except (IndexError, KeyError):
        # 如果无法获取维度信息，说明数据加载有问题，打印错误并退出
        print(
            f"!!! 错误: 无法获取变量 '{vars_all[0]}' 的维度信息，请检查数据是否正确加载。程序终止。"
        )
        return
    # 遍历所有加载的数据，找到真正的最小高度和宽度
    for v in vars_all:
        # 获取当前变量的高度和宽度
        h, w = raw[v].shape[1], raw[v].shape[2]
        # 如果当前高度更小，则更新最小高度
        if h < min_h:
            min_h = h
        # 如果当前宽度更小，则更新最小宽度
        if w < min_w:
            min_w = w
    print(f"已确定最小公共维度: 高={min_h}, 宽={min_w}。将以此为标准进行裁剪。")
    # 再次遍历，将所有数据裁剪到最小公共维度
    for v in vars_all:
        # 获取原始的高度和宽度
        original_h, original_w = raw[v].shape[1], raw[v].shape[2]
        # 仅当原始尺寸大于最小尺寸时才执行裁剪
        if original_h > min_h or original_w > min_w:
            print(f"  -> 正在裁剪变量 '{v}': 从 ({original_h}x{original_w}) -> ({min_h}x{min_w})")
            # 执行裁剪：取所有时间层，取从0到min_h的行，取从0到min_w的列
            raw[v] = raw[v][:, :min_h, :min_w]
    print("所有数据已裁剪至统一维度。")
    # -- 新增代码结束 ---
    # 找到所有变量共有的年份，首先用第一个变量的年份集合初始化
    common = set(dates_map[vars_all[0]])
    # 遍历其余变量，不断取年份的交集
    for v in vars_all[1:]:
        common &= set(dates_map[v])
    # 将最终的公共年份集合转换为列表并排序
    common = sorted(list(common))
    # 检查公共年份的数量是否满足最小观测要求
    if len(common) < min_observations_required:
        print(
            f"共同日期数量 ({len(common)}) 少于最小观测点数要求 ({min_observations_required})，程序退出。"
        )
        return
    # 创建一个新字典，用于存放按公共年份对齐后的数据
    aligned = {}
    # 遍历所有变量
    for v in vars_all:
        # 找到公共年份在当前变量年份列表中的索引
        idxs = [dates_map[v].index(d) for d in common]
        # 根据这些索引，从原始数据中提取出对应年份的数据层
        aligned[v] = raw[v][idxs]  # 此时raw中的数据已经是统一尺寸了
    # 获取对齐后数据的维度 (时间, 高, 宽)
    T, H, W = aligned[target_var_name].shape
    # 更新参考元数据中的高度和宽度信息以反映裁剪后的尺寸
    meta_ref["height"], meta_ref["width"] = H, W
    # 为每一行数据创建一个处理任务
    tasks = [
        (
            i,
            H,
            W,
            target_var_name,
            feature_var_names,
            aligned,
            max(len(feature_var_names) + 1, min_observations_required),
        )
        for i in range(H)
    ]
    # 优化并行进程数的设置
    # 获取CPU的核心数
    num_cores = multiprocessing.cpu_count()
    # 使用最多4个核心，或CPU核心数-1（留一个给系统），或图像行数中的最小值，且至少为1
    num_workers = max(1, min(num_cores - 1 if num_cores > 1 else 1, H, 4))
    print(f"启动 {num_workers} 个进程进行并行计算...")
    # 创建一个进程池执行器，这里将max_workers暂时设为1以进行调试或顺序执行，可改回num_workers
    with ProcessPoolExecutor(max_workers=1) as exe:
        # 将process_row函数应用到所有任务上，并用tqdm显示总进度
        results = list(tqdm(exe.map(process_row, tasks), total=H, desc="并行回归"))
    # 创建空的numpy数组，用于存储最终的整幅图像结果
    intercept_map = np.full((H, W), np.nan, np.float32)
    r2_map = np.full((H, W), np.nan, np.float32)
    rmse_map = np.full((H, W), np.nan, np.float32)
    coef_maps = {f: np.full((H, W), np.nan, np.float32) for f in feature_var_names}
    # 遍历从并行计算返回的结果列表
    for i, ib, rb, mb, cb in results:
        # 将每一行计算得到的结果赋值到总的地图数组的对应行上
        intercept_map[i] = ib
        r2_map[i] = rb
        rmse_map[i] = mb
        for f in feature_var_names:
            coef_maps[f][i] = cb[f]
    print("正在写入回归结果文件...")
    # 将截距图写入TIF文件
    write_raster(
        os.path.join(output_dir, f"{target_var_name}_intercept.tif"), intercept_map, meta_ref
    )
    # 将R²图写入TIF文件
    write_raster(os.path.join(output_dir, f"{target_var_name}_r2.tif"), r2_map, meta_ref)
    # 将RMSE图写入TIF文件
    write_raster(os.path.join(output_dir, f"{target_var_name}_rmse.tif"), rmse_map, meta_ref)
    # 遍历每个特征变量，将其系数图写入TIF文件
    for f in feature_var_names:
        write_raster(
            os.path.join(output_dir, f"{target_var_name}_coef_{f}.tif"), coef_maps[f], meta_ref
        )

    print("开始生成主导因子分析图...")
    # 创建一个字典，将每个特征变量名映射到一个唯一的整数索引（0, 1, 2, ...）
    feature_labels = {name: i for i, name in enumerate(feature_var_names)}
    # 将所有特征变量的系数图的绝对值堆叠成一个三维数组 (特征数, 高, 宽)
    abs_coef_stack = np.stack([np.abs(coef_maps[f]) for f in feature_var_names], axis=0)
    # 沿着特征轴（axis=0）对每个像元应用find_dominant_index函数
    # 这会生成一个二维数组，每个像元的值是其主导因子的索引
    dominant_factor_map = np.apply_along_axis(find_dominant_index, axis=0, arr=abs_coef_stack)
    # 定义主导因子图的输出路径
    dominant_factor_path = os.path.join(output_dir, f"{target_var_name}_dominant_factor.tif")
    # 将主导因子图写入TIF文件
    write_raster(dominant_factor_path, dominant_factor_map, meta_ref)
    save_array_preview(
        dominant_factor_map,
        os.path.join(output_dir, "preview.png"),
        "Dominant factor map",
        cmap="tab10",
        vmin=-0.5,
        vmax=len(feature_var_names) - 0.5,
        cbar_label="dominant factor",
        cbar_ticks=list(range(len(feature_var_names))),
        cbar_ticklabels=feature_var_names,
    )
    print("主导因子分析完成！")
    print(f"主导因子图已保存至: {dominant_factor_path}")
    # 打印图例说明，方便在GIS软件中查看和理解主导因子图
    print("图例说明: " + ", ".join([f"{name} = {label}" for name, label in feature_labels.items()]))
    print("所有处理完成！")


# 程序的入口点
if __name__ == "__main__":
    # 在Windows系统上，为了使多进程在打包成可执行文件后能正常工作，需要这行代码
    multiprocessing.freeze_support()
    # 调用主函数，开始执行整个分析流程
    run_parallel()

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

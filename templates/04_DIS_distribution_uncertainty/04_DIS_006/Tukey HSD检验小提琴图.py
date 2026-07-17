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
# ======================================1.库的导入=========================================
# =========================================================================================
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# =========================================================================================
# =====================================2.颜色库=========================================
# =========================================================================================
color_schemes = {
    0: [
        "#005f60",
        "#68d2d3",
        "#f89a94",
        "#d81e11",
        "#c71510",
        "#b60b0e",
        "#a5020d",
        "#94000b",
        "#83000a",
        "#720008",
    ],
    1: [
        "#5fad56",
        "#f2c14e",
        "#f78154",
        "#b4436c",
        "#a63e71",
        "#983976",
        "#8a347b",
        "#7c2f80",
        "#6e2a85",
        "#60258a",
    ],
    2: [
        "#2e4057",
        "#66a5ad",
        "#c4dfe6",
        "#f7d6e0",
        "#ffd1e0",
        "#ffcde0",
        "#ffc8e0",
        "#ffc4e0",
        "#ffc0e0",
        "#ffbbe0",
    ],
    3: [
        "#004e64",
        "#00a5cf",
        "#9fffcb",
        "#25a18e",
        "#199180",
        "#0e8172",
        "#027164",
        "#006156",
        "#005148",
        "#00413a",
    ],
    4: [
        "#3a0ca3",
        "#7209b7",
        "#f72585",
        "#4cc9f0",
        "#47b6d5",
        "#42a3bb",
        "#3d90a1",
        "#387d87",
        "#336a6d",
        "#2e5753",
    ],
    5: [
        "#1a535c",
        "#4ecdc4",
        "#f7fff7",
        "#ff6b6b",
        "#ff5a5a",
        "#ff4949",
        "#ff3838",
        "#ff2727",
        "#ff1616",
        "#ff0505",
    ],
    6: [
        "#d81159",
        "#8f2d56",
        "#218380",
        "#fbb13c",
        "#fcc239",
        "#fdd336",
        "#fee433",
        "#fff530",
        "#ffff2d",
        "#ffff2a",
    ],
    7: [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#f5a153",
        "#f6a045",
        "#f79f37",
        "#f89e29",
        "#f99d1b",
        "#fa9c0d",
    ],
    8: [
        "#e76f51",
        "#f4a261",
        "#e9c46a",
        "#2a9d8f",
        "#248e83",
        "#1e7f77",
        "#18706b",
        "#12615f",
        "#0c5253",
        "#064347",
    ],
    9: [
        "#03045e",
        "#0077b6",
        "#00b4d8",
        "#90e0ef",
        "#a3e4f2",
        "#b6e8f5",
        "#c9ecf8",
        "#dcf0fb",
        "#eff4fe",
        "#ffffff",
    ],
    10: [
        "#e63946",
        "#f1faee",
        "#a8dadc",
        "#457b9d",
        "#3d6c8d",
        "#355d7d",
        "#2d4e6d",
        "#253f5d",
        "#1d304d",
        "#15213d",
    ],
    11: [
        "#2d3142",
        "#bfc0c0",
        "#ffffff",
        "#ef8354",
        "#ee7644",
        "#ed6934",
        "#ec5c24",
        "#eb4f14",
        "#ea4204",
        "#e93500",
    ],
    12: [
        "#3d405b",
        "#81b29a",
        "#f2cc8f",
        "#e07a5f",
        "#dd6e53",
        "#da6247",
        "#d7563b",
        "#d44a2f",
        "#d13e23",
        "#ce3217",
    ],
    13: [
        "#ab3428",
        "#d7c9aa",
        "#6a8a82",
        "#2e464f",
        "#2a3e46",
        "#26363d",
        "#222e34",
        "#1e262b",
        "#1a1e22",
        "#161619",
    ],
    14: [
        "#ffc4d6",
        "#ffffff",
        "#b3e0ff",
        "#70b8ff",
        "#62a9f6",
        "#549aed",
        "#468be4",
        "#387cda",
        "#2a6dd1",
        "#1c5ec8",
    ],
    15: [
        "#335c67",
        "#fff3b0",
        "#e09f3e",
        "#9e2a2b",
        "#912728",
        "#842425",
        "#772122",
        "#6a1e1f",
        "#5d1b1c",
        "#501819",
    ],
    16: [
        "#540b0e",
        "#9e2a2b",
        "#e09f3e",
        "#fff3b0",
        "#fff4b9",
        "#fff5c2",
        "#fff6cb",
        "#fff7d4",
        "#fff8dd",
        "#fff9e6",
    ],
    17: [
        "#227c9d",
        "#17c3b2",
        "#ffcb77",
        "#fe6d73",
        "#fe5c66",
        "#fe4b59",
        "#fe3a4c",
        "#fe293f",
        "#fe1832",
        "#fe0725",
    ],
    18: [
        "#002642",
        "#840032",
        "#e59500",
        "#e5dada",
        "#e5d5d5",
        "#e5d0d0",
        "#e5cbcb",
        "#e5c6c6",
        "#e5c1c1",
        "#e5bcbc",
    ],
    19: [
        "#64113f",
        "#de4d86",
        "#ff9b54",
        "#266dd3",
        "#1e62c0",
        "#1657ac",
        "#0e4c99",
        "#064186",
        "#003673",
        "#002b60",
    ],
}

# =========================================================================================
# =====================================3.设置输入文件地址和输出结果路径=========================================
# =========================================================================================
input_excel_path = str(DATA_DIR / "simulated_data.xlsx")  # 输入文件
output_dir = str(OUTPUT_DIR)  # 输出地址
os.makedirs(output_dir, exist_ok=True)
# 选择颜色
color_scheme = 0
FULL_EXPORTS = _os.environ.get("MODELVIZ_FULL_SEARCH", "0") == "1"


# =========================================================================================
# ======================================4.单张小提琴图绘制函数=========================================
# =========================================================================================
def create_violin_plot(df, labels, plot_label, output_filepath):
    fig, ax = plt.subplots(figsize=(12, 8))
    # 从颜色库获取选定的颜色列表，默认使用第一套
    selected_colors = color_schemes.get(color_scheme, color_schemes[0])
    plot_order = ["CB", "DL", "DH", "XSBN"]  # 定义一个列表，指定小提琴图在x轴上的显示顺序。
    # 将颜色和分组名一一对应，创建调色板字典
    palette_colors = dict(zip(plot_order, selected_colors))

    sns.violinplot(  # 使用seaborn库的violinplot函数来绘制小提琴图。
        x="Group",  # x轴的数据来源
        y="Value",  # y轴的数据来源。
        data=df,  # 绘图所用的数据
        ax=ax,  # 在哪个坐标轴对象上进行绘图
        order=plot_order,  # 排列x轴
        palette=palette_colors,  # 颜色
        inner=None,  # 提琴内部不显示任何标记点或四分位数线
        linewidth=3,  # 小提琴图轮廓线的宽度
        hue="Group",  # 根据Group列对小提琴进行着色，确保颜色与分组正确对应
        legend=False,  # 不显示图例
    )

    sns.boxplot(  # 在小提琴图的内部叠加一个箱线图
        x="Group",  # x轴的数据来源
        y="Value",  # y轴的数据来源
        data=df,  # 指定绘图所用的数据
        ax=ax,  # 在同一个坐标轴对象上绘制
        order=plot_order,  # 确保箱线图的顺序与小提琴图一致
        width=0.15,  # 设置箱线图的宽度
        boxprops={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 2.5,
        },  # 设置箱体，白色填充，黑色边框，边框线宽
        medianprops={"color": "black", "linewidth": 3.5},  # 设置中位线，颜色，线宽
        whiskerprops={"color": "black", "linewidth": 2.5},  # 设置须线的属性，颜色，线宽
        showcaps=False,  # 不显示末端的横线
        showfliers=False,  # 不显示异常值点
    )

    y_max = df.groupby("Group")["Value"].max()  # 对数据进行分组，并计算每列的最大值。
    y_pos = y_max * 1.2  # 显著性字母标注的y轴位置

    for i, group in enumerate(plot_order):  # 遍历组名列表，同时获取索引和组名
        # 添加显著性字母
        ax.text(
            i,  # x轴位置，i为0, 1, 2, 3，对应每个组的中心
            y_pos[group],  # y轴位置，使用之前计算好的y_pos。
            labels[group],  # 显著性字母
            ha="center",  # 水平对齐方式
            va="bottom",  # 垂直对齐方式
            fontweight="bold",  # 字体粗细
            fontsize=22,  # 字体大小
        )

    for spine in ax.spines.values():  # 遍历边框
        spine.set_linewidth(2)  # 设置边框的粗细

    ax.tick_params(  # 自定义坐标轴刻度的样式
        axis="both",  # 应用于x轴和y轴
        which="major",  # 应用于主刻度线
        direction="in",  # 设置刻度线朝向图内
        length=6,  # 设置刻度线的长度
        width=2,  # 设置刻度线的宽度
    )

    ax.set_xlabel("")  # 显示x轴标题
    ax.set_ylabel(
        "Leaf N concentration (mg g⁻¹)", fontsize=28, fontweight="bold"
    )  # 设置y轴的标题、大小、粗细

    for label in ax.get_xticklabels():  # 遍历y轴的所有刻度线的数值标注
        label.set_fontsize(28)  # 设置y轴数值标注大小
        label.set_fontweight("bold")  # 设置y轴数值标注的粗细
    for label in ax.get_yticklabels():  # 遍历x轴的所有刻度线的数值标注
        label.set_fontsize(28)  # 设置x轴数值标注大小
        label.set_fontweight("bold")  # 设置x轴数值标注的粗细

    max_annotation_y = y_pos.max()  # 找到所有显著性标注位置中的最大y值。
    top_limit = max_annotation_y * 1.25  # 计算y轴的上限

    current_bottom_limit, _ = ax.get_ylim()  # 获取当前图形自动计算的y轴范围，并只保留下限值。

    ax.set_ylim(bottom=current_bottom_limit, top=top_limit)  # 设置y轴的范围

    # 在图的左上角添加子图标签，如a、b、c
    ax.text(
        0.03,
        0.9,  # 指定文本位置的x和y坐标
        plot_label,  # 要显示的具体文本内容
        transform=ax.transAxes,  # 指定坐标系
        fontsize=28,  # 字体大小
        fontweight="bold",  # 字体粗细
    )  # text函数调用结束。
    plt.tight_layout()
    # 保存
    plt.savefig(output_filepath, dpi=300)
    pdf_filepath = output_filepath.replace(".png", ".pdf")
    if FULL_EXPORTS:
        plt.savefig(pdf_filepath)
        print(f"图形已保存为 '{output_filepath}' 和 '{pdf_filepath}'")
    else:
        print(f"图形已保存为 '{output_filepath}'")
    plt.close(fig)


# =========================================================================================
# ======================================5. 图像拼接函数=========================================
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
        img.close()  # 关闭已粘贴的图片对象，释放内存。

    png_path_composite = os.path.join(
        save_dir, f"{output_filename_base}.png"
    )  # 使用os.path.join构造跨平台的PNG文件保存路径。
    composite_image.save(png_path_composite)  # 将拼接好的大图保存为PNG格式。
    print(f"\n组合图已保存为 '{png_path_composite}'")  # 在控制台打印PNG保存成功的提示。

    pdf_path_composite = os.path.join(
        save_dir, f"{output_filename_base}.pdf"
    )  # 构造PDF文件的保存路径。
    if composite_image.mode == "RGBA":  # 检查图像模式，如果是有透明通道的'RGBA'。
        composite_image = composite_image.convert(
            "RGB"
        )  # 则将其转换为'RGB'模式，因为PDF不支持透明度。
    if FULL_EXPORTS:
        composite_image.save(pdf_path_composite)  # 将拼接好的大图保存为PDF格式。
        print(f"组合图已保存为 '{pdf_path_composite}'")
    composite_image.close()


# =========================================================================================
# ======================================6.执行部分=========================================
# =========================================================================================
if __name__ == "__main__":
    all_sheets_data = pd.read_excel(input_excel_path, sheet_name=None)  # 读取输入数据的所有表
    print(f"成功从 '{input_excel_path}' 读取了 {len(all_sheets_data)} 个工作表。")

    generated_image_paths = []  # 初始化一个空列表，用于存储所有生成子图的文件路径，用于后续的组合图
    # 遍历读取的每一个表
    for i, (sheet_name, df) in enumerate(all_sheets_data.items()):
        print(f"正在处理表 '{sheet_name}'")
        # 执行Tukey's HSD检验
        tukey_result = pairwise_tukeyhsd(
            endog=df["Value"],  # 指定要进行检验的因变量，观测值
            groups=df["Group"],  # 指定分组变量，用于定义要比较的组
            alpha=0.05,  # 设置显著性水平
        )  # 函数调用结束。
        print(f"表 '{sheet_name}' 的 Tukey's HSD 检验结果")
        print(tukey_result)
        # 自动生成显著性标注
        results_df = pd.DataFrame(
            data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0]
        )  # 将检验结果的表格部分转换为dataFrame格式
        all_groups = sorted(
            df["Group"].unique().tolist()
        )  # 获取所有不重复的组名，并按字母顺序排序。
        labels = {
            group: "" for group in all_groups
        }  # 初始化一个字典，用于存放每个组的显著性字母，默认为空字符串。
        current_letter_code = ord("a")  # 初始化字母的ASCII码，从'a'开始。
        mean_values = (
            df.groupby("Group")["Value"].mean().sort_values(ascending=False)
        )  # 按组计算均值，并按均值从高到低对组进行排序。
        sorted_groups = mean_values.index.tolist()  # 获取排序后的组名列表。
        labels[sorted_groups[0]] = "a"  # 将显著性字母'a'分配给均值最高的分组。

        for j in range(1, len(sorted_groups)):  # 遍历从第二个分组开始的每一个组。
            current_group = sorted_groups[j]  # 获取当前正在处理的组。
            for k in range(j):  # 遍历所有在当前组之前、已经分配了字母的组。
                prev_group = sorted_groups[k]  # 获取一个排在前面的组。
                comparison = results_df[  # 在检验结果中查找当前组与前面组的比较记录。
                    (
                        (results_df["group1"] == current_group)
                        & (results_df["group2"] == prev_group)
                    )  # 条件1：group1是当前组，group2是前面组
                    | (
                        (results_df["group1"] == prev_group)
                        & (results_df["group2"] == current_group)
                    )
                    # 条件2：group1是前面组，group2是当前组
                ]  # 查询结束。
                reject = comparison["reject"].iloc[
                    0
                ]  # 获取比较结果中的'reject'列的值（True表示差异显著，False表示不显著）。
                if not reject:  # 如果两组之间没有显著差异。
                    labels[current_group] += labels[
                        prev_group
                    ]  # 则当前组继承前面那个无差异组的所有字母。
            is_different_from_all_prev = all(  # 使用all()函数检查一个条件是否对所有元素都成立。
                results_df[  # 这部分逻辑同上，再次查找比较结果。
                    (
                        (results_df["group1"] == current_group)
                        & (results_df["group2"] == sorted_groups[k])
                    )
                    | (
                        (results_df["group1"] == sorted_groups[k])
                        & (results_df["group2"] == current_group)
                    )
                ]["reject"].iloc[0]
                for k in range(j)  # 对所有前面的组(k)进行遍历检查，看是否都存在显著差异。
            )
            if is_different_from_all_prev:  # 如果当前组与所有前面的组都存在显著差异。
                current_letter_code += 1  # 字母的ASCII码加1，准备分配下一个字母（如'b'）。
                labels[current_group] = chr(
                    current_letter_code
                )  # 将新的字母分配给当前组（chr()将ASCII码转回字符）。
            labels[current_group] = "".join(
                sorted(set(labels[current_group]))
            )  # 对当前组获得的所有字母进行去重和排序

        print("\n显著性标注:", labels)

        # 调用绘图函数生成单个子图
        plot_label = f"({chr(ord('a') + i)})"  # 根据循环次数生成子图标签，如 (a), (b), (c)
        subplot_filename = f"{sheet_name}_{color_scheme}.png"  # 命名输出的子图
        output_filepath = os.path.join(output_dir, subplot_filename)  # 构造子图的完整保存路径。
        create_violin_plot(
            df, labels, plot_label, output_filepath
        )  # 调用之前封装好的绘图函数，生成并保存子图。
        generated_image_paths.append(
            output_filepath
        )  # 将生成好的子图的完整路径添加到列表中，以备后续拼接使用

    # 调用拼接函数
    stitch_images_grid(
        image_paths=generated_image_paths,  # 传入所有子图的路径列表。
        n_cols=2,  # 组合图的列数
        output_filename_base="composite_figure",  # 组合图的文件名
        save_dir=output_dir,  # 组合图的保存路径
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

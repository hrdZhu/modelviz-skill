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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats

plt.rcParams["font.family"] = "Times New Roman"
# =========================================================================================
# ======================================2.颜色库设置=========================================
# =========================================================================================
COLOR_THEMES = {
    1: {
        "group_colors": {
            "TFs": "#3B5F8A",
            "VIs": "#E68B3A",
            "DVS": "#707070",
            "PI": "#A65E60",
            "MC": "#8A6B94",
        },
        "heatmap_colors": ["#43A8A8", "white", "#D75F5F"],
    },
    2: {
        "group_colors": {
            "TFs": "#2E7D32",
            "VIs": "#FF8F00",
            "DVS": "#616161",
            "PI": "#8D6E63",
            "MC": "#6A1B9A",
        },
        "heatmap_colors": ["#388E3C", "white", "#D32F2F"],
    },
    3: {
        "group_colors": {
            "TFs": "#4A148C",
            "VIs": "#AD1457",
            "DVS": "#455A64",
            "PI": "#C2185B",
            "MC": "#004D40",
        },
        "heatmap_colors": ["#1976D2", "#FFFFFF", "#E91E63"],
    },
    4: {
        "group_colors": {
            "TFs": "#003B46",
            "VIs": "#07575B",
            "DVS": "#66A5AD",
            "PI": "#007580",
            "MC": "#0A918C",
        },
        "heatmap_colors": ["#0D47A1", "#FFFFFF", "#FFC107"],
    },
    5: {
        "group_colors": {
            "TFs": "#6D4C41",
            "VIs": "#BF360C",
            "DVS": "#795548",
            "PI": "#D84315",
            "MC": "#4E342E",
        },
        "heatmap_colors": ["#8D6E63", "#FFF3E0", "#E64A19"],
    },
    6: {
        "group_colors": {
            "TFs": "#42275A",
            "VIs": "#734B6D",
            "DVS": "#E86A92",
            "PI": "#F47983",
            "MC": "#6A1B9A",
        },
        "heatmap_colors": ["#311B92", "#FFECB3", "#E65100"],
    },
    7: {
        "group_colors": {
            "TFs": "#1B5E20",
            "VIs": "#2E7D32",
            "DVS": "#558B2F",
            "PI": "#7CB342",
            "MC": "#3E2723",
        },
        "heatmap_colors": ["#2E7D32", "#F1F8E9", "#BF360C"],
    },
    8: {
        "group_colors": {
            "TFs": "#1C3144",
            "VIs": "#30475E",
            "DVS": "#7E8A97",
            "PI": "#5C6F7F",
            "MC": "#046582",
        },
        "heatmap_colors": ["#263238", "#ECEFF1", "#0277BD"],
    },
    9: {
        "group_colors": {
            "TFs": "#311B92",
            "VIs": "#4527A0",
            "DVS": "#7B1FA2",
            "PI": "#9C27B0",
            "MC": "#F9A825",
        },
        "heatmap_colors": ["#4A148C", "#FFF8E1", "#FFAB00"],
    },
    10: {
        "group_colors": {
            "TFs": "#212121",
            "VIs": "#424242",
            "DVS": "#757575",
            "PI": "#9E9E9E",
            "MC": "#616161",
        },
        "heatmap_colors": ["#000000", "#FFFFFF", "#BDBDBD"],
    },
    11: {
        "group_colors": {
            "TFs": "#A2D0C1",
            "VIs": "#E2A0B7",
            "DVS": "#F6E5A8",
            "PI": "#B1A8D0",
            "MC": "#FFB085",
        },
        "heatmap_colors": ["#81C784", "#FFFFFF", "#CE93D8"],
    },
    12: {
        "group_colors": {
            "TFs": "#440154",
            "VIs": "#3B528B",
            "DVS": "#21908C",
            "PI": "#5DC863",
            "MC": "#FDE725",
        },
        "heatmap_colors": ["#0D0887", "#FFFFFF", "#F0F921"],
    },
    13: {
        "group_colors": {
            "TFs": "#93634B",
            "VIs": "#C88C5A",
            "DVS": "#E3B27D",
            "PI": "#A5492A",
            "MC": "#5D3A2D",
        },
        "heatmap_colors": ["#A1887F", "#FFFFFF", "#42A5F5"],
    },
    14: {
        "group_colors": {
            "TFs": "#009FB7",
            "VIs": "#FED766",
            "DVS": "#247BA0",
            "PI": "#F25F5C",
            "MC": "#70C1B3",
        },
        "heatmap_colors": ["#005969", "#FFFFFF", "#FF427E"],
    },
    15: {
        "group_colors": {
            "TFs": "#588157",
            "VIs": "#84A98C",
            "DVS": "#A3B18A",
            "PI": "#B38A58",
            "MC": "#3A5A40",
        },
        "heatmap_colors": ["#344E41", "#FFFFFF", "#BC6C25"],
    },
    16: {
        "group_colors": {
            "TFs": "#C36B84",
            "VIs": "#E69597",
            "DVS": "#F5B9B2",
            "PI": "#813745",
            "MC": "#59292F",
        },
        "heatmap_colors": ["#880E4F", "#FCE4EC", "#4E342E"],
    },
    17: {
        "group_colors": {
            "TFs": "#0B0C10",
            "VIs": "#1F2833",
            "DVS": "#45A29E",
            "PI": "#66FCF1",
            "MC": "#C5C6C7",
        },
        "heatmap_colors": ["#00003f", "#FFFFFF", "#7A0099"],
    },
    18: {
        "group_colors": {
            "TFs": "#4878CF",
            "VIs": "#6ACC65",
            "DVS": "#D65F5F",
            "PI": "#B47CC7",
            "MC": "#DC843D",
        },
        "heatmap_colors": ["#4C72B0", "#FFFFFF", "#DD8452"],
    },
    19: {
        "group_colors": {
            "TFs": "#F4B71B",
            "VIs": "#FA7921",
            "DVS": "#E55934",
            "PI": "#9BC53D",
            "MC": "#5BC0EB",
        },
        "heatmap_colors": ["#588157", "#FFFFFF", "#F9A620"],
    },
    20: {
        "group_colors": {
            "TFs": "#2D2D2A",
            "VIs": "#595954",
            "DVS": "#83837E",
            "PI": "#B46549",
            "MC": "#988880",
        },
        "heatmap_colors": ["#36454F", "#FFFFFF", "#B87333"],
    },
    21: {
        "group_colors": {
            "TFs": "#1A425D",
            "VIs": "#546A7B",
            "DVS": "#8C9BAB",
            "PI": "#C2CCDA",
            "MC": "#2B2D42",
        },
        "heatmap_colors": "coolwarm",
    },
    22: {
        "group_colors": {
            "TFs": "#440154",
            "VIs": "#31688E",
            "DVS": "#21918C",
            "PI": "#35B779",
            "MC": "#1F4060",
        },
        "heatmap_colors": "viridis",
    },
    23: {
        "group_colors": {
            "TFs": "#0D0887",
            "VIs": "#5B02A3",
            "DVS": "#9B179E",
            "PI": "#CC4678",
            "MC": "#4B0395",
        },
        "heatmap_colors": "plasma",
    },
    24: {
        "group_colors": {
            "TFs": "#000004",
            "VIs": "#57106E",
            "DVS": "#9C179E",
            "PI": "#ED6925",
            "MC": "#420A68",
        },
        "heatmap_colors": "inferno",
    },
    25: {
        "group_colors": {
            "TFs": "#000004",
            "VIs": "#3B0F70",
            "DVS": "#8C2981",
            "PI": "#DE4968",
            "MC": "#6A217C",
        },
        "heatmap_colors": "magma",
    },
    26: {
        "group_colors": {
            "TFs": "#00204E",
            "VIs": "#00528A",
            "DVS": "#4080A5",
            "PI": "#7AB0C2",
            "MC": "#2F4858",
        },
        "heatmap_colors": "cividis",
    },
    27: {
        "group_colors": {
            "TFs": "#053061",
            "VIs": "#2166AC",
            "DVS": "#92C5DE",
            "PI": "#67001F",
            "MC": "#4393C3",
        },
        "heatmap_colors": "bwr",
    },
    28: {
        "group_colors": {
            "TFs": "#003C30",
            "VIs": "#01665E",
            "DVS": "#5AB4AC",
            "PI": "#8E0152",
            "MC": "#35978F",
        },
        "heatmap_colors": "seismic",
    },
    29: {
        "group_colors": {
            "TFs": "#B2182B",
            "VIs": "#D6604D",
            "DVS": "#F4A582",
            "PI": "#4393C3",
            "MC": "#2166AC",
        },
        "heatmap_colors": "RdBu_r",
    },
    30: {
        "group_colors": {
            "TFs": "#00008F",
            "VIs": "#0090FF",
            "DVS": "#00FFFF",
            "PI": "#FFFF00",
            "MC": "#FF0000",
        },
        "heatmap_colors": "jet",
    },
    31: {
        "group_colors": {
            "TFs": "#23171B",
            "VIs": "#4D3C5D",
            "DVS": "#6A65A3",
            "PI": "#6895D1",
            "MC": "#53A196",
        },
        "heatmap_colors": "turbo",
    },
    32: {
        "group_colors": {
            "TFs": "#9400D3",
            "VIs": "#0000FF",
            "DVS": "#00FF00",
            "PI": "#FFFF00",
            "MC": "#FF0000",
        },
        "heatmap_colors": "gist_rainbow",
    },
    33: {
        "group_colors": {
            "TFs": "#00425A",
            "VIs": "#1F8A70",
            "DVS": "#BFDB38",
            "PI": "#FC7300",
            "MC": "#005B7F",
        },
        "heatmap_colors": "ocean",
    },
    34: {
        "group_colors": {
            "TFs": "#346B31",
            "VIs": "#779F63",
            "DVS": "#C4D69B",
            "PI": "#A49580",
            "MC": "#567D46",
        },
        "heatmap_colors": "terrain",
    },
    35: {
        "group_colors": {
            "TFs": "#2D274E",
            "VIs": "#4A4C7D",
            "DVS": "#6A74A8",
            "PI": "#8D9DD1",
            "MC": "#3A3A69",
        },
        "heatmap_colors": "cubehelix",
    },
    36: {
        "group_colors": {
            "TFs": "#000000",
            "VIs": "#550000",
            "DVS": "#AA0000",
            "PI": "#FF5500",
            "MC": "#FFAA00",
        },
        "heatmap_colors": "gnuplot",
    },
    37: {
        "group_colors": {
            "TFs": "#FADADD",
            "VIs": "#F9C4D2",
            "DVS": "#F7AEC1",
            "PI": "#F38FB8",
            "MC": "#E26D9F",
        },
        "heatmap_colors": "spring",
    },
    38: {
        "group_colors": {
            "TFs": "#C4DFBC",
            "VIs": "#A9D3B2",
            "DVS": "#8DC6A9",
            "PI": "#65B49F",
            "MC": "#4BA296",
        },
        "heatmap_colors": "summer",
    },
    39: {
        "group_colors": {
            "TFs": "#4B120B",
            "VIs": "#7B1B0B",
            "DVS": "#AF2B0B",
            "PI": "#E3440B",
            "MC": "#63150B",
        },
        "heatmap_colors": "autumn",
    },
    40: {
        "group_colors": {
            "TFs": "#0F3C58",
            "VIs": "#1E526C",
            "DVS": "#316C85",
            "PI": "#4989A1",
            "MC": "#255E7A",
        },
        "heatmap_colors": "winter",
    },
}


# =========================================================================================
# ======================================3.绘图函数=========================================
# =========================================================================================
def create_correlation_circos(df, df_sig, stages, groups, color_palette):
    # 提取特征分组的颜色
    group_colors = color_palette["group_colors"]
    # 提取热图的颜色）
    heatmap_colors_value = color_palette["heatmap_colors"]

    fig, ax = plt.subplots(figsize=(16, 14), subplot_kw=dict(projection="polar"))
    # 特征的数量
    N_features = len(df)
    # 用于绘图部分所占的区域，加上一个数是为了制造出一个缺口
    arc_length = 2 * np.pi * (N_features / (N_features + 2.5))

    single_bar_span = arc_length / N_features  # 每个特征所占的角度
    width = single_bar_span * 0.98  # 设置每个条形的宽度

    start_angle = (np.pi / 2) + (single_bar_span / 2)  # 第一个条形开始的角度
    end_angle = start_angle + (N_features - 1) * single_bar_span  # 最后一个条形结束的角度

    theta = np.linspace(start_angle, end_angle, N_features)  # 生成每个条形中心点的角度
    radii = np.arange(5, 5 + len(stages))  # 生成每层的半径
    category_radius = radii[0] - 1  # 表示特征分组的那一圈的半径

    # 判断热图颜色配置是Matplotlib自带的还是自定义的
    if isinstance(heatmap_colors_value, str):  # 如果是字符串，直接从matplotlib获取预设的颜色映射
        cmap = plt.get_cmap(heatmap_colors_value)
    else:
        # 创建自定义的颜色映射
        colors = heatmap_colors_value
        # 定义颜色列表在色带上的位置
        nodes = [0.0, 0.5, 1.0]
        # 根据节点和颜色创建分段线性颜色映射
        cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(nodes, colors)))

    # 创建一个归一化对象，将相关性系数值从映射到[0, 1]区间，便于颜色映射
    norm = plt.Normalize(vmin=-1, vmax=1)

    # 遍历每个目标变量，就是绘图的时候的每一层的圆环
    for i, stage in enumerate(stages):
        r = radii[i]  # 当前环的半径
        # 获取当前目标与所有特征的相关性数值
        values = df[stage]
        # 将相关性值通过归一化和颜色映射转换成具体的颜色
        cell_colors = cmap(norm(values))

        # 绘制条形图，形成热图的一层
        ax.bar(
            theta,  # 每个条形的中心角度位置
            height=0.9,  # 每个条形在径向上的高度
            width=width,  # 每个条形的角度宽度
            bottom=r,  # 每个条形的起始半径位置
            color=cell_colors,  # 填充颜色
            edgecolor="white",  # 边框颜色
            linewidth=0.5,
        )  # 条形边框的线宽

        # 遍历当前层的每个相关性数值
        for j, val in enumerate(values):
            angle = theta[j]  # 获取当前单元格的角度
            radius = r + 0.45  # 相关性标注的半径位置
            sig_marker = "*" if df_sig.iloc[j, i] else ""  # 设置显著性的标记
            text_val = f"{val:.2f}{sig_marker}"  # 设置相关性的标注文本

            rot = np.rad2deg(angle)  # 将角度从弧度转换为度
            rot -= 90  # 设置文本的的旋转角度，现在是与径向垂直

            # 在每一个小块上面加上标注
            ax.text(
                angle,  # 角度
                radius,  # 半径
                text_val,  # 标注的内容
                ha="center",  # 水平对齐
                va="center",  # 垂直对齐
                fontsize=11,  # 大小
                rotation=rot,  # 旋转角度
                rotation_mode="anchor",
            )  # 旋转模式，先旋转文本再进行对齐

    # 为每个特征确定其所属分组的颜色，如果找不到则使用默认的颜色
    category_colors = [
        next((group_colors[g] for g, m in groups.items() if f in m), "#CCCCCC") for f in df.index
    ]
    # 绘制最内圈的特征分组的颜色块
    ax.bar(
        theta,
        height=0.9,
        width=width,
        bottom=category_radius,
        color=category_colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # 计算最外圈特征名称标注的半径
    label_radius = radii[-1] + 1.2
    # 遍历每个特征名称，加上标注
    for i, feature_name in enumerate(df.index):
        angle = theta[i]  # 当前特征的角度
        rot = np.rad2deg(angle)  # 将角度从弧度转换为度
        rot -= 90  # 调整特征名称的旋转角度，这里要注意就是特征名称最好不要太长，要不然的画就是会很直，我现在也不会那个弄出一点弯曲的那种效果，就是特征名称短一点会好看

        # 在图上添加特征名称标签
        ax.text(
            angle,
            label_radius,
            feature_name,
            ha="center",
            va="center",
            fontsize=12.5,
            rotation=rot,
            rotation_mode="anchor",
        )

    # 创建图例
    legend_elements = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=name)
        for name, color in group_colors.items()
    ]
    # 在圆环的中心上面加上图例
    ax.legend(
        handles=legend_elements,  # 图例中的图形元素列表
        loc="center",  # 位置
        frameon=False,  # 边框
        fontsize=12,  # 字体大小
        title="Feature Types",  # 标题
        labelspacing=0.8,  # 图例中元素的垂直间隔
        title_fontproperties={"weight": "bold", "size": 14},
    )  # 标题的字体设置

    ax.grid(False)  # 去掉自带的网格线就是圆环和径向线
    ax.spines["polar"].set_visible(False)  # 去掉最外面的圆框
    ax.set_yticklabels([])  # 去掉半径方向的标签
    ax.set_xticklabels([])  # 去掉角度的标签

    ax.set_ylim(0, label_radius + 1)  # 设置半径的显示范围
    north_angle = np.pi / 2  # 定义正上方的角度，用来添加目标变量的标注

    # 直接使用传入的目标列表作为环的标签
    label_stages = stages
    # 标注的半径位置
    label_radii = radii + 0.45
    # 遍历每个环的标签
    for i, label in enumerate(label_stages):
        # 在每个环的起始位置添加其对应的目标的标注
        ax.text(
            north_angle,
            label_radii[i],
            f" {label}",
            ha="left",
            va="center",
            fontsize=12,
            rotation=0,
        )
    # 添加显著性标注
    ax.text(
        north_angle, radii[0] - 0.55, " * p < 0.05", ha="left", va="center", fontsize=12, rotation=0
    )

    # 在画布上添加一个新的坐标轴，用于放置颜色条
    cax = fig.add_axes([0.85, 0.2, 0.01, 0.6])
    # 创建一个可映射标量的对象，包含了颜色映射和归一化规则
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    # 在指定的坐标轴上创建颜色条
    cbar = fig.colorbar(sm, cax=cax)
    # 设置颜色条的标题
    cbar.set_label(f"{correlation_method.capitalize()} coefficient", size=12, labelpad=10)

    # 结果保存
    png_filename = str(OUTPUT_DIR / f"{selected_scheme}.png")
    pdf_filename = str(OUTPUT_DIR / f"{selected_scheme}.pdf")
    plt.savefig(png_filename, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # =========================================================================================
    # ======================================4.分析绘图前准备=========================================
    # =========================================================================================
    excel_file_path = str(DATA_DIR / "data.xlsx")  # 原始数据

    # 选择相关性分析方法，pearson,spearman,kendall
    correlation_method = "kendall"
    # 选择配色方案
    selected_scheme = 32
    # 从颜色库里面提取配色方案
    select_color = COLOR_THEMES.get(selected_scheme, 1)
    # 读取数据
    all_data = pd.read_excel(excel_file_path)

    # 提取特征数据
    final_features = all_data.columns[0:16].tolist()
    # 提取目标数据
    targets = all_data.columns[16:].tolist()
    # 将特征数据进行分组
    groups = {
        "MC": ["Plant height"],
        "TFs": [
            "band 5 contrast",
            "band 5 mean",
            "band 1 variance",
            "band 2 contrast",
            "band 5 dissimilarity",
            "band 4 contrast",
            "band 5 variance",
            "band 3 variance",
        ],
        "VIs": ["SIPI", "RVI", "MCARI", "LCI", "EVI2"],
        "DVS": ["DVS"],
        "PI": ["PI"],
    }

    # =========================================================================================
    # ======================================5.相关性分析=========================================
    # =========================================================================================

    # 用于存储每个特征与所有目标的相关性系数值
    feature_map_visual = {}
    # 用于存储每个相关性计算的p值是否显著
    sig_map_visual = {}

    # 遍历特征，计算每个特征与所有目标的相关性
    for feature in final_features:
        # 存储当前特征与所有目标的相关性系数
        corr_row = []
        # 存储当前特征与所有目标的显著性检验结果
        sig_row = []
        # 相关性分析，根据选则的方法来分析
        for target in targets:
            if correlation_method == "pearson":
                corr, p_value = stats.pearsonr(
                    all_data[feature].dropna(), all_data[target].dropna()
                )
            elif correlation_method == "spearman":
                corr, p_value = stats.spearmanr(
                    all_data[feature].dropna(), all_data[target].dropna()
                )
            elif correlation_method == "kendall":
                corr, p_value = stats.kendalltau(
                    all_data[feature].dropna(), all_data[target].dropna()
                )
            else:
                raise ValueError("请选择 'pearson', 'spearman', 或 'kendall'")

            # 添加相关系数的结果
            corr_row.append(corr)
            # 看看显著性是否合格，然后添加
            sig_row.append(p_value < 0.05)
        # 将当前特征与所有目标的相关性系数列表存入字典
        feature_map_visual[feature] = corr_row
        # 将当前特征与所有目标的显著性结果列表存入字典
        sig_map_visual[feature] = sig_row

    # 行为特征，列为目标
    df = pd.DataFrame(feature_map_visual, index=targets).T.reindex(final_features)
    print("相关性分析结果:")
    print(df.to_string())
    # 对显著性结果字典执行相同的操作，创建对应的DataFrame
    df_sig = pd.DataFrame(sig_map_visual, index=targets).T.reindex(final_features)

    # 调用之前定义的绘图函数进行绘图，并传入所有必要的参数
    create_correlation_circos(
        df=df,  # 相关性系数值
        df_sig=df_sig,  # 显著性
        stages=targets,  # 目标的数量决定了层数
        groups=groups,  # 特征分组信息
        color_palette=select_color,  # 颜色方案
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

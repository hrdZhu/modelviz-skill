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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pycirclize import Circos

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# =========================================================================================
# ======================================2.颜色库=======================================
# =========================================================================================
COLOR_SCHEMES = {
    1: [
        "#4DBBD5",
        "#E64B35",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B0",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
        "#B09C85",
        "#374E55",
        "#DF8F44",
        "#00A1D5",
        "#B98C46",
        "#6D9731",
        "#2878B5",
        "#9AC9DB",
        "#C82423",
        "#FF8884",
        "#54B345",
        "#05B9E2",
        "#8983BF",
        "#C76DA2",
        "#587848",
        "#E89132",
        "#08519C",
        "#41AB5D",
        "#A50F15",
        "#6A51A3",
        "#238B45",
        "#2171B5",
        "#6BAED6",
        "#4292C6",
        "#238B45",
        "#74C476",
    ],
    2: [
        "#E6194B",
        "#3CB44B",
        "#FFE119",
        "#4363D8",
        "#F58231",
        "#911EB4",
        "#46F0F0",
        "#F032E6",
        "#BCF60C",
        "#FABEBE",
        "#008080",
        "#E6BEFF",
        "#9A6324",
        "#FFFAC8",
        "#800000",
        "#AAFFC3",
        "#808000",
        "#FFD8B1",
        "#000075",
        "#808080",
        "#FF1493",
        "#1E90FF",
        "#00FF00",
        "#FF4500",
        "#8A2BE2",
        "#DAA520",
        "#5F9EA0",
        "#D2691E",
        "#FF7F50",
        "#6495ED",
        "#DC143C",
        "#00FFFF",
        "#00008B",
    ],
    3: [
        "#2F4F4F",
        "#8B4513",
        "#228B22",
        "#000080",
        "#B22222",
        "#DAA520",
        "#483D8B",
        "#008080",
        "#556B2F",
        "#8B008B",
        "#A0522D",
        "#4682B4",
        "#D2691E",
        "#9ACD32",
        "#BC8F8F",
        "#00CED1",
        "#663399",
        "#7F7F7F",
        "#BDB76B",
        "#CD5C5C",
        "#4B0082",
        "#ADFF2F",
        "#FF6347",
        "#00BFFF",
        "#32CD32",
        "#FFD700",
        "#6A5ACD",
        "#2E8B57",
        "#800000",
        "#20B2AA",
        "#7B68EE",
        "#FF8C00",
        "#191970",
    ],
    4: [
        "#FF00FF",
        "#00FF00",
        "#FF4500",
        "#00FFFF",
        "#FFFF00",
        "#1E90FF",
        "#ADFF2F",
        "#FF1493",
        "#00FF7F",
        "#FFD700",
        "#8A2BE2",
        "#FF6347",
        "#00FA9A",
        "#00BFFF",
        "#DA70D6",
        "#7FFF00",
        "#FF8C00",
        "#40E0D0",
        "#BA55D3",
        "#32CD32",
        "#F08080",
        "#20B2AA",
        "#9370DB",
        "#FF7F50",
        "#00CED1",
        "#DDA0DD",
        "#7CFC00",
        "#FA8072",
        "#48D1CC",
        "#9932CC",
        "#66CDAA",
        "#E9967A",
        "#AFEEEE",
        "#8B008B",
        "#BDB76B",
    ],
    5: [
        "#FFB3BA",
        "#BAFFC9",
        "#BAE1FF",
        "#FFFFBA",
        "#FFDFBA",
        "#E0BBE4",
        "#957DAD",
        "#D291BC",
        "#FEC8D8",
        "#FFDFD3",
        "#B5EAD7",
        "#C7CEEA",
        "#FF9AA2",
        "#FFDAC1",
        "#E2F0CB",
        "#B5EAD7",
        "#C7CEEA",
        "#F3FFE3",
        "#E1F7D5",
        "#FFB7B2",
        "#FFDAC1",
        "#E2F0CB",
        "#B5EAD7",
        "#C7CEEA",
        "#FF9AA2",
        "#FFDAC1",
        "#E2F0CB",
        "#B5EAD7",
        "#C7CEEA",
        "#FFB3BA",
        "#BAFFC9",
        "#BAE1FF",
        "#FFFFBA",
        "#FFDFBA",
    ],
    6: [
        "#0B3D91",
        "#1E5631",
        "#A8DADC",
        "#457B9D",
        "#1D3557",
        "#2E7D32",
        "#66BB6A",
        "#D4E157",
        "#FFEE58",
        "#FFCA28",
        "#FFA726",
        "#FF7043",
        "#8D6E63",
        "#BDBDBD",
        "#78909C",
        "#5C6BC0",
        "#26A69A",
        "#EC407A",
        "#AB47BC",
        "#7E57C2",
        "#EF5350",
        "#26C6DA",
        "#D4E157",
        "#DCE775",
        "#FFF176",
        "#FFD54F",
        "#FFB74D",
        "#FF8A65",
        "#A1887F",
        "#E0E0E0",
        "#90A4AE",
        "#9575CD",
        "#7986CB",
        "#4DB6AC",
        "#81C784",
    ],
    7: [
        "#333333",
        "#D32F2F",
        "#1976D2",
        "#FBC02D",
        "#388E3C",
        "#7B1FA2",
        "#F57C00",
        "#00796B",
        "#C2185B",
        "#5D4037",
        "#616161",
        "#E57373",
        "#64B5F6",
        "#FFF176",
        "#81C784",
        "#BA68C8",
        "#FFB74D",
        "#4DB6AC",
        "#F06292",
        "#A1887F",
        "#9E9E9E",
        "#FF8A65",
        "#4DD0E1",
        "#AED581",
        "#DCE775",
        "#FFD54F",
        "#D1C4E9",
        "#B3E5FC",
        "#B2DFDB",
        "#DCEDC8",
        "#F8BBD0",
        "#D7CCC8",
        "#F5F5F5",
        "#CFD8DC",
        "#FFCCBC",
    ],
    8: [
        "#002147",
        "#003366",
        "#005073",
        "#007399",
        "#0099CC",
        "#00BFFF",
        "#33CCFF",
        "#66E0FF",
        "#99F0FF",
        "#CCF9FF",
        "#FF6F61",
        "#FF8E71",
        "#FFAE81",
        "#FFCE91",
        "#FFEEA1",
        "#F0E68C",
        "#E6E6FA",
        "#D8BFD8",
        "#DDA0DD",
        "#EE82EE",
        "#FF00FF",
        "#8A2BE2",
        "#4B0082",
        "#483D8B",
        "#6A5ACD",
        "#7B68EE",
        "#9370DB",
        "#AB82FF",
        "#C1A0FF",
        "#D7BFFF",
        "#EBDFFF",
        "#F5F0FF",
        "#FFFFFF",
        "#F0F8FF",
        "#E0FFFF",
    ],
    9: [
        "#800020",
        "#A01030",
        "#C02040",
        "#E03050",
        "#FF4060",
        "#FF6080",
        "#FF80A0",
        "#FFA0C0",
        "#FFC0E0",
        "#FFE0F0",
        "#000080",
        "#1010A0",
        "#2020C0",
        "#3030E0",
        "#4040FF",
        "#6060FF",
        "#8080FF",
        "#A0A0FF",
        "#C0C0FF",
        "#E0E0FF",
        "#4B0082",
        "#601090",
        "#7520A0",
        "#8A30B0",
        "#A040C0",
        "#B550D0",
        "#CA60E0",
        "#DF70F0",
        "#F480FF",
        "#3CB371",
        "#4CD381",
        "#5CE391",
        "#6CF3A1",
        "#7CFFB1",
        "#8CFFC1",
    ],
    10: [
        "#EDC9AF",
        "#E2725B",
        "#C04000",
        "#964B00",
        "#6F4E37",
        "#228B22",
        "#32CD32",
        "#008000",
        "#006400",
        "#556B2F",
        "#FF8C00",
        "#FFA500",
        "#FFD700",
        "#F0E68C",
        "#FFFFE0",
        "#483D8B",
        "#6A5ACD",
        "#8A2BE2",
        "#9400D3",
        "#9932CC",
        "#BA55D3",
        "#D8BFD8",
        "#E6E6FA",
        "#FFF0F5",
        "#FFDEAD",
        "#F5DEB3",
        "#DEB887",
        "#D2B48C",
        "#BC8F8F",
        "#A0522D",
        "#8B4513",
        "#D2691E",
        "#B8860B",
        "#DAA520",
        "#EEE8AA",
    ],
    11: [
        "#F44336",
        "#E91E63",
        "#9C27B0",
        "#673AB7",
        "#3F51B5",
        "#2196F3",
        "#03A9F4",
        "#00BCD4",
        "#009688",
        "#4CAF50",
        "#8BC34A",
        "#CDDC39",
        "#FFEB3B",
        "#FFC107",
        "#FF9800",
        "#FF5722",
        "#795548",
        "#9E9E9E",
        "#607D8B",
        "#B71C1C",
        "#880E4F",
        "#4A148C",
        "#311B92",
        "#1A237E",
        "#0D47A1",
        "#01579B",
        "#006064",
        "#004D40",
        "#1B5E20",
        "#33691E",
        "#827717",
        "#F57F17",
        "#E65100",
        "#BF360C",
        "#3E2723",
    ],
    12: [
        "#FFADAD",
        "#FFD6A5",
        "#FDFFB6",
        "#CAFFBF",
        "#9BF6FF",
        "#A0C4FF",
        "#BDB2FF",
        "#FFC6FF",
        "#FFFFFC",
        "#E63946",
        "#F1FAEE",
        "#A8DADC",
        "#457B9D",
        "#1D3557",
        "#2B2D42",
        "#8D99AE",
        "#EDF2F4",
        "#EF233C",
        "#D90429",
        "#001219",
        "#005F73",
        "#0A9396",
        "#94D2BD",
        "#E9D8A6",
        "#EE9B00",
        "#CA6702",
        "#BB3E03",
        "#AE2012",
        "#9B2226",
        "#F94144",
        "#F3722C",
        "#F8961E",
        "#F9844A",
        "#F9C74F",
        "#90BE6D",
    ],
    13: [
        "#000000",
        "#111111",
        "#222222",
        "#333333",
        "#444444",
        "#555555",
        "#666666",
        "#777777",
        "#888888",
        "#999999",
        "#AAAAAA",
        "#BBBBBB",
        "#CCCCCC",
        "#DDDDDD",
        "#EEEEEE",
        "#FFFFFF",
        "#FF0000",
        "#00FF00",
        "#0000FF",
        "#FFFF00",
        "#00FFFF",
        "#FF00FF",
        "#C0C0C0",
        "#808080",
        "#800000",
        "#808000",
        "#008000",
        "#800080",
        "#008080",
        "#000080",
        "#FFA500",
        "#A52A2A",
        "#800000",
        "#00008B",
        "#4B0082",
    ],
    14: [
        "#543005",
        "#8c510a",
        "#bf812d",
        "#dfc27d",
        "#f6e8c3",
        "#f5f5f5",
        "#c7eae5",
        "#80cdc1",
        "#35978f",
        "#01665e",
        "#003c30",
        "#40004b",
        "#762a83",
        "#9970ab",
        "#c2a5cf",
        "#e7d4e8",
        "#f7f7f7",
        "#d9f0d3",
        "#a6dba0",
        "#5aae61",
        "#1b7837",
        "#00441b",
        "#7f3b08",
        "#b35806",
        "#e08214",
        "#fdb863",
        "#fee0b6",
        "#f7f7f7",
        "#d8daeb",
        "#b2abd2",
        "#8073ac",
        "#542788",
        "#2d004b",
        "#a50026",
        "#d73027",
    ],
    15: [
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
        "#8e0152",
        "#c51b7d",
        "#de77ae",
        "#f1b6da",
        "#fde0ef",
        "#f7f7f7",
        "#e6f5d0",
        "#b8e186",
        "#7fbc41",
        "#4d9221",
        "#276419",
        "#3f007d",
        "#54278f",
        "#6a51a3",
        "#807dba",
        "#9e9ac8",
        "#bcbddc",
        "#dadaeb",
        "#f2f0f7",
        "#00441b",
        "#006d2c",
        "#238b45",
        "#41ae76",
        "#66c2a4",
    ],
    16: [
        "#00429d",
        "#2e59a8",
        "#4771b2",
        "#5d8abd",
        "#73a2c6",
        "#8abccf",
        "#a5d5d8",
        "#c5eddf",
        "#ffffe0",
        "#ffdbc5",
        "#ffb8ab",
        "#ff9493",
        "#ff717c",
        "#ff4d64",
        "#ff234d",
        "#ff0035",
        "#7c1d6f",
        "#9c3187",
        "#bd46a0",
        "#de5bb9",
        "#ff70d3",
        "#ff8de0",
        "#ffaaed",
        "#ffc6fa",
        "#ffd6ff",
        "#003f5c",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
        "#488f31",
        "#de425b",
    ],
    17: [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
        "#a6cee3",
        "#1f78b4",
        "#b2df8a",
        "#33a02c",
        "#fb9a99",
        "#e31a1c",
        "#fdbf6f",
        "#ff7f00",
        "#cab2d6",
        "#6a3d9a",
        "#ffff99",
        "#b15928",
        "#fbb4ae",
        "#b3cde3",
        "#ccebc5",
        "#decbe4",
        "#fed9a6",
        "#ffffcc",
        "#e5d8bd",
        "#fddaec",
        "#f2f2f2",
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
    ],
    18: [
        "#b3e2cd",
        "#fdcdac",
        "#cbd5e8",
        "#f4cae4",
        "#e6f5c9",
        "#fff2ae",
        "#f1e2cc",
        "#cccccc",
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#ffff33",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
        "#8da0cb",
        "#e78ac3",
        "#a6d854",
        "#ffd92f",
        "#e5c494",
        "#b3b3b3",
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
    ],
    19: [
        "#ccebc5",
        "#ffed6f",
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
        "#393b79",
        "#5254a3",
        "#6b6ecf",
        "#9c9ede",
        "#637939",
        "#8ca252",
        "#b5cf6b",
        "#cedb9c",
        "#8c6d31",
        "#bd9e39",
        "#e7ba52",
        "#e7cb94",
        "#843c39",
    ],
    20: [
        "#ad494a",
        "#d6616b",
        "#e7969c",
        "#7b4173",
        "#a55194",
        "#ce6dbd",
        "#de9ed6",
        "#3182bd",
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#e6550d",
        "#fd8d3c",
        "#fdae6b",
        "#fdd0a2",
        "#31a354",
        "#74c476",
        "#a1d99b",
        "#c7e9c0",
        "#756bb1",
        "#9e9ac8",
        "#bcbddc",
        "#dadaeb",
        "#636363",
        "#969696",
        "#bdbdbd",
        "#d9d9d9",
        "#1a9850",
        "#91cf60",
        "#d9ef8b",
        "#ffffbf",
        "#fee08b",
        "#fc8d59",
        "#d73027",
        "#762a83",
    ],
}
scheme_index = 20  # 选择配色方案


# =========================================================================================
# ======================================3.绘图函数=======================================
# =========================================================================================
def draw_chord_diagram(df, labels, scheme_id=1):
    N = len(labels)  # 特征总数量
    # 每个扇区的大小
    sectors_dict = dict(zip(labels, df.sum(axis=1).values))
    circos = Circos(sectors=sectors_dict, space=0)  # 初始化Circos对象，传入扇区数据
    # 获取颜色方案
    full_color_list = COLOR_SCHEMES.get(scheme_id, COLOR_SCHEMES[1])
    # 如果特征数量超过了定义的35种，则循环使用颜色
    colors = [full_color_list[i % len(full_color_list)] for i in range(N)]
    # 创建颜色字典
    color_dict = dict(zip(labels, colors))
    # 绘制扇区和特征标签
    for sector in circos.sectors:  # 遍历每一个扇区
        # 获取颜色
        fill_color = color_dict.get(sector.name, "#CCCCCC")
        # 绘制半径93到100的主颜色环，带黑色细边框
        sector.rect(r_lim=(93, 100), facecolor=fill_color, edgecolor="black", linewidth=0.2)
        # 绘制外圈阴影圆环
        sector.rect(r_lim=(101, 108), facecolor="#F0F0F0", edgecolor="#D0D0D0", linewidth=0)
        label_text = sector.name  # 获取当前扇区的名称作为特征标签文本
        # 绘制特征标签文本
        sector.text(label_text, r=104, orientation="vertical", size=14, color="black")
    # 绘制连线
    for i in range(N):  # 遍历行
        for j in range(i, N):  # 遍历列
            value = df.iloc[i, j]  # 获取当前行列交叉处的数值
            # 只有当数值大于0时才绘制连线
            if value > 0:
                source_label = labels[i]  # 获取源节点标签
                target_label = labels[j]  # 获取目标节点标签
                src_start = df.iloc[i, :j].sum()  # 源节点在扇区内的起始弧度位置
                src_end = src_start + value  # 源节点在扇区内的结束弧度位置
                tgt_start = df.iloc[:i, j].sum()  # 目标节点在扇区内的起始弧度位置
                tgt_end = tgt_start + value  # 目标节点在扇区内的结束弧度位置
                # 获取颜色
                color = color_dict.get(source_label, "#888888")
                # 绘制连线
                circos.link(
                    (source_label, src_start, src_end),  # 源点坐标元组（标签，起点，终点）
                    (target_label, tgt_start, tgt_end),  # 终点坐标元组
                    color=color,  # 连线颜色
                    alpha=0.6,  # 连线透明度
                    direction=0,  # 连线方向参数
                )
    circos.plotfig()  # 生成并绘制最终的图形对象
    # 设置标题
    plt.title("Circos Graph", fontsize=18, pad=80)
    # 保存
    plt.savefig(str(OUTPUT_DIR / f"chord_diagram_{scheme_index}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(str(OUTPUT_DIR / f"chord_diagram_{scheme_index}.pdf"), bbox_inches="tight")


# =========================================================================================
# ======================================4.执行部分=======================================
# =========================================================================================
if __name__ == "__main__":
    data = str(DATA_DIR / "data.xlsx")  # 输入数据
    # 读取数据
    df = pd.read_excel(data, index_col=0)
    labels = df.index.tolist()  # 获取数据框的索引列表，作为节点的标签
    draw_chord_diagram(df, labels, scheme_id=scheme_index)  # 调用绘图函数

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

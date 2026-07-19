<div align="center">

# ModelViz Skill

### AI 数学建模 × 科研可视化

**从自然语言需求和真实数据出发，自动完成选图、模板适配、绘图生成、质量检查与有限自动修复。**

<p>
  <img src="https://img.shields.io/badge/Agent%20Skill-Codex%20%7C%20Claude%20Code-4F46E5?style=flat-square" alt="Agent Skill">
  <img src="https://img.shields.io/badge/Templates-Nearly%20100-2EA44F?style=flat-square" alt="Nearly 100 templates">
  <img src="https://img.shields.io/badge/Input-CSV%20%7C%20XLSX%20%7C%20XLS-0A66C2?style=flat-square" alt="Supported data files">
  <img src="https://img.shields.io/badge/Output-PNG%20%7C%20SVG%20%7C%20PDF%20%7C%20Python-8A2BE2?style=flat-square" alt="Outputs">
</p>

<table width="100%">
  <tr>
    <td width="25%" align="center" nowrap><strong>🧠 智能选图</strong><br><sub><nobr>理解表达目标与数据关系</nobr></sub></td>
    <td width="25%" align="center" nowrap><strong>🧩 模板适配</strong><br><sub><nobr>保留科研布局与视觉风格</nobr></sub></td>
    <td width="25%" align="center" nowrap><strong>📦 按需依赖</strong><br><sub><nobr>仅补全当前模板所需库</nobr></sub></td>
    <td width="25%" align="center" nowrap><strong>🔍 双重质检</strong><br><sub><nobr>技术检查与视觉检查</nobr></sub></td>
  </tr>
</table>

</div>

---

## 📚 目录

- [效果预览](#-效果预览)
- [核心能力](#-核心能力)
- [适用场景](#-适用场景)
- [使用示例](#-使用示例)
- [工作流程](#-工作流程)
- [项目结构](#-项目结构)
- [主要阶段入口](#-主要阶段入口)
- [按需依赖机制](#-按需依赖机制)
- [输出内容](#-输出内容)
- [质量评估](#-质量评估)
- [安全说明](#-安全说明)
- [当前限制](#-当前限制)
- [开发与测试](#-开发与测试)

> [!NOTE]
> ModelViz 不是简单地“按指令画一张图”。它会先理解用户想表达的趋势、对比、分布或相关关系，再结合真实数据，从模板库中筛选合适方案，保留科研版式与配色，最终输出可复现代码、图表和质量报告。

做数学建模图表时，真正耗时间的往往不是调用一次绘图库，而是找到合适的科研版式、完成数据列映射，并把标题、坐标轴、图例和配色调整到可以直接放进论文或答辩中。

ModelViz 将这套流程整理成一个可复用 Skill：内置近百个正式科学可视化模板，配套上百个预览与模板资产，覆盖聚类降维、相关性、分布不确定性、预测评估、敏感性分析、网络关系、空间分析和时序趋势等方向。

## ✨ 为什么使用 ModelViz

- **从表达目标出发**：先判断趋势、对比、分布、相关性等核心任务，再决定图表形式。
- **从真实数据出发**：基于实际字段、类型和关系完成模板筛选与数据映射。
- **从论文阅读习惯出发**：优先选择清晰、规范、低饱和、重点突出的科研版式。
- **从可复现流程出发**：同时输出图表、Python 代码和质量报告。



---

## 🖼️ 效果预览

<p align="center">
  <img src="docs/assets/readme-preview-3.jpg" alt="ModelViz 效果预览合集 3" width="100%">
</p>

<p align="center">
  <img src="docs/assets/readme-preview-2.jpg" alt="ModelViz 效果预览合集 2" width="100%">
</p>

<p align="center">
  <img src="docs/assets/readme-preview-1.jpg" alt="ModelViz 效果预览合集 1" width="100%">
</p>

<div align="center">
  <table width="100%">
    <tr>
      <td width="50%" align="center">
        <img src="templates/01_CLU_clustering_reduction/01_CLU_001/output/preview.png" alt="聚类关联矩阵预览" width="100%">
        <br>
        <sub>聚类关联矩阵</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/01_CLU_clustering_reduction/01_CLU_002/assets/preview.png" alt="聚类热力图预览" width="100%">
        <br>
        <sub>聚类热力图</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/10_SEN_sensitivity_robustness/10_SEN_005/output/preview.png" alt="GAM 等高线面板预览" width="100%">
        <br>
        <sub>GAM 等高线面板</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/10_SEN_sensitivity_robustness/10_SEN_011/output/preview.png" alt="SHAP 交互网络预览" width="100%">
        <br>
        <sub>SHAP 交互网络</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/09_REL_relationship_correlation/09_REL_005/output/preview.png" alt="三角相关热力图预览" width="100%">
        <br>
        <sub>三角相关热力图</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/09_REL_relationship_correlation/09_REL_013/output/preview.png" alt="三角网络相关图预览" width="100%">
        <br>
        <sub>三角网络相关图</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/06_NET_network_flow/06_NET_001/assets/preview.png" alt="弦图网络关系预览" width="100%">
        <br>
        <sub>弦图网络关系</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/03_EVL_composite_evaluation/03_EVL_002/output/preview.png" alt="径向综合评价预览" width="100%">
        <br>
        <sub>径向综合评价</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/01_CLU_clustering_reduction/01_CLU_004/assets/preview.png" alt="RDA 排序分析预览" width="100%">
        <br>
        <sub>RDA / 排序分析</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/02_CMP_comparison_ranking/02_CMP_001/output/preview.png" alt="环形柱状对比预览" width="100%">
        <br>
        <sub>环形柱状对比</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/01_CLU_clustering_reduction/01_CLU_003/assets/preview.png" alt="聚类投影密度预览" width="100%">
        <br>
        <sub>聚类投影密度</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/02_CMP_comparison_ranking/02_CMP_012/output/preview.png" alt="热力图排名矩阵预览" width="100%">
        <br>
        <sub>热力图排名矩阵</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/04_DIS_distribution_uncertainty/04_DIS_006/output/preview.png" alt="小提琴分布矩阵预览" width="100%">
        <br>
        <sub>小提琴分布矩阵</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/05_MPN_multi_panel_report/05_MPN_004/output/preview.png" alt="三维响应面组合预览" width="100%">
        <br>
        <sub>三维响应面组合</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/09_REL_relationship_correlation/09_REL_016/output/preview.png" alt="扇形相关热力图预览" width="100%">
        <br>
        <sub>扇形相关热力图</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/09_REL_relationship_correlation/09_REL_017/output/preview.png" alt="边际回归散点图预览" width="100%">
        <br>
        <sub>边际回归散点图</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/09_REL_relationship_correlation/09_REL_019/output/preview.png" alt="散点矩阵热力图预览" width="100%">
        <br>
        <sub>散点矩阵热力图</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/10_SEN_sensitivity_robustness/10_SEN_006/output/preview.png" alt="多关系散点面板预览" width="100%">
        <br>
        <sub>多关系散点面板</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center">
        <img src="templates/10_SEN_sensitivity_robustness/10_SEN_015/output/preview.png" alt="敏感性等高线矩阵预览" width="100%">
        <br>
        <sub>敏感性等高线矩阵</sub>
      </td>
      <td width="50%" align="center">
        <img src="templates/11_SPA_spatial_geographic/11_SPA_001/output/preview.png" alt="空间主导因子图预览" width="100%">
        <br>
        <sub>空间主导因子图</sub>
      </td>
    </tr>
  </table>
</div>

---

## 🧠 核心能力

<table width="100%">
  <thead>
    <tr>
      <th width="28%" align="left">能力</th>
      <th width="72%" align="left">说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="28%" align="left" valign="top">科学可视化模板库</td>
      <td width="72%" align="left" valign="top"><code>templates/</code> 保存数学建模和科研报告常见图表模板，<code>docs/template_catalog.yaml</code> 记录统一元数据。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">自然语言需求结构化</td>
      <td width="72%" align="left" valign="top">将用户描述解析为固定 Pydantic 结构，保留图表类型、功能、风格、使用场景和负面要求。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">候选模板召回</td>
      <td width="72%" align="left" valign="top">使用确定性匹配工具召回 5–10 个候选模板，优先考虑功能和图表类型。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">结合数据最终选择</td>
      <td width="72%" align="left" valign="top">由大语言模型读取数据上下文，并且只能从候选模板中选择最终模板。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">模板风格保留</td>
      <td width="72%" align="left" valign="top">学习选定模板的布局、配色、字体、图例和坐标轴风格，再适配用户数据。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">按需依赖补全</td>
      <td width="72%" align="left" valign="top">选定模板后才识别、检查和安装实际需要的第三方 Python 包。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">可复现绘图代码</td>
      <td width="72%" align="left" valign="top">适配后的代码保存为工作区脚本，不修改原始模板和用户数据。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">技术与视觉质量检查</td>
      <td width="72%" align="left" valign="top">检查执行结果、图片可读性、空白风险、警告以及图表是否满足用户目标。</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top">有限自动修复</td>
      <td width="72%" align="left" valign="top">只修复报告明确指出的问题，避免无限循环和过度重写。</td>
    </tr>
  </tbody>
</table>

## 🎯 适用场景

<table width="100%">
  <tr>
    <th width="50%">✅ 适合</th>
    <th width="50%">❌ 不适合</th>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <ul>
        <li>CSV、XLSX 或 XLS 数据的论文与答辩绘图</li>
        <li>只描述分析目标、但不确定图表类型</li>
        <li>希望复用现有模板的科研风、布局和配色</li>
        <li>需要图片、可复现代码和质量报告</li>
      </ul>
    </td>
    <td width="50%" valign="top">
      <ul>
        <li>只解释数学概念而不需要绘图</li>
        <li>任务依赖真实数据，但用户没有提供数据</li>
        <li>模板库完全不支持目标图表类型</li>
        <li>要求修改原始数据或执行与绘图无关的任意代码</li>
      </ul>
    </td>
  </tr>
</table>

## 💬 使用示例

在支持 Skill 的宿主软件中，可以直接用自然语言描述任务：

```text
请使用这个科学可视化 Skill。

数据文件是 data.xlsx。
我想比较不同地区各项指标的差异，要求适合数学建模论文，
突出整体差异和异常地区，不要使用三维图。
```

<details>
<summary><strong>查看更多示例</strong></summary>

```text
请用 data.csv 画不同年份产量和增长率的趋势图，适合论文正文，风格简洁。
```

```text
请分析这些变量之间的相关性，生成科研风热力图，要求标签清晰。
```

```text
我想展示不同算法的预测误差对比，不要饼图，尽量突出误差和排名。
```

```text
请展示多指标综合评价结果，适合数学建模答辩展示，低饱和配色。
```

</details>

## 🔄 工作流程

```mermaid
flowchart TD
    A["用户需求与数据"] --> B["需求结构化"]
    B --> C["候选模板召回"]
    C --> D["结合数据选择模板"]
    D --> E["按需检查依赖"]
    E --> F["模板适配"]
    F --> G["绘图执行"]
    G --> H["技术与视觉检查"]
    H --> I{"需要修复？"}
    I -- "是" --> J["局部自动修复"]
    J -. 重新执行 .-> G
    I -- "否" --> K["返回图片、代码和报告"]
```

> [!IMPORTANT]
> 宿主模型应先读取根目录 `SKILL.md`，再按其中的阶段流程调用 Prompt、Tools 和 Services。不要绕过工具直接编造候选模板、依赖状态、生成图片或质量结果。

## 🗂️ 项目结构

<details open>
<summary><strong>查看关键目录</strong></summary>

```text
modelviz-skill/
├── SKILL.md
├── README.md
├── prompts/
├── templates/
├── docs/
├── src/
│   ├── prompts/
│   ├── schemas/
│   ├── tools/
│   └── services/
├── tests/
├── evals/
├── examples/
├── gallery/
├── workspace/
├── outputs/
└── requirements.txt
```

<table width="100%">
  <thead>
    <tr>
      <th width="28%" align="left">目录</th>
      <th width="72%" align="left">作用</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="28%" align="left" valign="top"><code>SKILL.md</code></td>
      <td width="72%" align="left" valign="top">Skill 总入口和实际执行流程</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>templates/</code></td>
      <td width="72%" align="left" valign="top">原始绘图模板库，运行时应保持只读</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>prompts/</code></td>
      <td width="72%" align="left" valign="top">模板适配、视觉检查和代码修复规则</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>src/prompts/</code></td>
      <td width="72%" align="left" valign="top">可执行的 <code>ChatPromptTemplate</code> 定义</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>src/tools/</code></td>
      <td width="72%" align="left" valign="top">文件读取、匹配、依赖检查、执行和质量检查工具</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>src/services/</code></td>
      <td width="72%" align="left" valign="top">阶段 2–6 的流程入口</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>docs/</code></td>
      <td width="72%" align="left" valign="top">模板目录、需求词表、依赖映射和阶段说明</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>tests/</code></td>
      <td width="72%" align="left" valign="top">单元测试和集成测试</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>evals/</code></td>
      <td width="72%" align="left" valign="top">离线评估案例、脚本和指标报告</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>gallery/</code></td>
      <td width="72%" align="left" valign="top">汇总后的预览图库</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>workspace/</code></td>
      <td width="72%" align="left" valign="top">运行时中间文件</td>
    </tr>
    <tr>
      <td width="28%" align="left" valign="top"><code>outputs/</code></td>
      <td width="72%" align="left" valign="top">最终图表输出目录</td>
    </tr>
  </tbody>
</table>

</details>

## 🧭 主要阶段入口

<table width="100%">
  <thead>
    <tr>
      <th width="30%" align="left">阶段</th>
      <th width="70%" align="left">入口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="30%" align="left" valign="top">需求解析</td>
      <td width="70%" align="left" valign="top"><code>src.services.requirement_parser.parse_and_save_requirement</code></td>
    </tr>
    <tr>
      <td width="30%" align="left" valign="top">候选召回</td>
      <td width="70%" align="left" valign="top"><code>src.services.candidate_matching_pipeline.run_candidate_matching_pipeline</code></td>
    </tr>
    <tr>
      <td width="30%" align="left" valign="top">最终模板选择</td>
      <td width="70%" align="left" valign="top"><code>src.services.final_template_selection_pipeline.run_final_template_selection_pipeline</code></td>
    </tr>
    <tr>
      <td width="30%" align="left" valign="top">模板依赖与代码适配</td>
      <td width="70%" align="left" valign="top"><code>src.services.template_adaptation_pipeline.run_template_adaptation_pipeline</code></td>
    </tr>
    <tr>
      <td width="30%" align="left" valign="top">技术检查、视觉检查和修复</td>
      <td width="70%" align="left" valign="top"><code>src.services.plot_quality_pipeline.run_plot_quality_pipeline</code></td>
    </tr>
  </tbody>
</table>

这些入口由 `SKILL.md` 编排。模型语义判断步骤需要宿主提供支持结构化输出的模型对象；测试和离线评估使用 Mock 结构化输出验证流程衔接。

## 📦 按需依赖机制

模板库涉及多种绘图库和科学计算库，因此不建议预先安装所有模板可能用到的依赖。当前 `requirements.txt` 只保留 Skill 编排和测试所需的核心依赖。

```text
读取模板 import
→ 合并 catalog 依赖
→ 排除标准库和项目模块
→ 映射 pip 包名
→ 检查当前环境
→ 受控安装缺失依赖
→ 再次验证
```

<details>
<summary><strong>查看依赖安全规则</strong></summary>

- 不执行模板中携带的安装命令。
- 不使用 `sudo`。
- 不使用 `shell=True`。
- 不安装 URL、Git 地址、本地路径或额外 pip 参数。
- 不直接信任模型生成的包名。
- 不覆盖项目原始 `requirements.txt`。
- 当前任务依赖写入 `workspace/requirements.generated.txt`。
- 个别依赖系统级组件的模板可能需要人工处理。

</details>

## 📤 输出内容

一次完整任务通常会产生：

<table width="100%">
  <thead>
    <tr>
      <th width="45%" align="left">输出</th>
      <th width="55%" align="left">路径</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="45%" align="left" valign="top">最终 PNG、SVG 或 PDF 图表</td>
      <td width="55%" align="left" valign="top"><code>outputs/</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">适配后的 Python 绘图代码</td>
      <td width="55%" align="left" valign="top"><code>workspace/adapted_plot.py</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">最终模板选择结果</td>
      <td width="55%" align="left" valign="top"><code>workspace/final_template_selection.json</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">数据列映射与适配计划</td>
      <td width="55%" align="left" valign="top"><code>workspace/adaptation_plan.json</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">当前任务依赖报告</td>
      <td width="55%" align="left" valign="top"><code>workspace/dependency_report.json</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">技术质量报告</td>
      <td width="55%" align="left" valign="top"><code>workspace/technical_quality_report.json</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">视觉质量报告</td>
      <td width="55%" align="left" valign="top"><code>workspace/visual_quality_report.json</code></td>
    </tr>
    <tr>
      <td width="45%" align="left" valign="top">最终质量报告</td>
      <td width="55%" align="left" valign="top"><code>workspace/final_quality_report.json</code></td>
    </tr>
  </tbody>
</table>

普通用户不需要手动编辑这些中间 JSON。流程失败时，宿主模型应准确说明失败阶段、原因和建议，不应伪造成功结果。

## 📊 质量评估

> [!WARNING]
> 以下数据来自当前精简离线评估集。模型相关步骤使用 Mock 结构化输出，程序化工具调用真实实现；这些结果不代表所有真实数据和所有宿主模型的表现。

<table width="100%">
  <thead>
    <tr>
      <th width="65%" align="left">指标</th>
      <th width="35%" align="center">当前结果</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="65%" align="left" valign="top">端到端任务成功率</td>
      <td width="35%" align="center" valign="top"><strong>87.50%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">候选模板 Recall@5</td>
      <td width="35%" align="center" valign="top"><strong>100.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">候选模板 Recall@8</td>
      <td width="35%" align="center" valign="top"><strong>100.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">最终模板选择准确率</td>
      <td width="35%" align="center" valign="top"><strong>100.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">首次代码执行成功率</td>
      <td width="35%" align="center" valign="top"><strong>62.50%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">最终代码执行成功率</td>
      <td width="35%" align="center" valign="top"><strong>87.50%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">图片质量平均分</td>
      <td width="35%" align="center" valign="top"><strong>4.50 / 5</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">自动修复成功率</td>
      <td width="35%" align="center" valign="top"><strong>100.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">数据列幻觉率</td>
      <td width="35%" align="center" valign="top"><strong>0.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">候选范围违规率</td>
      <td width="35%" align="center" valign="top"><strong>0.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">原始模板修改率</td>
      <td width="35%" align="center" valign="top"><strong>0.00%</strong></td>
    </tr>
    <tr>
      <td width="65%" align="left" valign="top">用户数据修改率</td>
      <td width="35%" align="center" valign="top"><strong>0.00%</strong></td>
    </tr>
  </tbody>
</table>

完整报告见 `evals/evaluation_report.md` 和 `evals/evaluation_summary.json`。

## 🛡️ 安全说明

- 原始模板目录 `templates/` 应保持只读。
- 用户原始数据不会被覆盖。
- 适配代码、依赖报告和修复历史写入 `workspace/`。
- 最终图片写入 `outputs/`。
- 依赖安装受到包名格式、数量和轮数限制。
- 自动修复最多执行有限次数，不能无限循环。
- 使用第三方宿主或外部 Skill 前，应审查其 `SKILL.md` 和可执行脚本。

## ⚠️ 当前限制

- 模板库覆盖范围有限，候选模板都不合适时应停止并追问或说明原因。
- 大型数据会抽样进入模型上下文，不能声称模型看过全部原始数据。
- 复杂 Excel 工作簿可能需要用户指定 Sheet。
- 某些模板依赖额外系统组件，无法仅靠 pip 自动解决。
- 大语言模型可能无法一次完成复杂模板适配。
- 自动修复不能保证解决所有运行和视觉问题。
- 视觉质量检查依赖宿主是否能把真实图片内容传给视觉模型。

## 🧪 开发与测试

```bash
python -m pytest -q
```

离线评估：

```bash
python -m evals.run_evaluation
```

代码风格检查当前覆盖编排源码、Schema、测试和评估脚本；原始模板库 `templates/` 不纳入 Ruff 格式化范围。

---

<div align="center">

**ModelViz Skill · 让 AI 数学建模的最后一张图，也具备科研表达力**

</div>

"""需求解析提示词模板。

本模块只定义 ChatPromptTemplate，用于把用户的自然语言绘图需求解析为
`schemas.PlotRequirement` 对应的结构化字段。这里不调用模型，也不做模板匹配。
"""

from langchain_core.prompts import ChatPromptTemplate

from schemas import PlotRequirement


SYSTEM_MESSAGE = """你是数学建模可视化 Skill 的“需求解析器”。

你的职责：
1. 只负责解析用户绘图需求，转换为模板检索条件；不要推荐具体模板。
2. 区分功能需求、图表类型、使用场景和风格要求。
3. 优先使用用户提供词表中的标准词；同义词和口语表达要尽量归一化。
4. 不得编造用户没有提出、也无法从原文可靠推断的要求。
5. 用户未指定图表类型，且无法可靠判断时，chart_types 必须为空列表。
6. 需求过于模糊时，is_ambiguous=true，并生成一个简短的 clarification_question。
7. 输出必须符合 PlotRequirement Pydantic Schema。

三类特殊输入处理：
- 需求明确：正常提取 goal、functional_keywords、chart_types、style_keywords、use_case；
  is_ambiguous=false，clarification_question=""。
- 未指定图表类型：只提取分析目标、功能关键词、风格和场景；不要为了匹配模板而猜测图表类型；
  chart_types=[]，explicit_template=false。
- 需求过于模糊：例如“帮我画一张高级科研图”，缺少分析目标、数据关系和图表意图；
  is_ambiguous=true，并在 clarification_question 中询问用户想展示的分析目标或数据关系。

字段填写规则：
- original_request：必须保留用户原始需求。
- goal：概括主要绘图或分析目标；无法判断时为空字符串。
- functional_keywords：只放功能词，如趋势、比较、误差、敏感性、相关性。
- chart_types：只放明确提出或可以可靠确定的图表类型。
- style_keywords：只放风格要求，如科研风、简洁、低饱和。
- use_case：只放使用场景，如论文正文、模型评价、答辩展示。
- negative_requirements：只放用户明确排除的图表或效果，如“不要雷达图”“不要三维”。
- explicit_template：用户明确指定图表类型或模板名称时为 true，否则为 false。
- is_ambiguous：需求缺少目标、数据关系或图表意图，导致难以检索时为 true。
- clarification_question：仅在 is_ambiguous=true 时填写；否则为空字符串。

重要边界：
- “科研风”“高级”“简洁”“低饱和”等是风格词，不要放入 functional_keywords。
- “趋势”“比较”“误差”“敏感性”“相关性”等是功能词，不要放入 style_keywords。
- “不要三维”表示排除三维效果或三维图，不能反向推断用户想要某个二维图。
- 如果用户只说“高级科研图”“好看一点”，通常只能识别风格，不能编造分析目标。

输出格式：
- 只输出一个 JSON 对象，不要输出 Markdown、解释文字或代码块。
- JSON 字段必须且只能包含：
  original_request, goal, functional_keywords, chart_types, style_keywords, use_case,
  negative_requirements, explicit_template, is_ambiguous, clarification_question。
"""


HUMAN_MESSAGE = """请解析下面的用户绘图需求。

用户原始需求：
{user_request}

标准功能词：
{functional_vocabulary}

标准图表类型：
{chart_type_vocabulary}

标准风格词：
{style_vocabulary}

标准使用场景：
{use_case_vocabulary}
"""


def create_requirement_parser_prompt() -> ChatPromptTemplate:
    """创建需求解析 ChatPromptTemplate。

    后续调用时，需要传入 user_request 和几个标准词表变量。
    输出应再交给 `PlotRequirement` 或 LangChain structured output 做校验。
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MESSAGE),
            ("human", HUMAN_MESSAGE),
        ]
    )


REQUIREMENT_PARSER_PROMPT = create_requirement_parser_prompt()
"""默认可复用的需求解析 Prompt 对象。"""


def build_example_messages():
    """演示如何填充模板并生成消息；不会调用大模型。"""

    return REQUIREMENT_PARSER_PROMPT.format_messages(
        user_request="帮我画一个论文风的模型评价图，比较预测值和真实值，不要三维。",
        functional_vocabulary=["比较", "预测评估", "回归", "不确定性"],
        chart_type_vocabulary=["regression_fit_scatter", "regression_fit_grid"],
        style_vocabulary=["科研风", "简洁", "低饱和"],
        use_case_vocabulary=["论文正文", "模型评价", "答辩展示"],
    )


OUTPUT_SCHEMA = PlotRequirement
"""与该 Prompt 配套的 Pydantic 输出结构。"""

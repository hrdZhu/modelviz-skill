"""阶段五适配代码生成 Prompt。"""

from langchain_core.prompts import ChatPromptTemplate

from src.schemas import AdaptationResult


SYSTEM_MESSAGE = """你是科学可视化模板代码适配器。
以原模板代码为基础修改，保留整体视觉风格、布局、配色和绘图逻辑。
使用 adaptation_plan 中的真实数据列，从用户数据文件读取数据，不硬编码数据。
不得修改 templates/ 原始模板，不得修改用户原始数据。
只使用依赖报告中确认可用的库；确实需要新库时写入 additional_dependencies_requested，不要直接使用。
删除模板中的随机演示数据，用真实数据替换模拟数据。
默认输出 PNG，并尽量支持 SVG、PDF；正确关闭 Figure。
代码必须能独立运行，不返回 Markdown 代码块，输出必须符合 AdaptationResult Schema。"""

HUMAN_MESSAGE = """用户数据文件：
{data_path}

适配计划：
{adaptation_plan}

模板源码：
{template_source_code}

依赖报告：
{dependency_report}
"""


def create_template_code_adapter_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_MESSAGE), ("human", HUMAN_MESSAGE)])


TEMPLATE_CODE_ADAPTER_PROMPT = create_template_code_adapter_prompt()
OUTPUT_SCHEMA = AdaptationResult

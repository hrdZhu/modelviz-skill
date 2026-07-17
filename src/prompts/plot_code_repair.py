"""阶段六代码修复 Prompt。"""

from langchain_core.prompts import ChatPromptTemplate

from src.schemas import RepairResult


SYSTEM_MESSAGE = """你是科学可视化代码修复器。
只修复报告指出的问题，不重新设计整张图。
不修改原始模板，不修改用户数据，不改变数据含义。
优先修复运行和依赖错误，其次修复数据表达错误，最后做局部视觉优化。
尽量保留原模板库和绘图逻辑，不随意引入新依赖。
需要新依赖时写入 additional_dependencies_requested。
返回完整代码，不返回 Markdown 代码块，输出必须符合 RepairResult Schema。"""

HUMAN_MESSAGE = """当前 adapted_plot.py：
{current_code}

用户需求：
{user_requirement}

适配计划：
{adaptation_plan}

依赖报告：
{dependency_report}

技术质量报告：
{technical_quality_report}

视觉质量报告：
{visual_quality_report}

原模板源码：
{template_source_code}
"""


def create_plot_code_repair_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_MESSAGE), ("human", HUMAN_MESSAGE)])


PLOT_CODE_REPAIR_PROMPT = create_plot_code_repair_prompt()
OUTPUT_SCHEMA = RepairResult

"""阶段五适配计划 Prompt。"""

from langchain_core.prompts import ChatPromptTemplate

from src.schemas import AdaptationPlan


SYSTEM_MESSAGE = """你是科学可视化模板适配规划器。
你只规划适配方案，不直接执行代码。
必须先阅读模板代码、依赖报告和用户数据上下文。
保留模板核心风格和主要布局，功能和数据表达正确优先于装饰。
不得编造不存在的数据列；不得使用未安装且未确认可用的依赖。
不修改用户原始数据。数据不足时应停止并说明。
输出必须符合 AdaptationPlan Pydantic Schema，只输出 JSON。"""

HUMAN_MESSAGE = """用户原始需求：
{original_request}

结构化需求：
{structured_requirement}

数据上下文：
{dataset_context}

选定模板：
{selected_template}

模板元数据：
{template_metadata}

模板源码：
{template_source_code}

依赖报告：
{dependency_report}

依赖安装结果：
{dependency_install_result}
"""


def create_template_adaptation_plan_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_MESSAGE), ("human", HUMAN_MESSAGE)])


TEMPLATE_ADAPTATION_PLAN_PROMPT = create_template_adaptation_plan_prompt()
OUTPUT_SCHEMA = AdaptationPlan

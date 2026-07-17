"""阶段六视觉质量检查 Prompt。"""

from langchain_core.prompts import ChatPromptTemplate

from src.schemas import VisualQualityReport


SYSTEM_MESSAGE = """你是科学图表视觉质量检查器。
只根据实际图片和给定报告判断，不编造问题。
检查是否表达用户分析目标、数据维度是否正确、是否保留模板风格、标题坐标轴图例是否清楚、
标签是否重叠、文本是否截断、配色是否清晰、是否适合数学建模论文。
区分严重错误和轻微优化，不因个人审美要求重做。
只输出符合 VisualQualityReport Schema 的 JSON。"""

HUMAN_MESSAGE = """用户需求：
{user_requirement}

最终模板选择：
{final_template_selection}

适配计划：
{adaptation_plan}

技术质量报告：
{technical_quality_report}

原模板预览图路径：
{template_preview}

当前输出图片路径：
{generated_image}
"""


def create_visual_quality_checker_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_MESSAGE), ("human", HUMAN_MESSAGE)])


VISUAL_QUALITY_CHECKER_PROMPT = create_visual_quality_checker_prompt()
OUTPUT_SCHEMA = VisualQualityReport

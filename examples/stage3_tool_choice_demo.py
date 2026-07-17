"""阶段三 tool_choice 演示。

只用于课程学习：展示如何强制聊天模型调用指定工具，不创建 Agent，不依赖真实 API。
"""

from src.tools import STAGE3_CANDIDATE_TOOLS


def bind_stage3_tools_with_forced_choice(chat_model, tool_name: str = "match_candidate_templates"):
    """绑定阶段三 Tools，并强制模型调用指定工具。

    OpenAI 兼容模型通常支持 `tool_choice=tool_name`。
    有些供应商需要：
    `tool_choice={"type": "function", "function": {"name": tool_name}}`。
    """

    return chat_model.bind_tools(STAGE3_CANDIDATE_TOOLS, tool_choice=tool_name)


def example_prompt() -> str:
    """示例提示词，不会真实调用模型。"""

    return "请调用 match_candidate_templates 工具召回候选模板，不要直接编造候选列表。"

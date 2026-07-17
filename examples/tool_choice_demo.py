"""tool_choice 学习示例。

这个文件只演示如何把阶段三核心 tools 绑定到聊天模型，并强制模型调用指定工具。
不创建 Agent，不调用真实模型 API，测试也不依赖这个示例。
"""

from src.tools import STAGE3_CANDIDATE_TOOLS


def bind_tools_with_forced_choice(chat_model, tool_name: str = "match_candidate_templates"):
    """绑定工具并强制调用指定 tool。

    许多 OpenAI 兼容模型可使用 `tool_choice=tool_name`。
    部分 provider 需要写成：
    `tool_choice={"type": "function", "function": {"name": tool_name}}`。

    这样做的用途是防止模型绕过工具直接输出模板推荐。
    """

    return chat_model.bind_tools(
        STAGE3_CANDIDATE_TOOLS,
        tool_choice=tool_name,
    )


def example_prompt_for_forced_tool_call() -> str:
    """返回一个示例提示词，不会请求模型。"""

    return (
        "请调用 match_candidate_templates 工具，根据已加载的用户需求和模板目录计算候选模板。"
        "不要直接编造候选列表。"
    )

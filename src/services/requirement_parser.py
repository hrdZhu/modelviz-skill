"""
需求解析服务。
这里把 Prompt、结构化模型、Pydantic 校验和 JSON 保存串起来。
真实项目中传入支持 `with_structured_output(PlotRequirement)` 的 ChatModel 即可。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from schemas import PlotRequirement
from src.prompts.requirement_parser import REQUIREMENT_PARSER_PROMPT


VOCABULARY_PATH = Path("docs/requirement_vocabulary.yaml")
DEFAULT_OUTPUT_PATH = Path("workspace/user_requirement.json")


def load_requirement_vocabulary(path: Path | str = VOCABULARY_PATH) -> dict[str, Any]:
    """读取标准词表 YAML，供 Prompt 变量使用。"""

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_prompt_variables(vocabulary: dict[str, Any]) -> dict[str, Any]:
    """从完整词表中取出需求解析 Prompt 需要的几类词。"""

    return {
        "functional_vocabulary": list(vocabulary.get("functional_keywords", {}).keys()),
        "chart_type_vocabulary": vocabulary.get("chart_types", {}),
        "style_vocabulary": list(vocabulary.get("style_keywords", {}).keys()),
        "use_case_vocabulary": list(vocabulary.get("use_cases", {}).keys()),
    }


def _terms_with_synonyms(vocabulary: dict[str, Any], section: str) -> list[str]:
    """提取某类标准词和同义词，供确定性后处理使用。"""

    terms: list[str] = []
    for standard_term, config in vocabulary.get(section, {}).items():
        terms.append(standard_term)
        terms.extend(config.get("synonyms", []))
    return _dedupe_non_empty(terms)


def _dedupe_non_empty(values: list[str]) -> list[str]:
    """去重并移除空字符串，同时保留原有顺序。"""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def _prepare_raw_result(raw_result: Any, original_request: str) -> dict[str, Any]:
    """在 Pydantic 校验前做最小机械清洗。

    这里不推断新信息，只处理模型输出中常见的空字符串和原始需求缺失。
    """

    if isinstance(raw_result, PlotRequirement):
        data = raw_result.model_dump()
    else:
        data = dict(raw_result)

    data["original_request"] = original_request
    for field in (
        "functional_keywords",
        "chart_types",
        "style_keywords",
        "negative_requirements",
    ):
        data[field] = _dedupe_non_empty(data.get(field, []))
    return data


def postprocess_requirement(
    requirement: PlotRequirement,
    *,
    original_request: str,
    style_vocabulary: list[str] | None = None,
) -> PlotRequirement:
    """对模型输出做少量确定性清理。

    只做格式清理，不新增模型没有解析出的功能或图表类型。
    """

    style_terms = set(style_vocabulary or [])
    functional_keywords = _dedupe_non_empty(requirement.functional_keywords)
    if style_terms:
        # 避免把“科研风、简洁、低饱和”等风格词误放到功能关键词中。
        functional_keywords = [item for item in functional_keywords if item not in style_terms]

    chart_types = _dedupe_non_empty(requirement.chart_types)
    is_ambiguous = requirement.is_ambiguous or not functional_keywords and not chart_types
    clarification_question = requirement.clarification_question.strip()
    if is_ambiguous and not clarification_question:
        clarification_question = "请说明你想展示的分析目标、数据关系或图表类型。"
    if not is_ambiguous:
        clarification_question = ""
    goal = requirement.goal
    if is_ambiguous and not functional_keywords and not chart_types:
        goal = ""

    return PlotRequirement.model_validate(
        {
            **requirement.model_dump(),
            "original_request": original_request,
            "goal": goal,
            "functional_keywords": functional_keywords,
            "chart_types": chart_types,
            "style_keywords": _dedupe_non_empty(requirement.style_keywords),
            "negative_requirements": _dedupe_non_empty(requirement.negative_requirements),
            "is_ambiguous": is_ambiguous,
            "clarification_question": clarification_question,
        }
    )


def save_requirement(
    requirement: PlotRequirement,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> Path:
    """保存结构化需求到 JSON 文件。

    保存前再次进行 Pydantic 校验；JSON 中只写 Schema 字段。
    """

    validated = PlotRequirement.model_validate(requirement.model_dump())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(validated.model_dump(), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def _structured_model(model: Any):
    """优先使用 LangChain 的结构化输出接口。"""

    if hasattr(model, "with_structured_output"):
        return model.with_structured_output(PlotRequirement)
    return model


def parse_and_save_requirement(
    user_request: str,
    model: Any,
    *,
    vocabulary_path: Path | str = VOCABULARY_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
) -> PlotRequirement:
    """解析用户需求并保存为 `workspace/user_requirement.json`。

    参数 `model` 可以是真实 ChatModel，也可以是测试用的 Runnable。
    本函数不做模板匹配或模板推荐。
    """

    vocabulary = load_requirement_vocabulary(vocabulary_path)
    prompt_variables = build_prompt_variables(vocabulary)
    chain = REQUIREMENT_PARSER_PROMPT | _structured_model(model)
    raw_result = chain.invoke({"user_request": user_request, **prompt_variables})
    requirement = PlotRequirement.model_validate(_prepare_raw_result(raw_result, user_request))
    requirement = postprocess_requirement(
        requirement,
        original_request=user_request,
        style_vocabulary=_terms_with_synonyms(vocabulary, "style_keywords"),
    )
    save_requirement(requirement, output_path)
    return requirement

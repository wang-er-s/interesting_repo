import json
import os
import time
from typing import Any

from openai import OpenAI


AI_MAX_INPUT_CHARS = 6000
AI_REQUIRED_KEYS = ("summary", "category")
CATEGORY_SEPARATOR = " / "


def get_ai_config() -> dict[str, str] | None:
    provider = os.environ.get("AI_PROVIDER", "").strip() or "openai-response"
    model = os.environ.get("AI_MODEL", "").strip()
    api_key = os.environ.get("AI_API_KEY", "").strip()

    if not model or not api_key:
        return None

    base_url = os.environ.get("AI_BASE_URL", "").strip()
    if not base_url:
        base_url = "https://api.openai.com/v1"

    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }


def split_category_parts(category: str) -> list[str]:
    return [part.strip() for part in str(category).replace("／", "/").split("/") if part.strip()]


def format_category(category: str) -> str:
    parts = split_category_parts(category)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}{CATEGORY_SEPARATOR}{' / '.join(parts[1:])}"


def extract_parent_categories(categories: list[str]) -> list[str]:
    parents: list[str] = []
    for category in categories:
        parts = split_category_parts(category)
        if parts and parts[0] not in parents:
            parents.append(parts[0])
    return parents


def build_ai_prompt(
    repo: dict[str, Any],
    readme_text: str,
    custom_notes: str,
    prompt_source: str,
    fallback_category: str,
    allowed_categories: list[str],
) -> str:
    topics = [str(topic).strip() for topic in (repo.get("topics") or []) if str(topic).strip()]
    normalized_categories = [format_category(category) for category in allowed_categories if format_category(category)]
    parent_categories = extract_parent_categories(normalized_categories)
    return (
        "你是一个 GitHub 仓库整理助手。"
        "请基于输入内容输出严格 JSON，不要输出 markdown，不要输出额外解释。"
        "JSON 格式必须是："
        '{"summary":"一句中文简介","category":"大类 / 小类"}。'
        "要求：summary 不超过 50 个中文字符；category 简短。"
        "category 优先从已有分类中选择完全一致的一项。"
        "只有已有分类都不合适时，才允许新建分类。"
        "新分类必须使用“大类 / 小类”格式，大类优先复用已有大类。"
        "不要输出单级分类，不要输出“未分类”。"
        f"\n仓库名: {repo['full_name']}"
        f"\n仓库描述: {repo.get('description') or ''}"
        f"\n已有分类: {', '.join(normalized_categories) or '无'}"
        f"\n可优先复用大类: {', '.join(parent_categories) or '无'}"
        f"\n兜底候选分类: {format_category(fallback_category) or '未分类 / 其他'}"
        f"\n仓库 topics: {', '.join(topics)}"
        f"\n自定义备注: {custom_notes}"
        f"\nREADME 摘要:\n{prompt_source[:AI_MAX_INPUT_CHARS]}"
    )


def build_retry_prompt(previous_prompt: str, error_message: str) -> str:
    return (
        f"{previous_prompt}\n\n"
        "你上一次的输出不符合要求，请重新输出。"
        "必须只返回一个 JSON 对象，不能有任何额外文字。"
        f"错误原因: {error_message}"
    )


def extract_json_from_text(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("AI 返回内容中未找到 JSON")
    return json.loads(text[start : end + 1])


def create_openai_client(config: dict[str, str]) -> OpenAI:
    base_url = config["base_url"].rstrip("/")
    if base_url.endswith("/responses"):
        base_url = base_url[: -len("/responses")]

    return OpenAI(api_key=config["api_key"], base_url=base_url)


def call_openai_response_api(config: dict[str, str], prompt: str) -> dict[str, Any]:
    client = create_openai_client(config)
    provider = config.get("provider", "openai-response")

    if provider == "openai-chat":
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        output_text = response.choices[0].message.content or ""
        if not output_text.strip():
            raise ValueError("AI 响应为空")
        return extract_json_from_text(output_text)

    response = client.responses.create(
        model=config["model"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        store=False,
        stream=True,
    )

    output_text = ""
    for event in response:
        if getattr(event, "type", "") == "response.output_text.delta":
            delta = getattr(event, "delta", "")
            if delta:
                output_text += delta
    if not output_text.strip():
        raise ValueError("AI 响应为空")
    return extract_json_from_text(output_text)


def validate_ai_result(ai_result: dict[str, Any]) -> None:
    if not isinstance(ai_result, dict):
        raise ValueError("AI 返回结果必须是 JSON 对象")

    missing_keys = [key for key in AI_REQUIRED_KEYS if key not in ai_result]
    if missing_keys:
        raise ValueError(f"AI 返回结果缺少字段: {', '.join(missing_keys)}")

    if not isinstance(ai_result.get("summary"), str) or not ai_result["summary"].strip():
        raise ValueError("summary 必须是非空字符串")

    if not isinstance(ai_result.get("category"), str) or not ai_result["category"].strip():
        raise ValueError("category 必须是非空字符串")


def normalize_ai_result(
    ai_result: dict[str, Any],
    fallback_category: str,
    allowed_categories: list[str],
) -> dict[str, Any]:
    summary = str(ai_result.get("summary", "")).strip()
    raw_category = str(ai_result.get("category", "")).strip()
    normalized_allowed = {
        format_category(category): category
        for category in allowed_categories
        if format_category(category)
    }
    fallback_category = format_category(fallback_category) or "未分类 / 其他"
    normalized_category = format_category(raw_category)

    if normalized_category in normalized_allowed:
        category = normalized_allowed[normalized_category]
    else:
        parts = split_category_parts(normalized_category or fallback_category)
        fallback_parts = split_category_parts(fallback_category)
        known_parents = set(extract_parent_categories(list(normalized_allowed)))
        if len(parts) >= 2:
            category = f"{parts[0]}{CATEGORY_SEPARATOR}{' / '.join(parts[1:])}"
        elif len(parts) == 1:
            if parts[0] in known_parents:
                category = f"{parts[0]}{CATEGORY_SEPARATOR}其他"
            elif fallback_parts:
                category = f"{fallback_parts[0]}{CATEGORY_SEPARATOR}{parts[0]}"
            else:
                category = f"{parts[0]}{CATEGORY_SEPARATOR}其他"
        else:
            category = fallback_category

    return {
        "summary": summary,
        "category": category,
    }


def generate_ai_summary(
    config: dict[str, str] | None,
    repo: dict[str, Any],
    readme_text: str,
    custom_notes: str,
    prompt_source: str,
    fallback_category: str,
    allowed_categories: list[str],
    log,
) -> dict[str, Any] | None:
    if not config:
        return None

    prompt = build_ai_prompt(repo, readme_text, custom_notes, prompt_source, fallback_category, allowed_categories)
    log(f"调用 AI 生成摘要: {repo['full_name']} ({config['model']})")

    current_prompt = prompt
    for attempt in range(1, 4):
        try:
            result = call_openai_response_api(config, current_prompt)
            validate_ai_result(result)
            return normalize_ai_result(result, fallback_category, allowed_categories)
        except Exception as exc:
            log(f"AI 生成失败，第 {attempt} 次: {repo['full_name']} -> {exc}")
            if attempt == 3:
                return None
            current_prompt = build_retry_prompt(prompt, str(exc))
            time.sleep(1)

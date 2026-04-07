import json
import os
import time
from typing import Any

from openai import OpenAI


AI_MAX_INPUT_CHARS = 6000
AI_REQUIRED_KEYS = ("summary", "category")


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


def build_ai_prompt(
    repo: dict[str, Any],
    readme_text: str,
    custom_notes: str,
    prompt_source: str,
    allowed_categories: list[str],
) -> str:
    topics = [str(topic).strip() for topic in (repo.get("topics") or []) if str(topic).strip()]
    return (
        "你是一个 GitHub 仓库整理助手。"
        "请基于输入内容输出严格 JSON，不要输出 markdown，不要输出额外解释。"
        "JSON 格式必须是："
        '{"summary":"一句中文简介","category":"分类"}。'
        "要求：summary 不超过 50 个中文字符；category 简短。"
        "category 优先复用已有分类，没有合适项时再生成新分类。"
        f"\n仓库名: {repo['full_name']}"
        f"\n仓库描述: {repo.get('description') or ''}"
        f"\n可优先复用分类: {', '.join(allowed_categories)}"
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

def normalize_ai_result(ai_result: dict[str, Any], fallback_category: str) -> dict[str, Any]:
    summary = str(ai_result.get("summary", "")).strip()
    category = str(ai_result.get("category", "")).strip() or fallback_category

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

    prompt = build_ai_prompt(repo, readme_text, custom_notes, prompt_source, allowed_categories)
    log(f"调用 AI 生成摘要: {repo['full_name']} ({config['model']})")

    current_prompt = prompt
    for attempt in range(1, 4):
        try:
            result = call_openai_response_api(config, current_prompt)
            validate_ai_result(result)
            return normalize_ai_result(result, fallback_category)
        except Exception as exc:
            log(f"AI 生成失败，第 {attempt} 次: {repo['full_name']} -> {exc}")
            if attempt == 3:
                return None
            current_prompt = build_retry_prompt(prompt, str(exc))
            time.sleep(1)

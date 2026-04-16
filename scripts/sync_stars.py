import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_client import generate_ai_summary, get_ai_config


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_FILE = DATA_DIR / "processed.json"
CUSTOM_REPOS_FILE = DATA_DIR / "custom_repos.json"
README_FILE = ROOT / "README.md"
GITHUB_API = "https://api.github.com"
README_MAX_CHARS = 10000
ENV_FILE = ROOT / ".env"
CATEGORY_SEPARATOR = " / "
DEFAULT_CATEGORY = "未分类 / 其他"


def log(message: str) -> None:
    print(f"[sync] {message}")


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ[key] = value


def split_category_parts(category: str) -> list[str]:
    return [part.strip() for part in str(category).replace("／", "/").split("/") if part.strip()]


def normalize_category(category: str) -> str:
    parts = split_category_parts(category)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}{CATEGORY_SEPARATOR}{' / '.join(parts[1:])}"


def load_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def github_request(path: str, token: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
    url = f"{GITHUB_API}{path}"
    if params:
        query = urllib.parse.urlencode(params)
        url = f"{url}?{query}"

    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "interesting-repo-sync",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
            headers = {key.lower(): value for key, value in response.headers.items()}
            return json.loads(payload), headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"GitHub API 请求失败: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API 网络请求失败: {exc}") from exc


def fetch_starred_repositories(token: str) -> list[dict[str, Any]]:
    repos: list[dict[str, Any]] = []
    page = 1
    while True:
        log(f"正在拉取 Star 仓库，第 {page} 页")
        payload, _ = github_request("/user/starred", token, {"per_page": 100, "page": page})
        if not payload:
            break
        repos.extend(payload)
        log(f"第 {page} 页完成，累计 {len(repos)} 个 Star 仓库")
        if len(payload) < 100:
            break
        page += 1
    return repos


def normalize_repo_name(repo_value: str) -> str:
    trimmed = repo_value.strip()
    if not trimmed:
        raise ValueError("repo 不能为空")

    if trimmed.startswith("http://") or trimmed.startswith("https://"):
        parsed = urllib.parse.urlparse(trimmed)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"无法从 URL 解析仓库名: {repo_value}")
        return f"{parts[0]}/{parts[1].removesuffix('.git')}"

    parts = [part for part in trimmed.split("/") if part]
    if len(parts) != 2:
        raise ValueError(f"仓库名格式必须是 owner/repo: {repo_value}")
    return f"{parts[0]}/{parts[1]}"


def load_custom_repositories() -> list[dict[str, Any]]:
    raw_items = load_json_file(CUSTOM_REPOS_FILE, [])
    if not isinstance(raw_items, list):
        raise ValueError("data/custom_repos.json 必须是数组")

    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"custom_repos.json 第 {index} 项必须是对象")
        if "repo" not in item:
            raise ValueError(f"custom_repos.json 第 {index} 项缺少 repo 字段")

        normalized_items.append(
            {
                "repo": normalize_repo_name(str(item["repo"])),
                "category": normalize_category(str(item.get("category", "")).strip()),
                "tags": item.get("tags", []),
                "notes": item.get("notes", ""),
            }
        )
    return normalized_items


def fetch_repository_metadata(token: str, full_name: str) -> dict[str, Any]:
    log(f"拉取仓库元数据: {full_name}")
    payload, _ = github_request(f"/repos/{full_name}", token)
    return payload


def fetch_repository_readme(token: str, full_name: str) -> str:
    try:
        log(f"拉取 README: {full_name}")
        payload, _ = github_request(f"/repos/{full_name}/readme", token)
    except RuntimeError:
        log(f"README 不可用，跳过: {full_name}")
        return ""

    content = payload.get("content", "")
    encoding = payload.get("encoding")
    if not content or encoding != "base64":
        return ""

    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")[:README_MAX_CHARS]
    except ValueError:
        return ""


def build_prompt_source(repo: dict[str, Any], readme_text: str, custom_notes: str) -> str:
    parts = [
        repo.get("description") or "",
        " ".join(repo.get("topics") or []),
        custom_notes or "",
        readme_text or "",
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def infer_category(repo: dict[str, Any], combined_text: str, custom_category: str) -> str:
    if custom_category:
        return normalize_category(custom_category)

    text = combined_text.lower()
    topics = [topic.lower() for topic in repo.get("topics") or []]

    keyword_map = [
        ("AI / Agent 智能体", ["agentic", "multi-agent", "ai agent", "agent", "autogen"]),
        ("AI / LLM 开发框架", ["llm framework", "llm", "rag", "langchain", "semantic kernel", "workflow"]),
        ("AI / 助手/聊天", ["assistant", "chatbot", "chat", "copilot"]),
        ("AI / 图像生成", ["image generation", "text2image", "image2image", "diffusion", "stable diffusion"]),
        ("AI / 视频/字幕", ["subtitle", "caption", "dubbing", "video translation"]),
        ("AI / 学习与提示词", ["prompt", "cookbook", "tutorial", "course", "guide"]),
        ("AI / 集成/平台", ["mcp", "integration", "platform", "gateway", "context engineering"]),
        ("Unity / 插件", ["unity plugin", "unitypackage", "upm"]),
        ("Unity / 框架", ["unity framework", "mvvm", "data binding", "gas"]),
        ("Unity / 资源管理", ["assetbundle", "addressables", "resource management", "hot update"]),
        ("Unity / 工具", ["unity", "unity3d", "editor tool", "toolchain"]),
        ("开发工具 / CLI 工具", ["cli", "terminal", "shell", "command line"]),
        ("开发工具 / 代码编辑器", ["editor", "ide", "neovim", "vscode"]),
        ("开发工具 / 代码处理", ["code review", "search", "parser", "repo", "linter", "formatter"]),
        ("应用框架 / 桌面框架", ["tauri", "electron", "desktop framework"]),
        ("应用框架 / Web 框架", ["web framework", "fastapi", "django", "flask", "spring", "asp.net"]),
        ("应用框架 / 游戏框架", ["game engine", "gamedev", "game framework"]),
        ("媒体处理 / 视频处理", ["video", "iptv", "player", "stream"]),
        ("媒体处理 / 音频处理", ["audio", "speech", "voice", "asr", "tts"]),
        ("媒体处理 / 图片处理", ["image compression", "image", "ocr"]),
        ("媒体处理 / 文档转换", ["markdown", "pdf", "document", "epub", "html"]),
        ("知识管理 / 笔记工具", ["note", "knowledge base", "wiki", "obsidian"]),
        ("知识管理 / 阅读器", ["reader", "ebook", "rss"]),
        ("知识管理 / 文档管理", ["document management", "archive", "translation"]),
        ("教育学习 / 语言学习", ["language learning", "english", "speaking"]),
        ("教育学习 / 课程资源", ["textbook", "course", "codelab", "tutorial"]),
        ("教育学习 / 职业指南", ["career", "college", "cook", "guide"]),
        ("效率工具 / 通讯工具", ["messaging", "chat app", "im", "communication"]),
        ("效率工具 / 网盘工具", ["cloud drive", "webdav", "backup", "file list"]),
        ("效率工具 / 自动化工具", ["automation", "workflow", "scheduler"]),
        ("资源合集", ["awesome", "collection", "curated", "top charts", "best"]),
    ]

    for category, keywords in keyword_map:
        if any(keyword in text for keyword in keywords) or any(keyword in topics for keyword in keywords):
            return category
    return DEFAULT_CATEGORY


def infer_tags(repo: dict[str, Any], custom_tags: list[Any]) -> list[str]:
    tags: list[str] = []
    for tag in custom_tags:
        if isinstance(tag, str) and tag.strip():
            tags.append(tag.strip())

    for topic in repo.get("topics") or []:
        if topic and topic not in tags:
            tags.append(topic)

    return tags[:8]


def normalize_string_list(values: list[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if isinstance(value, str) and value.strip() and value.strip() not in normalized:
            normalized.append(value.strip())
    return normalized


def build_known_categories(processed_items: list[Any], custom_repos: list[dict[str, Any]]) -> list[str]:
    categories: list[str] = []
    for item in processed_items:
        if isinstance(item, dict):
            category = normalize_category(str(item.get("category", "")).strip())
            if category:
                categories.append(category)
    for item in custom_repos:
        category = normalize_category(str(item.get("category", "")).strip())
        if category:
            categories.append(category)
    return normalize_string_list(categories)


def infer_summary(repo: dict[str, Any], readme_text: str, custom_notes: str, category: str) -> str:
    if custom_notes:
        return custom_notes.strip()

    description = (repo.get("description") or "").strip()
    if description:
        return description

    if readme_text:
        first_line = next((line.strip("# ").strip() for line in readme_text.splitlines() if line.strip()), "")
        if first_line:
            return f"{first_line}，归类为{category}。"

    return f"{repo['full_name']}，归类为{category}。"


def merge_source(existing_source: str, new_source: str) -> str:
    sources = {part.strip() for part in (existing_source or "").split(",") if part.strip()}
    sources.add(new_source)
    return ",".join(sorted(sources))


def sort_category_key(category: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in category.split("/") if part.strip())


def split_category(category: str) -> tuple[str, str]:
    parts = [part.strip() for part in category.split("/") if part.strip()]
    if not parts:
        return "未分类", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], "/".join(parts[1:])


def build_heading_anchor(text: str) -> str:
    return text.replace(" ", "")


def build_cached_entry(
    repo: dict[str, Any],
    source: str,
    custom_item: dict[str, Any] | None = None,
    existing_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not existing_item:
        return {}

    custom_item = custom_item or {}
    existing_category = str(existing_item.get("category", "")).strip()
    existing_summary = str(existing_item.get("summary", "")).strip()
    existing_tags = normalize_string_list(existing_item.get("tags", []))
    custom_category = normalize_category(str(custom_item.get("category", "")).strip())
    custom_tags = normalize_string_list(custom_item.get("tags", []))
    custom_notes = str(custom_item.get("notes", "")).strip()
    synced_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {
        "full_name": repo["full_name"],
        "url": repo["html_url"],
        "source": merge_source(existing_item.get("source", ""), source),
        "category": custom_category or normalize_category(existing_category) or DEFAULT_CATEGORY,
        "summary": custom_notes or existing_summary,
        "tags": custom_tags or existing_tags,
        "notes": custom_notes,
        "description": (repo.get("description") or "").strip(),
        "topics": repo.get("topics") or [],
        "language": repo.get("language") or "",
        "synced_at": synced_at,
    }


def build_processed_entry(
    token: str,
    ai_config: dict[str, str] | None,
    repo: dict[str, Any],
    source: str,
    known_categories: list[str],
    custom_item: dict[str, Any] | None = None,
    existing_item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    custom_item = custom_item or {}
    log(f"处理仓库: {repo['full_name']} ({source})")
    readme_text = fetch_repository_readme(token, repo["full_name"])
    prompt_source = build_prompt_source(repo, readme_text, str(custom_item.get("notes", "")))
    existing_category = normalize_category(str(existing_item.get("category", "")).strip()) if existing_item else ""
    existing_summary = str(existing_item.get("summary", "")).strip() if existing_item else ""
    existing_tags = normalize_string_list(existing_item.get("tags", [])) if existing_item else []
    custom_category = normalize_category(str(custom_item.get("category", "")).strip())
    custom_tags = normalize_string_list(custom_item.get("tags", []))
    category = custom_category or existing_category or infer_category(repo, prompt_source, custom_category)
    tags = custom_tags or existing_tags or infer_tags(repo, custom_tags)
    summary = str(custom_item.get("notes", "")).strip() or existing_summary or infer_summary(
        repo,
        readme_text,
        str(custom_item.get("notes", "")),
        category,
    )
    description = (repo.get("description") or "").strip()
    allowed_categories = normalize_string_list([custom_category, existing_category, *known_categories])
    ai_result = None
    if not existing_summary:
        ai_result = generate_ai_summary(
            ai_config,
            repo,
            readme_text,
            str(custom_item.get("notes", "")),
            prompt_source,
            category,
            allowed_categories,
            log,
        )
    if ai_result:
        category = custom_category or existing_category or ai_result["category"]
        category = normalize_category(category) or DEFAULT_CATEGORY
        summary = str(custom_item.get("notes", "")).strip() or ai_result["summary"] or summary
    elif ai_config and not existing_summary:
        log(f"AI 生成失败，跳过写入 processed: {repo['full_name']}")
        return {}
    category = normalize_category(category) or DEFAULT_CATEGORY
    synced_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {
        "full_name": repo["full_name"],
        "url": repo["html_url"],
        "source": merge_source(existing_item.get("source", "") if existing_item else "", source),
        "category": category,
        "summary": summary,
        "tags": tags,
        "notes": str(custom_item.get("notes", "")),
        "description": description,
        "topics": repo.get("topics") or [],
        "language": repo.get("language") or "",
        "synced_at": synced_at,
    }


def render_readme(items: list[dict[str, Any]]) -> str:
    category_map: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for item in items:
        category = str(item.get("category", "")).strip() or "未分类"
        parent_category, child_category = split_category(category)
        category_map.setdefault(parent_category, {}).setdefault(child_category, []).append(item)

    lines = [
        "# GitHub 仓库收藏索引",
        "",
        f"共收录 {len(items)} 个仓库。",
        "",
        "## 目录",
        "",
    ]

    for parent_category in sorted(category_map, key=sort_category_key):
        lines.append(f"- [{parent_category}](#{build_heading_anchor(parent_category)})")
        for child_category in sorted(category_map[parent_category], key=sort_category_key):
            if not child_category:
                continue
            lines.append(f"  - [{child_category}](#{build_heading_anchor(child_category)})")

    lines.extend(["", "## 仓库列表", ""])

    for parent_category in sorted(category_map, key=sort_category_key):
        lines.append(f"## {parent_category}")
        lines.append("")
        for child_category in sorted(category_map[parent_category], key=sort_category_key):
            if child_category:
                lines.append(f"### {child_category}")
                lines.append("")
            for item in sorted(category_map[parent_category][child_category], key=lambda repo: repo["full_name"].lower()):
                repo_name = item["full_name"].split("/")[-1]
                summary = str(item.get("summary", "")).replace("\n", " ").strip()
                if summary:
                    lines.append(f"- [{repo_name}]({item['url']}) {summary}")
                else:
                    lines.append(f"- [{repo_name}]({item['url']})")
            lines.append("")
        lines.append("")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    load_dotenv_file(ENV_FILE)
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("缺少环境变量 GITHUB_TOKEN", file=sys.stderr)
        return 1
    ai_config = get_ai_config()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log("开始同步仓库索引")
    if ENV_FILE.exists():
        log("已加载 .env 配置")
    if ai_config:
        log(f"已启用 AI 摘要: {ai_config['provider']} / {ai_config['model']}")
    else:
        log("未配置 AI，使用规则摘要兜底")

    processed_items = load_json_file(PROCESSED_FILE, [])
    processed_map = {
        item["full_name"]: item for item in processed_items if isinstance(item, dict) and item.get("full_name")
    }
    log(f"已加载 {len(processed_map)} 条历史记录")

    starred_repos = fetch_starred_repositories(token)
    custom_repos = load_custom_repositories()
    known_categories = build_known_categories(processed_items, custom_repos)
    log(f"已读取 {len(custom_repos)} 条自定义仓库")

    combined_repos: dict[str, dict[str, Any]] = {}
    custom_map = {item["repo"]: item for item in custom_repos}

    for repo in starred_repos:
        combined_repos[repo["full_name"]] = {
            "repo": repo,
            "source": "starred",
            "custom_item": custom_map.get(repo["full_name"]),
        }

    for custom_item in custom_repos:
        if custom_item["repo"] in combined_repos:
            combined_repos[custom_item["repo"]]["source"] = merge_source(
                combined_repos[custom_item["repo"]]["source"],
                "custom",
            )
            combined_repos[custom_item["repo"]]["custom_item"] = custom_item
            continue

        metadata = fetch_repository_metadata(token, custom_item["repo"])
        combined_repos[custom_item["repo"]] = {
            "repo": metadata,
            "source": "custom",
            "custom_item": custom_item,
        }

    updated_items: list[dict[str, Any]] = []
    total_repos = len(combined_repos)
    log(f"去重后共 {total_repos} 个仓库待处理")
    for index, full_name in enumerate(sorted(combined_repos), start=1):
        payload = combined_repos[full_name]
        existing_item = processed_map.get(full_name)
        log(f"进度 {index}/{total_repos}")
        if existing_item:
            entry = build_cached_entry(
                repo=payload["repo"],
                source=payload["source"],
                custom_item=payload.get("custom_item"),
                existing_item=existing_item,
            )
        else:
            entry = build_processed_entry(
                token=token,
                ai_config=ai_config,
                repo=payload["repo"],
                source=payload["source"],
                known_categories=known_categories,
                custom_item=payload.get("custom_item"),
                existing_item=existing_item,
            )
        if entry:
            updated_items.append(entry)
            entry_category = normalize_category(str(entry.get("category", "")).strip())
            if entry_category and entry_category not in known_categories:
                known_categories.append(entry_category)

    save_json_file(PROCESSED_FILE, updated_items)
    log(f"已写入 {PROCESSED_FILE.relative_to(ROOT)}")
    README_FILE.write_text(render_readme(updated_items), encoding="utf-8", newline="\n")
    log("已写入 README.md")
    print(f"同步完成，共处理 {len(updated_items)} 个仓库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

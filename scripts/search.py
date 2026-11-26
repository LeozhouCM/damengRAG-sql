# 本文件做什么：1B 单条查询 CLI。人工验证“角色选子块、父块补上下文、证据单独取”是否符合预期。
# 被谁使用：人工执行；不参与评测脚本。
# 运行：
#   python scripts/search.py --query "这条 SQL 为什么没走索引，应该怎么优化？"
#   python scripts/search.py --query "..." --mode global
#   python scripts/search.py --query "..." --roles 原因,办法
#   python scripts/search.py --query "..." --json
# 依据：实践计划书 表42 的检索流程；角色/父块分工见 src/knowledge/retrieval.py 文件头。
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.config import load_settings  # noqa: E402
from src.knowledge.retrieval import load_search_index, search  # noqa: E402


def snippet(text: str, limit: int = 180) -> str:
    """把多行原文压成一行短摘要，避免 CLI 输出被长计划刷屏。"""
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "…"


def print_global(result: dict) -> None:
    print(f"query: {result['query']}")
    print(f"mode: {result['mode']}")
    print(f"inferred_roles: {result.get('inferred_roles', [])}")
    print(f"elapsed_ms: {result.get('elapsed_ms')}")
    print("results:")
    for index, item in enumerate(result.get("results", []), start=1):
        print(
            f"  {index}. {item['chunk_id']} | {item['section_role']} | "
            f"{item['content_type']} | rrf={item['score']:.6f} | "
            f"dense={item.get('dense_rank')} bm25={item.get('bm25_rank')}"
        )
        print(f"     {item['section_path']}")
        print(f"     {snippet(item.get('text', ''))}")


def print_role(result: dict, show_context: bool) -> None:
    print(f"query: {result['query']}")
    print(f"mode: {result['mode']}")
    print(f"inferred_roles: {result['inferred_roles']}")
    print(f"requested_roles: {result['requested_roles']}")
    print(f"root: {result['root_id']} | {result['root_title']}")
    print(
        "root_debug: coverage={coverage} covered_roles={covered_roles} "
        "score_sum={score_sum:.6f}".format(**result["root_debug"])
    )
    print("selected:")
    for role, items in result["selected"].items():
        if not items:
            print(f"  [{role}] (none)")
            continue
        print(f"  [{role}]")
        for item in items:
            print(
                f"    - {item['chunk_id']} | {item['content_type']} | "
                f"score={item['score']:.6f} | rank={item['rank']} | {item['source']}"
            )
            print(f"      {item['section_path']}")
    print("role_contexts:")
    for context in result["role_contexts"]:
        print(f"  [{context['role']}] chars={context['chars']}")
        for item in context["chunks"]:
            marker = "HIT" if item["is_hit"] else f"+{item['distance']}"
            print(
                f"    - {marker} {item['chunk_id']} | {item['chunk_index']} | "
                f"{item['content_type']} | {item['section_path']}"
            )
            if show_context:
                print(f"      {snippet(item['text'], 1000)}")
    print("evidence:")
    if not result["evidence"]:
        print("  (none)")
    for item in result["evidence"]:
        print(
            f"  - {item['chunk_id']} | {item['content_type']} | "
            f"from_selected_parent={item['from_selected_parent']} | {item['section_path']}"
        )
        print(f"    {snippet(item['text'], 400)}")
    print(f"missing_roles: {result['missing_roles']}")
    print(f"backfilled_roles: {result['backfilled_roles']}")
    print(
        f"context_chars: {result['context_chars']} "
        f"(role={result['role_context_chars']}, evidence={result['evidence_chars']})"
    )
    print(f"recall_debug: {result['recall_debug']}")
    print(f"elapsed_ms: {result['elapsed_ms']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="1B 单条查询")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["role", "global"], default="role")
    parser.add_argument(
        "--roles",
        default=None,
        help="逗号分隔的角色；不传则用问句规则推断，传空字符串表示走全局兜底",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--show-context", action="store_true", help="打印命中上下文原文")
    parser.add_argument("--json", action="store_true", help="打印完整 JSON 结果")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    settings = load_settings(config_path)
    index = load_search_index(ROOT, settings)
    roles = None
    if args.roles is not None:
        roles = [item.strip() for item in args.roles.split(",") if item.strip()]
    result = search(index, args.query, mode=args.mode, required_roles=roles, top_k=args.top_k)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["mode"] in {"global", "global_fallback"}:
        print_global(result)
    else:
        print_role(result, args.show_context)


if __name__ == "__main__":
    main()

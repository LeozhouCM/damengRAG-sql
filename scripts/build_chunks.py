"""1A 构建入口：来源登记 + 清洗 + 切块 + 打标签（本文件负责串起整条流水线）。

本文件做什么：
- 扫描 data/raw/dm_cases/（案例）和 data/raw/dm_docs/（手册样章）下的 Markdown；
- 登记来源（类型、标题、文件名、本地 URL、版本、H1 类型）到 data/registry/sources.json；
- 清洗原文，写到 data/interim/cleaned/，方便人工复核；
- 调用 src/knowledge/ingest.py 切块、打标签，写出 chunks.jsonl 和 stats.json；
- 把清洗/切块/标签告警写到 data/logs/build_warnings.md。

两种运行模式：
    python scripts/build_chunks.py --register-only   # 只登记来源，不切块
    python scripts/build_chunks.py                   # 读已登记来源，清洗 + 切块 + 打标签

调用链：
    main() -> register_sources() 或 build()
    build() -> clean_markdown() -> chunks_from_cleaned() -> write_jsonl() -> make_stats()

依据：
- 实践计划书 §4(五)：来源登记 -> 清洗 -> 标题化 -> 切块 -> chunks.jsonl -> index -> stats。
- DM-RAG 论文 §V：小块检索 + 父块补上下文；检索块 128 token，重叠取 0。
- 实践计划书 §4(一)：四类主题；§4(三)：切块与元数据规则。
- data/raw、data/interim、data/processed、data/logs 不进 Git（项目理解 表12）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.ingest import (  # noqa: E402
    Source,
    chunks_from_cleaned,
    classify_h1_intro,
    clean_markdown,
    extract_h1,
    extract_version,
    load_tags,
    make_stats,
    write_json,
    write_jsonl,
)


def load_settings(path: Path) -> dict[str, Any]:
    """读取 config/settings.yaml：路径、切块参数、来源 URL 前缀。"""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(root: Path, value: str) -> Path:
    """把 settings.yaml 里的相对路径转成以项目根目录为基准的绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else root / path


def discover_docs(raw_dir: Path) -> list[Path]:
    """列出原始目录下的 .md 文件，按文件名排序，保证每次构建顺序一致。"""
    if not raw_dir.exists():
        return []
    return sorted(
        [path for path in raw_dir.glob("*.md") if path.is_file()],
        key=lambda path: path.name,
    )


def source_groups(settings: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """返回来源组：案例和手册分开登记，source_id 前缀不同，互不影响编号。"""
    groups: list[dict[str, Any]] = [
        {
            "raw_dir": resolve(root, settings["raw_dir"]),
            "source_type": "case",
            "source_id_prefix": "case",
            "source_url_prefix": settings["source_url_prefix"],
            "copyright_note": "公开技术文章，仅本地学习使用；原文不进入 Git",
        }
    ]
    if settings.get("doc_raw_dir"):
        groups.append(
            {
                "raw_dir": resolve(root, settings["doc_raw_dir"]),
                "source_type": "doc",
                "source_id_prefix": "doc",
                "source_url_prefix": settings.get("doc_source_url_prefix", "local://dm_docs/"),
                "copyright_note": "达梦公开文档，仅本地学习使用；原文不进入 Git",
            }
        )
    return groups


def register_sources(settings: dict[str, Any], root: Path) -> list[Source]:
    """扫描原始文档并登记来源，写 sources.json 和 registration_report.md。

    登记字段：source_id / source_type / source_name / source_file / source_url /
    version / acquired_at / intro_type / copyright_note。
    source_url 用 local:// 前缀，不伪造原始公众号链接；版本找不到就写“未注明”。
    """
    registry_file = resolve(root, settings["registry_file"])
    log_dir = resolve(root, settings["log_dir"])
    acquired_at = date.today().isoformat()

    sources: list[Source] = []
    intro_counts: Counter[str] = Counter()
    report_rows: list[str] = []
    seen_files: set[str] = set()
    for group in source_groups(settings, root):
        docs = discover_docs(group["raw_dir"])
        if not docs and group["source_type"] == "case":
            raise SystemExit(f"没有找到案例 Markdown 原始文档：{group['raw_dir']}")
        url_prefix = group["source_url_prefix"].rstrip("/") + "/"
        for index, path in enumerate(docs, start=1):
            if path.name in seen_files:
                raise SystemExit(f"来源文件名重复，无法区分：{path.name}")
            seen_files.add(path.name)
            raw = path.read_text(encoding="utf-8-sig")
            title, h1_body = extract_h1(raw)
            intro_type = classify_h1_intro(h1_body)
            intro_counts[intro_type] += 1
            source = Source(
                source_id=f"{group['source_id_prefix']}-{index:03d}",
                source_type=group["source_type"],
                source_name=title,
                source_file=path.name,
                source_url=url_prefix + path.name,
                version=extract_version(raw),
                acquired_at=acquired_at,
                intro_type=intro_type,
                copyright_note=group["copyright_note"],
            )
            sources.append(source)
            report_rows.append(
                f"| {source.source_id} | {source.source_type} | {source.intro_type} | {source.version} | {source.source_name} | {source.source_file} |"
            )

    if not sources:
        raise SystemExit("没有登记到任何 Markdown 原始文档")

    registry = {
        "generated_at": acquired_at,
        "source_types": sorted({source.source_type for source in sources}),
        "source_count": len(sources),
        "sources": [
            {
                "source_id": source.source_id,
                "source_type": source.source_type,
                "source_name": source.source_name,
                "source_file": source.source_file,
                "source_url": source.source_url,
                "version": source.version,
                "acquired_at": source.acquired_at,
                "intro_type": source.intro_type,
                "copyright_note": source.copyright_note,
            }
            for source in sources
        ],
    }
    write_json(registry_file, registry)

    log_dir.mkdir(parents=True, exist_ok=True)
    report = "\n".join(
        [
            "# 来源登记报告",
            "",
            f"- 生成日期：{acquired_at}",
            f"- 文档数：{len(sources)}",
            f"- 来源类型：{registry['source_types']}",
            f"- H1 类型：{dict(intro_counts)}",
            "",
            "| source_id | source_type | intro_type | version | source_name | source_file |",
            "| --- | --- | --- | --- | --- | --- |",
            *report_rows,
            "",
        ]
    )
    (log_dir / "registration_report.md").write_text(report, encoding="utf-8")
    print(f"registered sources: {len(sources)}")
    print(f"source_types: {registry['source_types']}")
    print(f"intro_type distribution: {dict(intro_counts)}")
    print(f"registry: {registry_file}")
    return sources


def build(settings: dict[str, Any], root: Path) -> None:
    """按已登记来源逐篇清洗、切块、打标签，并写出全部 1A 产物。

    输入：data/registry/sources.json（必须先跑 --register-only）。
    输出：data/interim/cleaned/*.md、data/processed/chunks.jsonl、
          data/processed/stats.json、data/logs/build_warnings.md。
    """
    interim_dir = resolve(root, settings["interim_dir"])
    processed_dir = resolve(root, settings["processed_dir"])
    registry_file = resolve(root, settings["registry_file"])
    log_dir = resolve(root, settings["log_dir"])
    tags_file = root / "config" / "tags.yaml"

    if not registry_file.exists():
        raise SystemExit(f"缺少来源登记：{registry_file}，先运行 --register-only")
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    sources = [Source(**item) for item in registry["sources"]]
    tags_cfg = load_tags(tags_file)

    # 来源文件只存文件名；这里把案例目录和手册目录合在一起建索引，重名直接报错。
    raw_paths: dict[str, Path] = {}
    for group in source_groups(settings, root):
        for path in discover_docs(group["raw_dir"]):
            if path.name in raw_paths:
                raise SystemExit(f"来源文件名重复，无法区分：{path.name}")
            raw_paths[path.name] = path

    all_records: list[dict[str, Any]] = []
    warnings_by_source: dict[str, list[str]] = {}
    for source in sources:
        raw_path = raw_paths.get(source.source_file)
        if raw_path is None:
            raise SystemExit(f"找不到来源文件：{source.source_file}")
        raw = raw_path.read_text(encoding="utf-8-sig")
        cleaned, clean_warnings = clean_markdown(raw, source.intro_type)

        interim_dir.mkdir(parents=True, exist_ok=True)
        (interim_dir / source.source_file).write_text(cleaned, encoding="utf-8")

        records, chunk_warnings = chunks_from_cleaned(cleaned, source, settings, tags_cfg)
        all_records.extend(records)
        warnings_by_source[source.source_id] = clean_warnings + chunk_warnings

    chunks_file = processed_dir / "chunks.jsonl"
    stats_file = processed_dir / "stats.json"
    write_jsonl(chunks_file, all_records)
    stats = make_stats(
        all_records,
        sources,
        warnings_by_source,
        min_chars=int(settings["chunk_min_chars"]),
        max_chars=int(settings["chunk_max_chars"]),
    )
    write_json(stats_file, stats)

    log_dir.mkdir(parents=True, exist_ok=True)
    warning_lines: list[str] = []
    for source_id, warnings in warnings_by_source.items():
        if warnings:
            warning_lines.append(f"## {source_id}")
            warning_lines.extend(f"- {item}" for item in warnings)
            warning_lines.append("")
    (log_dir / "build_warnings.md").write_text(
        "# 清洗/切块/标签告警\n\n" + "\n".join(warning_lines) + "\n",
        encoding="utf-8",
    )

    print(f"chunks: {stats['total_chunks']}")
    print(f"avg_chunk_chars: {stats['avg_chunk_chars']}")
    print(f"min/max: {stats['min_chunk_chars']}/{stats['max_chunk_chars']}")
    print(f"p90_chunk_chars: {stats['p90_chunk_chars']}")
    print(
        f"under_min/over_max: {stats['under_min_chunks']}/{stats['over_max_chunks']}"
    )
    print(f"parent_count: {stats['parent_count']}")
    print(f"role_coverage: {stats['role_coverage']}")
    print(f"content_type_distribution: {stats['content_type_distribution']}")
    print(f"duplicate_chunks: {stats['duplicate_chunks']}")
    print(f"topic_coverage: {stats['topic_coverage']}")
    print(f"chunks_file: {chunks_file}")
    print(f"stats_file: {stats_file}")


def main() -> None:
    """命令行分发：--register-only 只登记，否则执行完整构建。"""
    parser = argparse.ArgumentParser(description="1A 来源登记与切块")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    settings = load_settings(config_path)

    if args.register_only:
        register_sources(settings, ROOT)
    else:
        build(settings, ROOT)


if __name__ == "__main__":
    main()

"""1A 校验入口：校验 chunks.jsonl，并生成 20 块 + 10 组父块人工抽看清单（只读产物）。

本文件做什么：
- 检查 chunks.jsonl 的必需字段、chunk_id/parent_id 格式与唯一性；
- 检查 root_id/parent_id/chunk_index 是否属于同一来源、同一父块内编号是否连续；
- 检查来源字段与 data/registry/sources.json 是否一致；
- 检查角色和内容类型是否都在 config/tags.yaml 登记过；
- 检查空文本、代码围栏是否成对、表格是否带表头；
- 检查不足 min_chars 的块是否只允许 H1 摘要或已标 short_chunk；
- 检查 stats.json 的总块数、父块数与 chunks.jsonl 是否一致；
- 用固定种子抽 20 块，并抽 10 个有多个子块的父块，生成 spot_check_20.md。

用法：
    python scripts/check_chunks.py

依据：
- 实践计划书 §4(三)：min/max 字、代码块、表格表头规则。
- 实践计划书 §4(五)：统计项一致性。
- 实践计划书 §4(六)：人工抽看；DM-RAG 论文 §V 的父块上下文还原。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.ingest import TOPIC_ORDER, load_tags  # noqa: E402

REQUIRED_FIELDS = {
    "chunk_id",
    "root_id",
    "parent_id",
    "chunk_index",
    "source_type",
    "source_name",
    "source_url",
    "section_path",
    "title",
    "section_role",
    "content_type",
    "tags",
    "version",
    "text",
}
CHUNK_ID_RE = re.compile(r"^(?:case|doc|web)-\d{3}-c\d{3}$")
PARENT_ID_RE = re.compile(r"^(?:case|doc|web)-\d{3}-p\d{3}$")
TABLE_SEPARATOR_CHARS = set("|-:+= ")


def load_json(path: Path) -> Any:
    """读取 JSON 文件（sources.json / stats.json）。"""
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行读取 chunks.jsonl；某行 JSON 坏了就报出行号并停止。"""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} JSON 解析失败：{exc}") from exc
    return records


def is_table_separator(line: str) -> bool:
    """判断是否是 Markdown 表格分隔行（|---|:--:|）或 psql 纯虚线分隔行。"""
    stripped = line.strip()
    if not stripped or "-" not in stripped:
        return False
    if set(stripped) - TABLE_SEPARATOR_CHARS:
        return False
    return len(stripped.replace(" ", "")) >= 3


def check_table_header(text: str) -> bool:
    """每个连续 Markdown 表格块的第一行是表头，第二行必须是分隔行。

    代码围栏里的“|”是程序输出，不是 Markdown 表格，跳过不查。
    """
    lines = text.splitlines()
    index = 0
    in_fence = False
    while index < len(lines):
        if lines[index].strip().startswith("```"):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence or not lines[index].lstrip().startswith("|"):
            index += 1
            continue
        if index + 1 >= len(lines) or not is_table_separator(lines[index + 1]):
            return False
        index += 1
        while index < len(lines) and (
            lines[index].lstrip().startswith("|") or is_table_separator(lines[index])
        ):
            index += 1
    return True


def main() -> None:
    """校验 + 统计 + 生成 20 块抽看清单；有 errors 时以退出码 1 结束。"""
    parser = argparse.ArgumentParser(description="校验切块结果")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    # [STEP 1] 读配置、标签词典和产物路径。
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tags_cfg = load_tags(ROOT / "config" / "tags.yaml")
    max_chars = int(settings["chunk_max_chars"])
    min_chars = int(settings["chunk_min_chars"])

    processed_dir = Path(settings["processed_dir"])
    if not processed_dir.is_absolute():
        processed_dir = ROOT / processed_dir
    registry_file = Path(settings["registry_file"])
    if not registry_file.is_absolute():
        registry_file = ROOT / registry_file
    chunks_file = processed_dir / "chunks.jsonl"
    stats_file = processed_dir / "stats.json"

    # [STEP 2] 读 chunks.jsonl、sources.json、stats.json；来源按 source_id 建索引。
    records = load_jsonl(chunks_file)
    registry = load_json(registry_file)
    sources = {item["source_id"]: item for item in registry["sources"]}
    stats = load_json(stats_file) if stats_file.exists() else {}

    allowed_tags = set(tags_cfg.get("topics", {}).keys())
    allowed_tags.update(tags_cfg.get("operators", []))
    allowed_tags.update(tags_cfg.get("parameters", []))
    allowed_tags.update(tags_cfg.get("sql_features", {}).keys())
    allowed_roles = set(tags_cfg.get("section_roles", {}).keys())
    allowed_roles.update(tags_cfg.get("manual_section_roles", {}).keys())
    allowed_roles.add("未分类")
    allowed_content_types = set(tags_cfg.get("content_types", []))

    # [STEP 3] 逐块校验：字段、ID、来源一致性、标签、围栏、表格、长度。
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    text_counts: Counter[str] = Counter()
    flags_by_chunk: dict[str, list[str]] = {
        item["chunk_id"]: item["flags"] for item in stats.get("flagged_chunks", [])
    }

    for record in records:
        chunk_id = record.get("chunk_id", "<missing>")
        missing = REQUIRED_FIELDS - record.keys()
        extra = record.keys() - REQUIRED_FIELDS
        if missing:
            errors.append(f"{chunk_id}: 缺字段 {sorted(missing)}")
        if extra:
            warnings.append(f"{chunk_id}: 多字段 {sorted(extra)}")

        if not CHUNK_ID_RE.match(str(chunk_id)):
            errors.append(f"{chunk_id}: chunk_id 格式错误")
        if chunk_id in seen_ids:
            errors.append(f"{chunk_id}: chunk_id 重复")
        seen_ids.add(chunk_id)

        root_id = str(record.get("root_id", ""))
        parent_id = str(record.get("parent_id", ""))
        if not root_id:
            errors.append(f"{chunk_id}: root_id 为空")
        if not PARENT_ID_RE.match(parent_id):
            errors.append(f"{chunk_id}: parent_id 格式错误 {parent_id!r}")
        elif not parent_id.startswith(root_id + "-p"):
            errors.append(f"{chunk_id}: parent_id 不属于 root_id")
        chunk_index = record.get("chunk_index")
        if not isinstance(chunk_index, int) or chunk_index < 1:
            errors.append(f"{chunk_id}: chunk_index 不是正整数")

        section_role = record.get("section_role")
        if section_role not in allowed_roles:
            errors.append(f"{chunk_id}: 未登记角色 {section_role!r}")
        content_type = record.get("content_type")
        if content_type not in allowed_content_types:
            errors.append(f"{chunk_id}: 未登记内容类型 {content_type!r}")

        source_id = str(chunk_id).split("-c")[0] if "-c" in str(chunk_id) else ""
        source = sources.get(source_id)
        if source is None:
            errors.append(f"{chunk_id}: 来源 {source_id} 不在 sources.json")
        else:
            if root_id != source_id:
                errors.append(f"{chunk_id}: root_id 与 source_id 不一致")
            if record.get("source_type") != source["source_type"]:
                errors.append(f"{chunk_id}: source_type 与登记不一致")
            if record.get("source_name") != source["source_name"]:
                errors.append(f"{chunk_id}: source_name 与登记不一致")
            if record.get("source_url") != source["source_url"]:
                errors.append(f"{chunk_id}: source_url 与登记不一致")
            if record.get("version") != source["version"]:
                errors.append(f"{chunk_id}: version 与登记不一致")

        text = record.get("text", "")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{chunk_id}: text 为空")
            continue
        text_counts[text] += 1

        if record.get("source_type") not in {"case", "doc", "web"}:
            errors.append(f"{chunk_id}: source_type 非法")
        if not str(record.get("section_path", "")).strip():
            errors.append(f"{chunk_id}: section_path 为空")
        if not str(record.get("title", "")).strip():
            errors.append(f"{chunk_id}: title 为空")
        section_path = str(record.get("section_path", ""))
        if section_path and not section_path.startswith(str(record.get("source_name", ""))):
            warnings.append(f"{chunk_id}: section_path 未以 source_name 开头")
        if section_path.split(" > ")[-1] != record.get("title"):
            warnings.append(f"{chunk_id}: title 不是 section_path 最后一级")
        if not str(record.get("version", "")).strip():
            errors.append(f"{chunk_id}: version 为空")

        tags = record.get("tags")
        if not isinstance(tags, list):
            errors.append(f"{chunk_id}: tags 不是列表")
        else:
            invalid = [tag for tag in tags if tag not in allowed_tags]
            if invalid:
                errors.append(f"{chunk_id}: 未登记标签 {invalid}")

        if text.count("```") % 2 != 0:
            errors.append(f"{chunk_id}: 代码围栏数量为奇数")
        if "|" in text and not check_table_header(text):
            errors.append(f"{chunk_id}: 表格块缺少表头/分隔行")

        if len(text) < min_chars:
            is_h1_abstract = (
                record.get("title") == record.get("source_name")
                and section_path == record.get("source_name")
            )
            is_flagged_short = "short_chunk" in flags_by_chunk.get(chunk_id, [])
            if not is_h1_abstract and not is_flagged_short:
                errors.append(
                    f"{chunk_id}: 非 H1 摘要块不足 {min_chars} 字且未标 short_chunk（{len(text)}）"
                )
            else:
                warnings.append(f"{chunk_id}: 短块保留（{len(text)} 字）")
        if len(text) > max_chars:
            warnings.append(f"{chunk_id}: 超过 {max_chars} 字（{len(text)}）")

        if not any(tag in TOPIC_ORDER for tag in (tags or [])):
            warnings.append(f"{chunk_id}: 未命中四类主题")
        if chunk_id in flags_by_chunk:
            warnings.append(f"{chunk_id}: 质量标记 {flags_by_chunk[chunk_id]}")

    # [STEP 3.5] 父块一致性：同一个 parent_id 只能有一个角色/标题/路径，chunk_index 必须连续。
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_parent.setdefault(record.get("parent_id", ""), []).append(record)
    for parent_id, group in by_parent.items():
        roles = {item.get("section_role") for item in group}
        titles = {item.get("title") for item in group}
        paths = {item.get("section_path") for item in group}
        indices = sorted(item.get("chunk_index", 0) for item in group)
        if len(roles) != 1:
            errors.append(f"{parent_id}: 同一父块有多个角色 {sorted(roles)}")
        if len(titles) != 1:
            errors.append(f"{parent_id}: 同一父块有多个 title")
        if len(paths) != 1:
            errors.append(f"{parent_id}: 同一父块有多个 section_path")
        if indices != list(range(1, len(indices) + 1)):
            errors.append(f"{parent_id}: chunk_index 不连续 {indices}")

    duplicates = [text for text, count in text_counts.items() if count > 1]
    if duplicates:
        warnings.append(f"重复文本 {len(duplicates)} 组")

    # [STEP 4] 全局校验：重复文本、stats.json 总块数、主题覆盖与长度统计。
    recalculated_total = len(records)
    if stats and stats.get("total_chunks") != recalculated_total:
        errors.append(
            f"stats.json total_chunks={stats.get('total_chunks')} 与实际 {recalculated_total} 不一致"
        )
    if stats and stats.get("parent_count") != len(by_parent):
        errors.append(
            f"stats.json parent_count={stats.get('parent_count')} 与实际 {len(by_parent)} 不一致"
        )

    topic_coverage = {
        topic: sum(1 for record in records if topic in (record.get("tags") or []))
        for topic in TOPIC_ORDER
    }
    role_counts = Counter(record.get("section_role") for record in records)
    content_type_counts = Counter(record.get("content_type") for record in records)
    lengths = [len(record.get("text", "")) for record in records]
    source_counts = Counter(record.get("source_name", "") for record in records)

    print(f"chunks: {len(records)}")
    print(f"source_names: {len(source_counts)}")
    print(f"parent_count: {len(by_parent)}")
    print(f"source_type_distribution: {dict(Counter(r.get('source_type') for r in records))}")
    print(f"topic_coverage: {topic_coverage}")
    print(f"role_coverage: {dict(role_counts)}")
    print(f"content_type_distribution: {dict(content_type_counts)}")
    print(f"length avg/min/max: {sum(lengths) / len(lengths):.1f}/{min(lengths)}/{max(lengths)}")
    print(f"under_{min_chars}: {sum(1 for length in lengths if length < min_chars)}")
    print(f"over_{max_chars}: {sum(1 for length in lengths if length > max_chars)}")
    print(f"duplicate_texts: {len(duplicates)}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    for item in errors[:50]:
        print(f"ERROR {item}")
    for item in warnings[:50]:
        print(f"WARN {item}")

    # [STEP 5] 固定种子抽 20 块；保证最长块一定出现在清单里。
    sample_size = min(20, len(records))
    rng = random.Random(20261117)
    sample_indices = rng.sample(range(len(records)), sample_size)
    if len(records) > sample_size:
        longest_index = max(range(len(records)), key=lambda index: len(records[index]["text"]))
        if longest_index not in sample_indices:
            sample_indices[-1] = longest_index
    sample_indices.sort()
    lines = [
        "# 人工抽看：20 个子块 + 10 组父块",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        f"- 语料总块数：{len(records)}",
        f"- 父块总数：{len(by_parent)}",
        f"- 子块抽样：固定种子 20261117 随机抽 {sample_size} 块；若最长块未被抽中，替换其中一块",
        "- 父块抽样：固定种子 20261118 从“至少 2 个子块”的父块中抽 10 组，按 chunk_index 拼回整节",
        "- 代码被拆开时每段各自带围栏，拼回后会看到多个连续代码块，这是预期结果",
        "",
        "## 一、20 个子块",
        "",
        "- 检查项：本块语义可读、代码围栏成对、标题/路径正确、角色/内容类型正确、标签正确、来源正确",
        "",
    ]
    for order, index in enumerate(sample_indices, start=1):
        record = records[index]
        flags = flags_by_chunk.get(record["chunk_id"], [])
        lines.extend(
            [
                f"### {order}. {record['chunk_id']}",
                "",
                f"- root_id：{record['root_id']}",
                f"- parent_id：{record['parent_id']}",
                f"- chunk_index：{record['chunk_index']}",
                f"- source_name：{record['source_name']}",
                f"- source_url：{record['source_url']}",
                f"- section_path：{record['section_path']}",
                f"- title：{record['title']}",
                f"- section_role：{record['section_role']}",
                f"- content_type：{record['content_type']}",
                f"- tags：{', '.join(record['tags']) if record['tags'] else '（空）'}",
                f"- version：{record['version']}",
                f"- length：{len(record['text'])}",
                f"- flags：{', '.join(flags) if flags else '（无）'}",
                "",
                "~~~~text",
                record["text"],
                "~~~~",
                "",
                "- [ ] 本块语义可读（允许是父块中的一段）",
                "- [ ] 代码围栏成对，拆行位置可接受",
                "- [ ] 标题/路径正确",
                "- [ ] 角色/内容类型正确",
                "- [ ] 标签正确",
                "- [ ] 来源正确",
                "",
            ]
        )

    # [STEP 6] 从“至少 2 个子块”的父块里固定抽 10 组，按 chunk_index 拼回整节。
    multi_child_parents = [
        parent_id for parent_id, group in by_parent.items() if len(group) >= 2
    ]
    parent_rng = random.Random(20261118)
    parent_ids = sorted(
        parent_rng.sample(multi_child_parents, min(10, len(multi_child_parents)))
    )
    lines.extend(
        [
            "## 二、10 组父块还原",
            "",
            "- 检查项：子块顺序正确（chunk_index 1..N）、拼起来是一节完整的上下文、角色与标题一致",
            "",
        ]
    )
    for order, parent_id in enumerate(parent_ids, start=1):
        group = sorted(by_parent[parent_id], key=lambda item: item["chunk_index"])
        first = group[0]
        joined = "\n\n".join(item["text"] for item in group)
        lines.extend(
            [
                f"### {order}. {parent_id}",
                "",
                f"- source_name：{first['source_name']}",
                f"- section_path：{first['section_path']}",
                f"- title：{first['title']}",
                f"- section_role：{first['section_role']}",
                f"- 子块数：{len(group)}",
                f"- 拼接后总字数：{len(joined)}",
                f"- chunk_ids：{', '.join(item['chunk_id'] for item in group)}",
                "",
                "~~~~text",
                joined,
                "~~~~",
                "",
                "- [ ] chunk_index 连续且顺序正确",
                "- [ ] 拼起来能读成完整一节",
                "- [ ] 角色与标题一致",
                "- [ ] 没有把不同章节混进同一父块",
                "",
            ]
        )
    sample_file = processed_dir / "spot_check_20.md"
    sample_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"spot_check: {sample_file}")
    print(f"parent_sample: {len(parent_ids)} groups")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

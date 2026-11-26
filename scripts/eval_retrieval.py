# 本文件做什么：1B 20 题基线评测。把检索结果量化成 RoleRecall@3、RoleMRR@3、CaseHit@1、
#               角色覆盖、证据覆盖、缺失角色、上下文字数、全局 Recall@5，并写 JSON + Markdown 报告。
# 被谁使用：人工执行；1B 验收/答辩数据来源。
# 运行：
#   python scripts/eval_retrieval.py
#   python scripts/eval_retrieval.py --limit 3    # 只跑前 3 题，用于冒烟
# 输出：
#   data/eval/eval_metrics.json
#   data/eval/eval_report.md
# 说明：第一版只记录基线，不设质量通过线；工程硬校验（索引块数、哈希、gold 父块存在性）失败才中止。
# 依据：实践计划书 表42；DM-RAG 论文 §V 的 Recall/角色覆盖评测思路。
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.config import load_settings, retrieval_config  # noqa: E402
from src.knowledge.retrieval import (  # noqa: E402
    load_search_index,
    read_jsonl,
    search,
)


def pct(value: float | None) -> str:
    """把 0-1 的比例转成百分比文本；None 显示为“-”。"""
    return "-" if value is None else f"{value * 100:.1f}%"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], ratio: float) -> float | None:
    """最近秩百分位；样本量小，不引入 numpy。"""
    if not values:
        return None
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return ordered[position]


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"评测题为空：{path}")
    return rows


def preflight(
    questions: list[dict[str, Any]],
    index: Any,
    settings: dict[str, Any],
    tags_cfg: dict[str, Any],
) -> dict[str, Any]:
    """评测前硬校验：题量/分布/gold 父块存在且角色、root、section_path 都对得上。"""
    errors: list[str] = []
    warnings: list[str] = []
    ids = [q.get("question_id") for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("question_id 有重复")
    if len(questions) != 20:
        errors.append(f"题目数应为 20，实际 {len(questions)}")

    topics = Counter(q.get("topic") for q in questions)
    expected_topics = set(tags_cfg.get("topics", {}).keys())
    for topic in expected_topics:
        if topics.get(topic, 0) != 5:
            errors.append(f"主题 {topic} 应有 5 题，实际 {topics.get(topic, 0)}")
    extra_topics = set(topics) - expected_topics
    if extra_topics:
        errors.append(f"出现未定义主题：{sorted(extra_topics)}")

    root_counts = Counter(q.get("gold_root_id") for q in questions)
    if len(root_counts) < 10:
        errors.append(f"gold root 应至少 10 个，实际 {len(root_counts)}")
    for root_id, count in sorted(root_counts.items()):
        if count > 3:
            warnings.append(f"gold root {root_id} 有 {count} 题，超过原计划的 max 3；原因见 README_1B")

    role_counts = Counter()
    for q in questions:
        role_counts.update(q.get("required_roles", []))
    cfg = retrieval_config(settings)
    for role in cfg.get("report_roles", []):
        if role_counts.get(role, 0) < 5:
            errors.append(f"角色 {role} 至少应出现在 5 题，实际 {role_counts.get(role, 0)}")
    multi_role = sum(1 for q in questions if len(q.get("required_roles", [])) >= 2)
    three_plus = sum(1 for q in questions if len(q.get("required_roles", [])) >= 3)
    evidence_questions = sum(1 for q in questions if q.get("required_content_types"))
    if multi_role < 4:
        errors.append(f"多角色题应至少 4 道，实际 {multi_role}")
    if three_plus < 2:
        errors.append(f"三角色题应至少 2 道，实际 {three_plus}")
    if evidence_questions < 4:
        errors.append(f"带证据类型题应至少 4 道，实际 {evidence_questions}")

    allowed_evidence = set(cfg.get("evidence_content_types", []))
    for q in questions:
        qid = q.get("question_id", "?")
        roles = list(q.get("required_roles", []))
        gold_parents = q.get("gold_parents", {})
        if set(roles) != set(gold_parents):
            errors.append(f"{qid}: required_roles 与 gold_parents 的键不一致")
        gold_root = q.get("gold_root_id")
        if gold_root not in index.by_root:
            errors.append(f"{qid}: gold_root_id 不存在：{gold_root}")
            continue
        section_paths = q.get("gold_section_paths", {})
        for role, parent_ids in gold_parents.items():
            for parent_id in parent_ids:
                records = index.by_parent.get(parent_id)
                if not records:
                    errors.append(f"{qid}: gold 父块不存在：{parent_id}")
                    continue
                if records[0]["root_id"] != gold_root:
                    errors.append(f"{qid}: 父块 {parent_id} 不属于 {gold_root}")
                if records[0]["section_role"] != role:
                    errors.append(
                        f"{qid}: 父块 {parent_id} 角色是 {records[0]['section_role']}，"
                        f"gold 写成 {role}"
                    )
                expected_path = section_paths.get(parent_id)
                if expected_path != records[0]["section_path"]:
                    errors.append(
                        f"{qid}: 父块 {parent_id} section_path 不一致："
                        f"gold={expected_path!r} 实际={records[0]['section_path']!r}"
                    )
        invalid = set(q.get("required_content_types", [])) - allowed_evidence
        if invalid:
            errors.append(f"{qid}: required_content_types 不在证据类型里：{sorted(invalid)}")
        # 题目要求“执行计划/统计输出”不代表 gold 案例里真有；没有时证据覆盖 0 是
        # “资料不足”而不是检索错。这里只提示，不把它改成错误，避免用改题的方式抬指标。
        root_content_types = {
            record["content_type"] for record in index.by_root.get(gold_root, [])
        }
        missing_evidence = set(q.get("required_content_types", [])) - root_content_types
        if missing_evidence:
            warnings.append(
                f"{qid}: gold root {gold_root} 不含证据类型 {sorted(missing_evidence)}，"
                f"证据覆盖按资料不足解释"
            )

    if errors:
        raise SystemExit("评测集校验失败：\n- " + "\n- ".join(errors))

    return {
        "question_count": len(questions),
        "topics": dict(topics),
        "root_count": len(root_counts),
        "max_questions_per_root": max(root_counts.values()),
        "role_question_counts": dict(role_counts),
        "multi_role_questions": multi_role,
        "three_plus_role_questions": three_plus,
        "evidence_questions": evidence_questions,
        "warnings": warnings,
    }


def role_metrics(result: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    """角色级指标：只用 top3 候选，不拿最终选中块倒推召回。"""
    gold_parents = question.get("gold_parents", {})
    roles = list(question.get("required_roles", []))
    role_hits = 0
    role_mrr = 0.0
    scored_roles = 0
    per_role: dict[str, Any] = {}
    for role in roles:
        gold = set(gold_parents.get(role, []))
        if not gold:
            continue
        scored_roles += 1
        top3 = result.get("candidate_top3", {}).get(role, [])
        hit_rank = None
        for rank, candidate in enumerate(top3, start=1):
            if candidate.get("parent_id") in gold:
                hit_rank = rank
                break
        hit = hit_rank is not None
        role_hits += int(hit)
        role_mrr += (1.0 / hit_rank) if hit_rank else 0.0
        per_role[role] = {
            "hit": hit,
            "hit_rank": hit_rank,
            "top3_parents": [candidate.get("parent_id") for candidate in top3],
        }
    return {
        "role_recall_at_3": role_hits / scored_roles if scored_roles else None,
        "role_mrr_at_3": role_mrr / scored_roles if scored_roles else None,
        "role_scored_count": scored_roles,
        "role_hit_count": role_hits,
        "role_mrr_sum": role_mrr,
        "per_role": per_role,
    }


def evaluate_one(
    index: Any,
    question: dict[str, Any],
    global_top_k: int = 5,
) -> dict[str, Any]:
    """跑单题：gold 角色、推断角色、全局基线各一次，算全部指标。"""
    gold_result = search(index, question["question"], mode="role", required_roles=question["required_roles"])
    inferred_result = search(index, question["question"], mode="role", required_roles=None)
    global_result = search(index, question["question"], mode="global", top_k=global_top_k)

    gold_root = question["gold_root_id"]
    gold_parent_union = {
        parent_id for parents in question["gold_parents"].values() for parent_id in parents
    }
    role_info = role_metrics(gold_result, question)
    selected = gold_result.get("selected", {})
    required_roles = list(question.get("required_roles", []))
    role_coverage = (
        sum(1 for role in required_roles if selected.get(role)) / len(required_roles)
        if required_roles
        else None
    )
    role_gold_hits = 0
    for role in required_roles:
        gold = set(question["gold_parents"].get(role, []))
        if any(item["parent_id"] in gold for item in selected.get(role, [])):
            role_gold_hits += 1
    role_gold_coverage = role_gold_hits / len(required_roles) if required_roles else None
    missing_roles = gold_result.get("missing_roles", [])
    selected_count = sum(len(items) for items in selected.values())

    required_types = list(question.get("required_content_types", []))
    evidence_types = {item["content_type"] for item in gold_result.get("evidence", [])}
    evidence_coverage = (
        sum(1 for content_type in required_types if content_type in evidence_types) / len(required_types)
        if required_types
        else None
    )

    global_top5_parents = [item["parent_id"] for item in global_result.get("results", [])]
    global_recall = any(parent_id in gold_parent_union for parent_id in global_top5_parents)

    inferred_roles = set(inferred_result.get("inferred_roles", []))
    gold_roles = set(required_roles)
    union_roles = inferred_roles | gold_roles
    role_infer_exact = inferred_roles == gold_roles
    role_infer_coverage = gold_roles <= inferred_roles
    role_infer_jaccard = len(inferred_roles & gold_roles) / len(union_roles) if union_roles else 1.0

    selected_parents_by_role = {
        role: [item["parent_id"] for item in selected.get(role, [])]
        for role in required_roles
    }

    return {
        "question_id": question["question_id"],
        "topic": question["topic"],
        "question": question["question"],
        "required_roles": required_roles,
        "inferred_roles": inferred_result.get("inferred_roles", []),
        "gold_root_id": gold_root,
        "selected_root_id": gold_result.get("root_id"),
        "selected_root_id_inferred": inferred_result.get("root_id"),
        "case_hit_at_1": gold_result.get("root_id") == gold_root,
        "case_hit_at_1_inferred": inferred_result.get("root_id") == gold_root,
        "role_recall_at_3": role_info["role_recall_at_3"],
        "role_mrr_at_3": role_info["role_mrr_at_3"],
        "role_hit_count": role_info["role_hit_count"],
        "role_scored_count": role_info["role_scored_count"],
        "role_mrr_sum": role_info["role_mrr_sum"],
        "role_coverage": role_coverage,
        "role_gold_coverage": role_gold_coverage,
        "missing_roles": missing_roles,
        "missing_role_count": len(missing_roles),
        "selected_chunk_count": selected_count,
        "evidence_coverage": evidence_coverage,
        "required_content_types": required_types,
        "evidence_types_present": sorted(evidence_types),
        "context_chars": gold_result.get("context_chars", 0),
        "global_recall_at_5": global_recall,
        "global_top5_parents": global_top5_parents,
        "role_infer_exact": role_infer_exact,
        "role_infer_coverage": role_infer_coverage,
        "role_infer_jaccard": role_infer_jaccard,
        "per_role": role_info["per_role"],
        "selected_parents_by_role": selected_parents_by_role,
        "latency_ms": gold_result.get("elapsed_ms"),
        "latency_ms_inferred": inferred_result.get("elapsed_ms"),
        "gold_parent_union": sorted(gold_parent_union),
    }


def aggregate(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合 20 题；同时给 macro（每题等权）和 micro（每个角色等权）两种口径。"""
    n = len(per_question)
    total_role_hits = sum(item["role_hit_count"] for item in per_question)
    total_role_scored = sum(item["role_scored_count"] for item in per_question)
    total_role_mrr = sum(item["role_mrr_sum"] for item in per_question)
    evidence_values = [
        item["evidence_coverage"]
        for item in per_question
        if item["evidence_coverage"] is not None
    ]
    context_values = [float(item["context_chars"]) for item in per_question]
    latency_values = [float(item["latency_ms"]) for item in per_question if item["latency_ms"] is not None]
    selected_values = [float(item["selected_chunk_count"]) for item in per_question]
    missing_values = [float(item["missing_role_count"]) for item in per_question]
    return {
        "question_count": n,
        "case_hit_at_1": mean([float(item["case_hit_at_1"]) for item in per_question]),
        "case_hit_at_1_inferred": mean([float(item["case_hit_at_1_inferred"]) for item in per_question]),
        "role_recall_at_3_macro": mean([float(item["role_recall_at_3"]) for item in per_question]),
        "role_recall_at_3_micro": total_role_hits / total_role_scored if total_role_scored else None,
        "role_mrr_at_3_macro": mean([float(item["role_mrr_at_3"]) for item in per_question]),
        "role_mrr_at_3_micro": total_role_mrr / total_role_scored if total_role_scored else None,
        "role_instance_count": total_role_scored,
        "role_hit_count": total_role_hits,
        "role_coverage": mean([float(item["role_coverage"]) for item in per_question]),
        "role_gold_coverage": mean([float(item["role_gold_coverage"]) for item in per_question]),
        "evidence_coverage": mean(evidence_values),
        "evidence_question_count": len(evidence_values),
        "missing_role_count_mean": mean(missing_values),
        "missing_role_count_max": int(max(missing_values)) if missing_values else 0,
        "context_chars_mean": mean(context_values),
        "context_chars_p50": percentile(context_values, 0.50),
        "context_chars_max": int(max(context_values)) if context_values else 0,
        "selected_chunk_count_mean": mean(selected_values),
        "global_recall_at_5": mean([float(item["global_recall_at_5"]) for item in per_question]),
        "role_infer_exact": mean([float(item["role_infer_exact"]) for item in per_question]),
        "role_infer_coverage": mean([float(item["role_infer_coverage"]) for item in per_question]),
        "role_infer_jaccard": mean([float(item["role_infer_jaccard"]) for item in per_question]),
        "latency_ms_mean": mean(latency_values),
        "latency_ms_p50": percentile(latency_values, 0.50),
        "latency_ms_p90": percentile(latency_values, 0.90),
    }


def topic_breakdown(per_question: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按主题拆开，方便看执行计划/索引/统计信息/SQL写法哪一类最弱。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_question:
        grouped[item["topic"]].append(item)
    result: dict[str, dict[str, Any]] = {}
    for topic, items in sorted(grouped.items()):
        result[topic] = {
            "question_count": len(items),
            "case_hit_at_1": mean([float(item["case_hit_at_1"]) for item in items]),
            "role_recall_at_3": mean([float(item["role_recall_at_3"]) for item in items]),
            "role_mrr_at_3": mean([float(item["role_mrr_at_3"]) for item in items]),
            "role_gold_coverage": mean([float(item["role_gold_coverage"]) for item in items]),
            "evidence_coverage": mean(
                [
                    float(item["evidence_coverage"])
                    for item in items
                    if item["evidence_coverage"] is not None
                ]
            ),
            "context_chars_mean": mean([float(item["context_chars"]) for item in items]),
            "global_recall_at_5": mean([float(item["global_recall_at_5"]) for item in items]),
        }
    return result


def determinism_check(index: Any, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """同一题、同一角色跑两次，比对 root 和选中块，确认检索可复现。"""
    if not questions:
        return {"checked": False}
    question = questions[0]
    first = search(index, question["question"], mode="role", required_roles=question["required_roles"])
    second = search(index, question["question"], mode="role", required_roles=question["required_roles"])
    first_ids = {
        role: [item["chunk_id"] for item in items] for role, items in first.get("selected", {}).items()
    }
    second_ids = {
        role: [item["chunk_id"] for item in items] for role, items in second.get("selected", {}).items()
    }
    return {
        "checked": True,
        "question_id": question["question_id"],
        "same_root": first.get("root_id") == second.get("root_id"),
        "same_selected_chunks": first_ids == second_ids,
        "same_missing_roles": first.get("missing_roles") == second.get("missing_roles"),
    }


def write_report(
    path: Path,
    metrics: dict[str, Any],
    per_question: list[dict[str, Any]],
    topics: dict[str, dict[str, Any]],
    preflight_info: dict[str, Any],
    index: Any,
    determinism: dict[str, Any],
) -> None:
    """生成人可读的 Markdown 报告；不写通过/失败结论，只给基线数据。"""
    m = metrics
    lines: list[str] = [
        "# 1B 检索基线评测报告",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 题数：{m['question_count']}",
        f"- 模型：`{index.meta.get('embedding_model')}`，维度 {index.meta.get('embedding_dim')}",
        f"- 向量集合：`{index.meta.get('chroma_collection')}`，{index.meta.get('chunk_count')} 个子块",
        f"- 父块：{index.meta.get('parent_count')}；来源：{index.meta.get('root_count')}",
        f"- 索引构建耗时：{index.meta.get('elapsed_sec')} 秒",
        "",
        "> 本报告是第一版基线，**不设质量通过线**。只做工程硬校验；质量目标等这版数据出来后再讨论。",
        "",
        "## 1. 总体指标",
        "",
        "| 指标 | 数值 | 口径 |",
        "| --- | --- | --- |",
        f"| CaseHit@1 | {pct(m['case_hit_at_1'])} | 选中案例的 root_id 与 gold 一致 |",
        f"| CaseHit@1（用推断角色） | {pct(m['case_hit_at_1_inferred'])} | 不喂 gold 角色，走问句规则 |",
        f"| RoleRecall@3（macro） | {pct(m['role_recall_at_3_macro'])} | 每题等权，角色 top3 是否命中 gold 父块 |",
        f"| RoleRecall@3（micro） | {pct(m['role_recall_at_3_micro'])} | 每个角色等权（{m['role_hit_count']}/{m['role_instance_count']}） |",
        f"| RoleMRR@3（macro） | {pct(m['role_mrr_at_3_macro'])} | 首个 gold 父块在 top3 的名次倒数 |",
        f"| RoleMRR@3（micro） | {pct(m['role_mrr_at_3_micro'])} | 每个角色等权 |",
        f"| RoleCoverage | {pct(m['role_coverage'])} | 必需角色里有选中块的比例（不看是否 gold） |",
        f"| RoleGoldCoverage | {pct(m['role_gold_coverage'])} | 必需角色里选中 gold 父块的比例 |",
        f"| EvidenceCoverage | {pct(m['evidence_coverage'])} | {m['evidence_question_count']} 道带证据类型题的覆盖率 |",
        f"| MissingRoleCount（均值/最大） | {m['missing_role_count_mean']:.2f} / {m['missing_role_count_max']} | 最终仍缺的角色数 |",
        f"| ContextChars（均值/P50/最大） | {m['context_chars_mean']:.1f} / {m['context_chars_p50']:.1f} / {m['context_chars_max']} | 角色上下文 + 证据去重后的字符数 |",
        f"| 选中子块数（均值） | {m['selected_chunk_count_mean']:.2f} | 每个角色的名额内选中块 |",
        f"| Global Recall@5 | {pct(m['global_recall_at_5'])} | 纯子块全局 top5 基线，不做角色/父块 |",
        f"| 角色推断完全一致 | {pct(m['role_infer_exact'])} | 推断角色集合 == gold 角色集合 |",
        f"| 角色推断覆盖 gold | {pct(m['role_infer_coverage'])} | gold 角色是推断角色的子集 |",
        f"| 角色推断 Jaccard | {pct(m['role_infer_jaccard'])} | 交集/并集 |",
        f"| 单题耗时（均值/P50/P90） | {m['latency_ms_mean']:.1f} / {m['latency_ms_p50']:.1f} / {m['latency_ms_p90']:.1f} ms | 单题一次角色检索 |",
        "",
        "## 2. 主题拆分",
        "",
        "| 主题 | 题数 | CaseHit@1 | RoleRecall@3 | RoleMRR@3 | RoleGoldCoverage | EvidenceCoverage | ContextChars 均值 | Global Recall@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for topic, item in topics.items():
        lines.append(
            f"| {topic} | {item['question_count']} | {pct(item['case_hit_at_1'])} | "
            f"{pct(item['role_recall_at_3'])} | {pct(item['role_mrr_at_3'])} | "
            f"{pct(item['role_gold_coverage'])} | {pct(item['evidence_coverage'])} | "
            f"{item['context_chars_mean']:.1f} | {pct(item['global_recall_at_5'])} |"
        )

    lines.extend(
        [
            "",
            "## 3. 逐题结果",
            "",
            "| ID | 主题 | 期望角色 | 推断角色 | gold root | 选中 root | CaseHit | RoleRecall@3 | MRR@3 | RoleGoldCoverage | 证据覆盖 | 缺角色 | ContextChars | GlobalRecall@5 |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in per_question:
        lines.append(
            f"| {item['question_id']} | {item['topic']} | {'/'.join(item['required_roles'])} | "
            f"{'/'.join(item['inferred_roles']) or '-'} | {item['gold_root_id']} | "
            f"{item['selected_root_id'] or '-'} | {'✅' if item['case_hit_at_1'] else '❌'} | "
            f"{pct(item['role_recall_at_3'])} | {pct(item['role_mrr_at_3'])} | "
            f"{pct(item['role_gold_coverage'])} | {pct(item['evidence_coverage'])} | "
            f"{item['missing_role_count']} | {item['context_chars']} | "
            f"{'✅' if item['global_recall_at_5'] else '❌'} |"
        )

    lines.extend(["", "## 4. 失败明细", ""])
    case_misses = [item for item in per_question if not item["case_hit_at_1"]]
    lines.append("### 4.1 案例未命中")
    if case_misses:
        lines.append("| ID | gold root | 选中 root | 推断角色 |")
        lines.append("| --- | --- | --- | --- |")
        for item in case_misses:
            lines.append(
                f"| {item['question_id']} | {item['gold_root_id']} | "
                f"{item['selected_root_id'] or '-'} | {'/'.join(item['inferred_roles']) or '-'} |"
            )
    else:
        lines.append("无。")

    role_misses = []
    for item in per_question:
        for role, info in item["per_role"].items():
            if not info["hit"]:
                role_misses.append((item["question_id"], role, info["top3_parents"]))
    lines.append("")
    lines.append("### 4.2 RoleRecall@3 未命中")
    if role_misses:
        lines.append("| ID | 角色 | top3 父块 |")
        lines.append("| --- | --- | --- |")
        for qid, role, parents in role_misses:
            lines.append(f"| {qid} | {role} | {', '.join(parents) or '-'} |")
    else:
        lines.append("无。")

    missing = [
        (item["question_id"], "/".join(item["missing_roles"]))
        for item in per_question
        if item["missing_roles"]
    ]
    lines.append("")
    lines.append("### 4.3 最终缺失角色")
    if missing:
        lines.append("| ID | 缺失角色 |")
        lines.append("| --- | --- |")
        for qid, roles in missing:
            lines.append(f"| {qid} | {roles} |")
    else:
        lines.append("无。")

    evidence_misses = [
        (item["question_id"], "/".join(item["required_content_types"]), "/".join(item["evidence_types_present"]) or "-")
        for item in per_question
        if item["evidence_coverage"] is not None and item["evidence_coverage"] < 1.0
    ]
    lines.append("")
    lines.append("### 4.4 证据类型未覆盖")
    if evidence_misses:
        lines.append("| ID | 要求 | 实际 |")
        lines.append("| --- | --- | --- |")
        for qid, required, present in evidence_misses:
            lines.append(f"| {qid} | {required} | {present} |")
    else:
        lines.append("无。")

    lines.extend(
        [
            "",
            "## 5. 工程硬校验",
            "",
            "| 检查 | 结果 |",
            "| --- | --- |",
            f"| chunks 数 == Chroma 向量数 | {len(index.chunks)} / {index.collection.count()} |",
            f"| chunks 数 == BM25 文档数 | {len(index.chunks)} / {len(index.bm25_ids)} |",
            f"| 索引块数/父块数/来源数 | {index.meta.get('chunk_count')} / {index.meta.get('parent_count')} / {index.meta.get('root_count')} |",
            f"| chunks_sha256 一致 | 已在加载索引时校验 |",
            f"| tags_sha256 一致 | 已在加载索引时校验 |",
            f"| 同题两次检索确定性 | {determinism} |",
            "",
            "## 6. 评测集结构",
            "",
            f"- 主题分布：{preflight_info['topics']}",
            f"- gold root 数：{preflight_info['root_count']}",
            f"- 每个 root 最多题数：{preflight_info['max_questions_per_root']}",
            f"- 多角色题：{preflight_info['multi_role_questions']}；三角色及以上：{preflight_info['three_plus_role_questions']}；带证据类型：{preflight_info['evidence_questions']}",
            "",
        ]
    )
    if preflight_info["warnings"]:
        lines.append("评测集提示：")
        for warning in preflight_info["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    lines.extend(
        [
            "## 7. 怎么读这版数据",
            "",
            "- `RoleRecall@3` 看“角色候选里有没有 gold 父块”，不依赖最终选案例；它是检索层指标。",
            "- `CaseHit@1` 看“最终选中的案例对不对”，会受角色覆盖和候选分数影响。",
            "- `RoleGoldCoverage` 看“最终选中的块是不是 gold 父块”，比 RoleRecall 更严格。",
            "- `Global Recall@5` 是纯子块基线；它高于 RoleRecall@3 不代表角色方案差，二者口径不同。",
            "- 质量目标（例如 RoleRecall@3 要到多少）等这版数据出来后再定，不在本报告里预设。",
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="1B 20 题基线评测")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--output-dir", default="data/eval")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题，用于冒烟")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = ROOT / questions_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(config_path)
    cfg = retrieval_config(settings)
    with (ROOT / "config" / "tags.yaml").open("r", encoding="utf-8") as handle:
        tags_cfg = yaml.safe_load(handle)

    questions = load_questions(questions_path)
    print("loading index...")
    index = load_search_index(ROOT, settings)
    preflight_info = preflight(questions, index, settings, tags_cfg)
    if preflight_info["warnings"]:
        print("preflight warnings:")
        for warning in preflight_info["warnings"]:
            print(f"  - {warning}")

    run_questions = questions[: args.limit] if args.limit else questions
    per_question: list[dict[str, Any]] = []
    for position, question in enumerate(run_questions, start=1):
        item = evaluate_one(index, question, global_top_k=int(cfg["global_top_k"]))
        per_question.append(item)
        print(
            f"[{position}/{len(run_questions)}] {item['question_id']} "
            f"case_hit={item['case_hit_at_1']} root={item['selected_root_id']} "
            f"roleR@3={pct(item['role_recall_at_3'])} missing={item['missing_role_count']}"
        )

    metrics = aggregate(per_question)
    topics = topic_breakdown(per_question)
    determinism = determinism_check(index, run_questions)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_file": str(questions_path),
        "index_meta": index.meta,
        "preflight": preflight_info,
        "metrics": metrics,
        "topics": topics,
        "determinism": determinism,
        "per_question": per_question,
    }
    metrics_path = output_dir / "eval_metrics.json"
    report_path = output_dir / "eval_report.md"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(report_path, metrics, per_question, topics, preflight_info, index, determinism)
    print(f"metrics: {metrics_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

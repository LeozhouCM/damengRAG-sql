# 本文件做什么：1B 检索核心。把“角色选子块、父块只补上下文”落成可测试的纯函数和检索流程。
# 被谁使用：scripts/build_index.py（切词、哈希）、scripts/search.py、scripts/eval_retrieval.py、
#            tests/test_retrieval.py（只测纯函数，不加载模型/向量库）。
# 关键边界：
#   1. 子块是唯一检索单位；父块不建索引、不参与排序。
#   2. 角色只负责“选哪些子块”；父块只在选完之后按 parent_id 补上下文。
#   3. 证据不是角色，是 content_type=执行计划/统计输出，单独取。
#   4. 缺角色只补一次；补不到就记录 missing_roles，不硬凑。
# 依据：实践计划书 表41/表42；DM-RAG 论文 §V（小块 + 父块展开、BM25 + 向量 + RRF）；
#       之前与用户确认的角色/父块分工；bge-base-zh-v1.5 模型卡的查询指令。
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jieba
import yaml
from rank_bm25 import BM25Okapi

from src.knowledge.config import chroma_relative_path, resolve_path, retrieval_config

# jieba 首次加载会打印词典日志；这里只保留警告，避免污染检索输出。
jieba.setLogLevel(logging.WARNING)

# 支持角色在报告四类之外的补充顺序。参考/总结不进强制提纲，但检索仍可选中。
SUPPORT_ROLE_ORDER = ["概念", "步骤", "总结", "参考", "未分类"]

# 查询词 -> 需要哪些角色。顺序只用于截断，最终输出顺序按 report_roles + SUPPORT_ROLE_ORDER。
# 依据：config/tags.yaml 的 section_roles 是“标题词表”；这里补的是用户问句里的动作词，
#       两者不混用，避免把“优化”这种标题词硬套到所有问句上。
ROLE_QUERY_PATTERNS: dict[str, str] = {
    "验证": r"验证|确认|对比|效果|执行结果|优化后|是否改善|是否提升",
    "原因": r"为什么|为何|原因|根因|怎么会|机制|原理",
    "问题": r"问题|现象|不走索引|没走索引|失效|性能差|哪里不对|瓶颈|耗时",
    "办法": (
        r"优化(?!器)|解决|改写|调整|建议|方法|怎么办|如何处理|如何优化|怎么优化|"
        r"怎么选|怎么改|如何改"
    ),
    "步骤": (
        r"具体操作步骤|操作步骤|步骤|怎么收集|如何收集|怎么查看|如何查看|"
        r"怎么配置|如何配置|怎么监控|如何监控|统计信息.*(收集|查看|监控)|(收集|查看|监控).*统计信息"
    ),
    "概念": r"是什么|有哪些|定义|区别|种类|作用|什么情况|什么场景|什么时候|介绍|概念",
    "总结": r"总结|结论|小结",
}

@dataclass
class SearchIndex:
    """检索时需要的全部只读资源；由 load_search_index() 一次性装载。"""

    settings: dict[str, Any]
    chunks: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    by_parent: dict[str, list[dict[str, Any]]]
    by_root: dict[str, list[dict[str, Any]]]
    bm25: BM25Okapi
    bm25_ids: list[str]
    terms: list[str]
    model: Any
    collection: Any
    meta: dict[str, Any]
    index_dir: Path


# ---------------------------------------------------------------------------
# 基础 IO / 哈希 / 切词
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """逐行读 JSONL；空行跳过，坏行直接报错并带行号。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """写 JSONL；固定 ensure_ascii=False，保证中文可直接复核。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    """按字节计算文件哈希；用于确认索引与 chunks/tags 是同一版。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_terms(tags_cfg: dict[str, Any]) -> list[str]:
    """收集 BM25 需要额外保留的领域词：主题、操作符、参数、SQL特征。

    这些词来自 config/tags.yaml，是语料中实际出现过的词；jieba 可能把它们切碎，
    所以命中时额外追加整词。排序固定，保证索引和查询端用同一份词表。
    """
    terms: set[str] = set()
    for values in tags_cfg.get("topics", {}).values():
        terms.update(str(value) for value in values)
    terms.update(str(value) for value in tags_cfg.get("operators", []))
    terms.update(str(value) for value in tags_cfg.get("parameters", []))
    terms.update(str(value) for value in tags_cfg.get("sql_features", {}).keys())
    cleaned = {term.strip().lower() for term in terms if term and term.strip()}
    return sorted(cleaned, key=lambda item: (-len(item), item))


def tokenize_text(text: str, terms: Sequence[str]) -> list[str]:
    """jieba 精确模式 + 领域词整词追加。

    索引端和查询端必须调用同一个函数；否则 BM25 的 token 对不上。
    """
    if not text:
        return []
    lowered = text.lower()
    tokens: list[str] = []
    for token in jieba.lcut(lowered, cut_all=False):
        token = token.strip()
        if token and any(ch.isalnum() or "一" <= ch <= "鿿" for ch in token):
            tokens.append(token)
    for term in terms:
        term_lower = term.strip().lower()
        if term_lower and term_lower in lowered:
            tokens.append(term_lower)
    return tokens


def infer_required_roles(
    query: str,
    report_roles: Sequence[str],
    max_roles: int = 4,
) -> list[str]:
    """从问句里推断需要哪些角色。

    只做显式词匹配，不做语义猜测；推断不出角色时返回空列表，由调用方退回全局检索。
    """
    if not query or not query.strip():
        return []
    matched = {
        role for role, pattern in ROLE_QUERY_PATTERNS.items() if re.search(pattern, query)
    }
    # “原因是什么/问题是什么/步骤是什么”里的“是什么”是在问角色内容，不是在问概念；
    # 这里把误命中的“概念”去掉，避免一条问句挂上第五个角色。
    if "概念" in matched and re.search(r"(原因|问题|步骤|办法|方案|结果|效果)是什么", query):
        matched.remove("概念")
    if not matched:
        return []

    # 输出顺序：报告四类在前，补充角色在后；这样 CLI 和评测看到的顺序稳定。
    ordered = [role for role in report_roles if role in matched]
    ordered.extend(role for role in SUPPORT_ROLE_ORDER if role in matched)
    # 极端长问句可能同时命中很多角色；按报告顺序截断，避免一次挂太多角色。
    return ordered[:max_roles]


# ---------------------------------------------------------------------------
# 召回与融合
# ---------------------------------------------------------------------------


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：只使用名次，不使用原始分数，避免向量/BM25 量纲不同。

    返回 [(chunk_id, rrf_score)]，按分数降序；同分用“最早出现的名次”再按 chunk_id 排序，
    保证同一输入每次输出完全一致。
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, 10**9), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))


def dense_ranking(
    index: SearchIndex,
    query_vector: Sequence[float],
    depth: int,
    where: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Chroma 向量召回；返回按余弦相似度降序的 chunk_id 和分数。"""
    if depth <= 0 or not index.chunks:
        return [], {}
    n_results = min(depth, limit if limit is not None else len(index.chunks))
    if n_results <= 0:
        return [], {}
    kwargs: dict[str, Any] = {
        "query_embeddings": [list(query_vector)],
        "n_results": n_results,
        "include": ["distances"],
    }
    if where:
        kwargs["where"] = where
    result = index.collection.query(**kwargs)
    ids = (result.get("ids") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []
    pairs: list[tuple[str, float]] = []
    for chunk_id, distance in zip(ids, distances):
        if chunk_id not in index.by_id:
            continue
        pairs.append((chunk_id, 1.0 - float(distance)))
    # Chroma 已按距离升序返回；这里再排一次，避免版本差异导致顺序不稳定。
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return [chunk_id for chunk_id, _ in pairs], dict(pairs)


def bm25_ranking(
    index: SearchIndex,
    query_tokens: Sequence[str],
    depth: int,
    root_id: str | None = None,
) -> tuple[list[str], dict[str, float]]:
    """BM25 召回；只保留正分文档，避免“全 0 也参与 RRF”污染名次。"""
    if depth <= 0 or not query_tokens:
        return [], {}
    scores = index.bm25.get_scores(list(query_tokens))
    pairs: list[tuple[str, float]] = []
    for chunk_id, score in zip(index.bm25_ids, scores):
        if score <= 0:
            continue
        if root_id is not None and index.by_id[chunk_id]["root_id"] != root_id:
            continue
        pairs.append((chunk_id, float(score)))
    pairs.sort(key=lambda item: (-item[1], item[0]))
    pairs = pairs[:depth]
    return [chunk_id for chunk_id, _ in pairs], dict(pairs)


def child_recall(
    index: SearchIndex,
    query_text: str,
    query_vector: Sequence[float],
    depth: int,
) -> list[dict[str, Any]]:
    """子块召回：向量 topN + BM25 topN -> RRF。返回融合后的子块名次。"""
    dense_ids, dense_scores = dense_ranking(index, query_vector, depth)
    query_tokens = tokenize_text(query_text, index.terms)
    bm25_ids, bm25_scores = bm25_ranking(index, query_tokens, depth)
    fused = rrf_fuse([dense_ids, bm25_ids], k=int(index.settings["retrieval"]["rrf_k"]))
    dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
    bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
    return [
        {
            "chunk_id": chunk_id,
            "score": score,
            "dense_rank": dense_rank.get(chunk_id),
            "bm25_rank": bm25_rank.get(chunk_id),
            "dense_score": dense_scores.get(chunk_id),
            "bm25_score": bm25_scores.get(chunk_id),
        }
        for chunk_id, score in fused
    ]


def root_local_recall(
    index: SearchIndex,
    query_text: str,
    query_vector: Sequence[float],
    root_id: str,
    depth: int,
) -> list[dict[str, Any]]:
    """在已选案例内部再召回一次，只用于补缺失角色；不参与选案例。"""
    root_chunks = index.by_root.get(root_id, [])
    if not root_chunks:
        return []
    limit = len(root_chunks)
    dense_ids, dense_scores = dense_ranking(
        index, query_vector, depth, where={"root_id": root_id}, limit=limit
    )
    query_tokens = tokenize_text(query_text, index.terms)
    bm25_ids, bm25_scores = bm25_ranking(index, query_tokens, depth, root_id=root_id)
    fused = rrf_fuse([dense_ids, bm25_ids], k=int(index.settings["retrieval"]["rrf_k"]))
    dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(dense_ids, start=1)}
    bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ids, start=1)}
    return [
        {
            "chunk_id": chunk_id,
            "score": score,
            "dense_rank": dense_rank.get(chunk_id),
            "bm25_rank": bm25_rank.get(chunk_id),
            "dense_score": dense_scores.get(chunk_id),
            "bm25_score": bm25_scores.get(chunk_id),
        }
        for chunk_id, score in fused
    ]


# ---------------------------------------------------------------------------
# 角色候选 -> 选案例 -> 选最终子块
# ---------------------------------------------------------------------------


def role_candidates(
    fused: Sequence[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    roles: Sequence[str],
    depth: int,
) -> dict[str, list[dict[str, Any]]]:
    """按角色切候选：每个角色取前 depth 个父块，每个父块只留一个代表子块。"""
    result: dict[str, list[dict[str, Any]]] = {role: [] for role in roles}
    role_set = set(roles)
    for row in fused:
        record = by_id.get(row["chunk_id"])
        if record is None:
            continue
        role = record["section_role"]
        if role not in role_set:
            continue
        bucket = result[role]
        if len(bucket) >= depth:
            continue
        if any(item["chunk"]["parent_id"] == record["parent_id"] for item in bucket):
            continue
        bucket.append(
            {
                **row,
                "chunk": record,
                "role": role,
                "rank": len(bucket) + 1,
            }
        )
    return result


def score_roots(
    candidates: dict[str, list[dict[str, Any]]],
    roles: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """给每个案例打分：先看覆盖几个角色，再看各角色最佳候选的 RRF 分数之和。"""
    stats: dict[str, dict[str, Any]] = {}
    for role in roles:
        best_by_root: dict[str, dict[str, Any]] = {}
        for candidate in candidates.get(role, []):
            root_id = candidate["chunk"]["root_id"]
            current = best_by_root.get(root_id)
            if current is None or candidate["score"] > current["score"]:
                best_by_root[root_id] = candidate
        for root_id, candidate in best_by_root.items():
            item = stats.setdefault(root_id, {"coverage": 0, "score_sum": 0.0, "roles": []})
            item["coverage"] += 1
            item["score_sum"] += float(candidate["score"])
            item["roles"].append(role)
    return stats


def select_root(
    candidates: dict[str, list[dict[str, Any]]],
    roles: Sequence[str],
) -> str | None:
    """选案例：角色覆盖数优先 -> 候选分数和 -> root_id（保证确定性）。"""
    stats = score_roots(candidates, roles)
    if not stats:
        return None
    return sorted(
        stats.items(),
        key=lambda item: (-item[1]["coverage"], -item[1]["score_sum"], item[0]),
    )[0][0]


def select_role_chunks(
    candidates: dict[str, list[dict[str, Any]]],
    root_id: str,
    roles: Sequence[str],
    quotas: dict[str, int],
    default_quota: int,
) -> dict[str, list[dict[str, Any]]]:
    """在选定案例内按角色名额取子块；每个角色每个父块最多一个。"""
    selected: dict[str, list[dict[str, Any]]] = {}
    for role in roles:
        quota = int(quotas.get(role, default_quota))
        chosen: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        for candidate in candidates.get(role, []):
            record = candidate["chunk"]
            if record["root_id"] != root_id:
                continue
            if record["parent_id"] in seen_parents:
                continue
            chosen.append({**candidate, "source": "role"})
            seen_parents.add(record["parent_id"])
            if len(chosen) >= quota:
                break
        selected[role] = chosen
    return selected


def backfill_missing_roles(
    index: SearchIndex,
    query_text: str,
    query_vector: Sequence[float],
    root_id: str,
    roles: Sequence[str],
    selected: dict[str, list[dict[str, Any]]],
    depth: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """缺角色只补一次：在已选案例内部做一次根内召回，按角色各取最高名次。"""
    missing = [role for role in roles if not selected.get(role)]
    if not missing:
        return selected, []
    local = root_local_recall(index, query_text, query_vector, root_id, depth)
    used_ids = {
        candidate["chunk"]["chunk_id"]
        for items in selected.values()
        for candidate in items
    }
    added: list[str] = []
    for role in missing:
        for row in local:
            record = index.by_id.get(row["chunk_id"])
            if record is None or record["section_role"] != role:
                continue
            if record["chunk_id"] in used_ids:
                continue
            selected[role] = [
                {
                    **row,
                    "chunk": record,
                    "role": role,
                    "rank": 1,
                    "source": "backfill",
                }
            ]
            used_ids.add(record["chunk_id"])
            added.append(role)
            break
    return selected, added


# ---------------------------------------------------------------------------
# 父块展开与证据选择
# ---------------------------------------------------------------------------


def expand_parent(
    by_parent: dict[str, list[dict[str, Any]]],
    selected: Sequence[dict[str, Any]],
    window: int,
    parent_max_chars: int,
    role_max_chars: int,
) -> list[dict[str, Any]]:
    """父块展开：命中子块前后各 window 块，按父块/角色字数上限截断。

    先放所有命中块，再按距离放兄弟块；这样角色名额 3 个时，不会因为第一个父块
    把 1,500 字吃满而把后两个命中块挤掉。
    """
    if not selected:
        return []
    seen: set[str] = set()
    parent_chars: dict[str, int] = {}
    role_chars = 0
    output: list[dict[str, Any]] = []

    def add(record: dict[str, Any], is_hit: bool, distance: int) -> bool:
        nonlocal role_chars
        chunk_id = record["chunk_id"]
        if chunk_id in seen:
            return False
        length = len(record.get("text", ""))
        parent_id = record["parent_id"]
        if not is_hit and parent_chars.get(parent_id, 0) + length > parent_max_chars:
            return False
        if not is_hit and role_chars + length > role_max_chars:
            return False
        seen.add(chunk_id)
        parent_chars[parent_id] = parent_chars.get(parent_id, 0) + length
        role_chars += length
        output.append({"chunk": record, "is_hit": is_hit, "distance": distance})
        return True

    # 第一遍：命中块无条件先放，保证每个角色名额都有原文可读。
    for candidate in selected:
        add(candidate["chunk"], is_hit=True, distance=0)

    # 第二遍：按距离补兄弟块，受父块和角色字数上限约束。
    for candidate in selected:
        hit = candidate["chunk"]
        siblings = by_parent.get(hit["parent_id"], [])
        hit_pos = next(
            (pos for pos, record in enumerate(siblings) if record["chunk_id"] == hit["chunk_id"]),
            None,
        )
        if hit_pos is None:
            continue
        for distance in range(1, window + 1):
            if role_chars >= role_max_chars:
                break
            for pos in (hit_pos - distance, hit_pos + distance):
                if 0 <= pos < len(siblings):
                    add(siblings[pos], is_hit=False, distance=distance)

    output.sort(key=lambda item: (item["chunk"]["parent_id"], item["chunk"]["chunk_index"]))
    return output


def assemble_evidence(
    by_parent: dict[str, list[dict[str, Any]]],
    by_root: dict[str, list[dict[str, Any]]],
    root_id: str,
    selected: dict[str, list[dict[str, Any]]],
    content_types: Sequence[str],
    quota: int,
) -> list[dict[str, Any]]:
    """在选中案例内按 content_type 取证据；优先选中父块，再补案例内其他父块。

    证据不是角色，所以不占用角色名额；执行计划/统计输出按 content_types 顺序轮转，
    保证两种证据都有机会进入前 quota 条。
    """
    if not content_types or quota <= 0:
        return []
    selected_parents: list[str] = []
    for items in selected.values():
        for candidate in items:
            parent_id = candidate["chunk"]["parent_id"]
            if parent_id not in selected_parents:
                selected_parents.append(parent_id)

    other_parents: list[str] = []
    for record in by_root.get(root_id, []):
        parent_id = record["parent_id"]
        if parent_id not in selected_parents and parent_id not in other_parents:
            other_parents.append(parent_id)

    buckets: dict[str, list[dict[str, Any]]] = {content_type: [] for content_type in content_types}
    for priority, parents in enumerate((selected_parents, other_parents)):
        for parent_order, parent_id in enumerate(parents):
            for record in by_parent.get(parent_id, []):
                content_type = record["content_type"]
                if content_type in buckets:
                    buckets[content_type].append(
                        {
                            "chunk": record,
                            "priority": priority,
                            "parent_order": parent_order,
                        }
                    )

    for content_type in content_types:
        buckets[content_type].sort(
            key=lambda item: (
                item["priority"],
                item["parent_order"],
                item["chunk"]["chunk_index"],
                item["chunk"]["chunk_id"],
            )
        )

    picked: list[dict[str, Any]] = []
    while len(picked) < quota and any(buckets.values()):
        progressed = False
        for content_type in content_types:
            if len(picked) >= quota:
                break
            if buckets[content_type]:
                picked.append(buckets[content_type].pop(0))
                progressed = True
        if not progressed:
            break

    return [
        {
            "chunk_id": item["chunk"]["chunk_id"],
            "parent_id": item["chunk"]["parent_id"],
            "chunk_index": item["chunk"]["chunk_index"],
            "content_type": item["chunk"]["content_type"],
            "title": item["chunk"].get("title", ""),
            "section_path": item["chunk"].get("section_path", ""),
            "text": item["chunk"].get("text", ""),
            "from_selected_parent": item["priority"] == 0,
        }
        for item in picked
    ]


# ---------------------------------------------------------------------------
# 模型 / 向量库装载
# ---------------------------------------------------------------------------


def _validate_index_meta(
    meta: dict[str, Any],
    settings: dict[str, Any],
    chunks_file: Path,
    tags_file: Path,
) -> None:
    """索引与当前配置/数据必须同版；不一致就要求重建，不静默混用。"""
    cfg = retrieval_config(settings)
    expected = {
        "embedding_model": cfg["embedding_model"],
        "chroma_collection": cfg["chroma_collection"],
        "bm25_k1": float(cfg["bm25_k1"]),
        "bm25_b": float(cfg["bm25_b"]),
        "chunks_sha256": file_sha256(chunks_file),
        "tags_sha256": file_sha256(tags_file),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"索引与当前数据/配置不一致：{key} 索引={meta.get(key)!r} 当前={value!r}；"
                "请先运行 python scripts/build_index.py --force"
            )


def load_search_index(root: Path, settings: dict[str, Any]) -> SearchIndex:
    """装载 Chroma、BM25、chunks 和 embedding 模型；所有硬校验在此完成。"""
    cfg = retrieval_config(settings)
    index_dir = resolve_path(root, cfg["index_dir"])
    meta_path = index_dir / "index_meta.json"
    chroma_dir = index_dir / "chroma"
    bm25_path = index_dir / "bm25_tokens.jsonl"
    chunks_file = resolve_path(root, settings["processed_dir"]) / "chunks.jsonl"
    tags_file = root / "config" / "tags.yaml"

    if not meta_path.exists() or not bm25_path.exists() or not chroma_dir.exists():
        raise FileNotFoundError(
            f"索引不完整：{index_dir}；请先运行 python scripts/build_index.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    _validate_index_meta(meta, settings, chunks_file, tags_file)

    chunks = read_jsonl(chunks_file)
    by_id = {record["chunk_id"]: record for record in chunks}
    by_parent: dict[str, list[dict[str, Any]]] = {}
    by_root: dict[str, list[dict[str, Any]]] = {}
    for record in chunks:
        by_parent.setdefault(record["parent_id"], []).append(record)
        by_root.setdefault(record["root_id"], []).append(record)
    for items in by_parent.values():
        items.sort(key=lambda record: record["chunk_index"])
    for items in by_root.values():
        items.sort(key=lambda record: (record["parent_id"], record["chunk_index"]))

    token_rows = read_jsonl(bm25_path)
    tokens_by_id = {row["chunk_id"]: row.get("tokens", []) for row in token_rows}
    if set(tokens_by_id) != set(by_id):
        raise RuntimeError("bm25_tokens.jsonl 与 chunks.jsonl 的 chunk_id 不一致，请重建索引")
    bm25_ids = [record["chunk_id"] for record in chunks]
    bm25 = BM25Okapi(
        [tokens_by_id[chunk_id] for chunk_id in bm25_ids],
        k1=float(cfg["bm25_k1"]),
        b=float(cfg["bm25_b"]),
    )

    with tags_file.open("r", encoding="utf-8") as handle:
        tags_cfg = yaml.safe_load(handle)
    terms = collect_terms(tags_cfg)

    # 延迟导入：纯函数测试不需要 sentence-transformers / chromadb。
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    import chromadb  # noqa: PLC0415

    model = SentenceTransformer(cfg["embedding_model"], device=cfg.get("device", "cpu"))
    model.max_seq_length = int(cfg["max_seq_length"])
    dimension_getter = getattr(model, "get_embedding_dimension", None)
    if dimension_getter is None:
        dimension_getter = model.get_sentence_embedding_dimension
    if dimension_getter() != int(meta["embedding_dim"]):
        raise RuntimeError("模型维度与索引元数据不一致，请重建索引")

    # 必须传相对路径：Chroma 1.5.9 在 Windows 下对含中文的绝对路径加载不了 HNSW。
    client = chromadb.PersistentClient(path=str(chroma_relative_path(chroma_dir)))
    collection = client.get_collection(name=cfg["chroma_collection"])
    if collection.count() != len(chunks):
        raise RuntimeError(
            f"Chroma 向量数 {collection.count()} 与 chunks {len(chunks)} 不一致，请重建索引"
        )

    return SearchIndex(
        settings=settings,
        chunks=chunks,
        by_id=by_id,
        by_parent=by_parent,
        by_root=by_root,
        bm25=bm25,
        bm25_ids=bm25_ids,
        terms=terms,
        model=model,
        collection=collection,
        meta=meta,
        index_dir=index_dir,
    )


def encode_query(index: SearchIndex, query: str) -> list[float]:
    """按 bge 模型卡要求给查询加指令；文档端不加指令。"""
    instruction = str(retrieval_config(index.settings).get("query_instruction", ""))
    text = f"{instruction}{query}" if instruction else query
    vector = index.model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]
    return [float(value) for value in vector]


# ---------------------------------------------------------------------------
# 对外检索入口
# ---------------------------------------------------------------------------


def _compact_chunk(record: dict[str, Any]) -> dict[str, Any]:
    """检索结果只保留复核需要的字段，不把内部元数据整包吐出。"""
    return {
        "chunk_id": record["chunk_id"],
        "root_id": record["root_id"],
        "parent_id": record["parent_id"],
        "chunk_index": record["chunk_index"],
        "section_role": record["section_role"],
        "content_type": record["content_type"],
        "title": record.get("title", ""),
        "section_path": record.get("section_path", ""),
    }


def _global_results(
    index: SearchIndex,
    rows: Sequence[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows[:top_k]:
        record = index.by_id.get(row["chunk_id"])
        if record is None:
            continue
        results.append(
            {
                **_compact_chunk(record),
                "score": row["score"],
                "dense_rank": row.get("dense_rank"),
                "bm25_rank": row.get("bm25_rank"),
                "text": record.get("text", ""),
            }
        )
    return results


def search(
    index: SearchIndex,
    query: str,
    mode: str = "role",
    required_roles: Sequence[str] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """单条查询入口。

    mode="role"：角色选子块 -> 选案例 -> 名额 -> 缺角色补一次 -> 父块展开 -> 证据。
    mode="global"：只用子块全局 topK，作为 Recall@5 基线，不做角色/父块处理。
    required_roles=None 时用问句规则推断；显式传入时用于评测/调试。
    """
    started = time.perf_counter()
    cfg = retrieval_config(index.settings)
    report_roles = list(cfg.get("report_roles", []))
    inferred = infer_required_roles(query, report_roles)
    roles = list(required_roles) if required_roles is not None else inferred
    # 去重但保留顺序；空列表表示明确不要角色，走全局兜底。
    roles = list(OrderedDict.fromkeys(roles))

    if mode == "global" or not roles:
        query_vector = encode_query(index, query)
        depth = int(top_k if top_k is not None else cfg["global_top_k"])
        rows = child_recall(index, query, query_vector, depth)
        return {
            "query": query,
            "mode": "global" if mode == "global" else "global_fallback",
            "inferred_roles": inferred,
            "requested_roles": roles,
            "results": _global_results(index, rows, depth),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    if mode != "role":
        raise ValueError(f"未知检索模式：{mode}")

    query_vector = encode_query(index, query)
    fused = child_recall(index, query, query_vector, int(cfg["recall_depth"]))
    candidates = role_candidates(
        fused,
        index.by_id,
        roles,
        depth=int(cfg["role_candidate_depth"]),
    )
    root_id = select_root(candidates, roles)
    if root_id is None:
        rows = child_recall(index, query, query_vector, int(cfg["global_top_k"]))
        return {
            "query": query,
            "mode": "global_fallback",
            "inferred_roles": inferred,
            "requested_roles": roles,
            "results": _global_results(index, rows, int(cfg["global_top_k"])),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }

    selected = select_role_chunks(
        candidates,
        root_id,
        roles,
        quotas={key: int(value) for key, value in cfg.get("role_quotas", {}).items()},
        default_quota=int(cfg.get("default_role_quota", 2)),
    )
    selected, backfilled = backfill_missing_roles(
        index,
        query,
        query_vector,
        root_id,
        roles,
        selected,
        depth=int(cfg["recall_depth"]),
    )
    missing_roles = [role for role in roles if not selected.get(role)]

    contexts: list[dict[str, Any]] = []
    for role in roles:
        if not selected.get(role):
            continue
        chunks = expand_parent(
            index.by_parent,
            selected[role],
            window=int(cfg["parent_window"]),
            parent_max_chars=int(cfg["parent_max_chars"]),
            role_max_chars=int(cfg["role_context_max_chars"]),
        )
        contexts.append(
            {
                "role": role,
                "chars": sum(len(item["chunk"].get("text", "")) for item in chunks),
                "chunks": [
                    {
                        **_compact_chunk(item["chunk"]),
                        "is_hit": item["is_hit"],
                        "distance": item["distance"],
                        "text": item["chunk"].get("text", ""),
                    }
                    for item in chunks
                ],
            }
        )

    evidence = assemble_evidence(
        index.by_parent,
        index.by_root,
        root_id,
        selected,
        content_types=list(cfg.get("evidence_content_types", [])),
        quota=int(cfg.get("evidence_quota", 3)),
    )

    unique_ids = {
        item["chunk_id"]
        for context in contexts
        for item in context["chunks"]
    } | {item["chunk_id"] for item in evidence}
    context_chars = sum(len(index.by_id[chunk_id].get("text", "")) for chunk_id in unique_ids)
    role_context_chars = sum(context["chars"] for context in contexts)
    evidence_chars = sum(len(item.get("text", "")) for item in evidence)

    selected_compact = {
        role: [
            {
                **_compact_chunk(candidate["chunk"]),
                "score": candidate["score"],
                "rank": candidate["rank"],
                "source": candidate["source"],
            }
            for candidate in items
        ]
        for role, items in selected.items()
    }

    root_chunks = index.by_root.get(root_id, [])
    root_title = root_chunks[0].get("source_name", "") if root_chunks else ""
    root_stats = score_roots(candidates, roles).get(root_id, {})
    dense_count = sum(1 for row in fused if row.get("dense_rank") is not None)
    bm25_count = sum(1 for row in fused if row.get("bm25_rank") is not None)

    return {
        "query": query,
        "mode": "role",
        "inferred_roles": inferred,
        "requested_roles": roles,
        "root_id": root_id,
        "root_title": root_title,
        "root_debug": {
            "coverage": root_stats.get("coverage", 0),
            "score_sum": root_stats.get("score_sum", 0.0),
            "covered_roles": root_stats.get("roles", []),
        },
        "selected": selected_compact,
        "candidate_top3": {
            role: [
                {
                    "chunk_id": candidate["chunk"]["chunk_id"],
                    "root_id": candidate["chunk"]["root_id"],
                    "parent_id": candidate["chunk"]["parent_id"],
                    "chunk_index": candidate["chunk"]["chunk_index"],
                    "score": candidate["score"],
                    "rank": candidate["rank"],
                }
                for candidate in candidates.get(role, [])[:3]
            ]
            for role in roles
        },
        "missing_roles": missing_roles,
        "backfilled_roles": backfilled,
        "role_contexts": contexts,
        "evidence": evidence,
        "context_chars": context_chars,
        "role_context_chars": role_context_chars,
        "evidence_chars": evidence_chars,
        "recall_debug": {
            "dense_count": dense_count,
            "bm25_count": bm25_count,
            "fused_count": len(fused),
            "candidate_counts": {role: len(candidates.get(role, [])) for role in roles},
        },
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }

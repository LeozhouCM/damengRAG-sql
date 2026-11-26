# 本文件做什么：1B 纯函数单元测试，锁定角色推断、RRF、选案例、名额、父块展开、证据选择。
# 被谁使用：pytest；不加载 sentence-transformers / Chroma，不需要网络。
# 运行：
#     python -m pytest -q
# 依据：src/knowledge/retrieval.py 的规则；实践计划书 表42 的检索流程。
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.knowledge import retrieval
from src.knowledge.retrieval import (
    assemble_evidence,
    backfill_missing_roles,
    expand_parent,
    infer_required_roles,
    role_candidates,
    rrf_fuse,
    score_roots,
    select_role_chunks,
    select_root,
    tokenize_text,
)

REPORT_ROLES = ["问题", "原因", "办法", "验证"]


def make_chunk(
    chunk_id: str,
    parent_id: str,
    chunk_index: int,
    role: str,
    content_type: str = "文字",
    text: str | None = None,
    root_id: str = "case-999",
) -> dict:
    """造一个与 chunks.jsonl 同构的最小记录；只填被测函数真正读取的字段。"""
    return {
        "chunk_id": chunk_id,
        "root_id": root_id,
        "parent_id": parent_id,
        "chunk_index": chunk_index,
        "section_role": role,
        "content_type": content_type,
        "text": text if text is not None else f"{chunk_id} 的测试文本",
        "title": f"标题 {parent_id}",
        "section_path": f"测试文档 > {parent_id}",
        "source_type": "case",
        "source_name": "测试文档",
    }


def make_candidate(
    chunk: dict,
    score: float,
    rank: int = 1,
) -> dict:
    """造一个 role_candidates 风格的候选。"""
    return {
        "chunk_id": chunk["chunk_id"],
        "score": score,
        "dense_rank": rank,
        "bm25_rank": None,
        "dense_score": None,
        "bm25_score": None,
        "chunk": chunk,
        "role": chunk["section_role"],
        "rank": rank,
    }


# ---------------------------------------------------------------------------
# 问句角色推断
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("这条 left join 和 inner join 混用的 SQL 慢的原因是什么？优化思路是什么？", ["原因", "办法"]),
        ("谓词下推为什么能减少大表的扫描行数？优化后怎么验证效果？", ["原因", "办法", "验证"]),
        ("问题 SQL 的执行计划暴露了什么问题，原因是什么，怎么验证效果？", ["问题", "原因", "验证"]),
        ("排序加分页的 SQL 问题出在哪？怎么通过索引和 Hint 优化，并验证优化后的计划？", ["问题", "办法", "验证"]),
        ("分页排序 SQL 的 SORT 能不能消除？TOP_ORDER_OPT_FLAG 应该怎么调整，具体操作步骤是什么？", ["办法", "步骤"]),
        ("请总结 DM 中 rowid 是什么，什么场景可以用 rowid 快速过滤数据？", ["概念", "总结"]),
        ("达梦的统计信息有哪些种类，对优化器有什么作用？", ["概念"]),
        ("达梦统计信息应该在查询前收集还是查询时收集？", ["步骤"]),
        ("分组和分析函数为什么需要索引？", ["原因"]),
        ("今天天气怎么样", []),
    ],
)
def test_infer_required_roles(query: str, expected: list[str]) -> None:
    assert infer_required_roles(query, REPORT_ROLES) == expected


def test_infer_required_roles_truncates_to_max() -> None:
    query = "这个问题为什么慢，怎么优化，怎么验证效果，具体操作步骤是什么？"
    roles = infer_required_roles(query, REPORT_ROLES, max_roles=2)
    assert len(roles) == 2
    # 截断保持报告顺序（问题 -> 原因 -> 办法 -> 验证），不是按命中词出现顺序。
    assert roles == ["问题", "原因"]


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def test_rrf_fuse_tie_break_is_deterministic() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "a"]], k=60)
    # a/b 两路名的名次相同，按 chunk_id 稳定排序；c 只有一路，排最后。
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_rrf_fuse_ignores_duplicate_within_one_ranking() -> None:
    fused = rrf_fuse([["a", "a", "b"], []], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(1 / 61)


# ---------------------------------------------------------------------------
# 角色候选 / 选案例 / 名额
# ---------------------------------------------------------------------------


def test_role_candidates_keep_one_chunk_per_parent() -> None:
    records = [
        make_chunk("c1", "p1", 1, "原因"),
        make_chunk("c2", "p1", 2, "原因"),
        make_chunk("c3", "p2", 1, "原因"),
        make_chunk("c4", "p3", 1, "办法"),
    ]
    by_id = {record["chunk_id"]: record for record in records}
    fused = [
        {"chunk_id": "c1", "score": 0.03},
        {"chunk_id": "c2", "score": 0.02},
        {"chunk_id": "c3", "score": 0.01},
        {"chunk_id": "c4", "score": 0.005},
    ]
    candidates = role_candidates(fused, by_id, ["原因", "办法"], depth=10)
    assert [item["chunk"]["chunk_id"] for item in candidates["原因"]] == ["c1", "c3"]
    assert [item["chunk"]["chunk_id"] for item in candidates["办法"]] == ["c4"]


def test_score_roots_coverage_before_score_sum() -> None:
    reason_a = make_candidate(make_chunk("c1", "p1", 1, "原因"), 0.03)
    reason_b = make_candidate(make_chunk("c2", "p2", 1, "原因", root_id="case-002"), 0.10)
    method_a = make_candidate(make_chunk("c3", "p3", 1, "办法"), 0.02)
    verify_a = make_candidate(make_chunk("c4", "p4", 1, "验证"), 0.01)
    candidates = {
        "原因": [reason_a, reason_b],
        "办法": [method_a],
        "验证": [verify_a],
    }
    stats = score_roots(candidates, ["原因", "办法", "验证"])
    assert stats["case-999"]["coverage"] == 3
    assert stats["case-002"]["coverage"] == 1
    assert select_root(candidates, ["原因", "办法", "验证"]) == "case-999"


def test_select_root_breaks_tie_by_root_id() -> None:
    a = make_candidate(make_chunk("c1", "p1", 1, "原因", root_id="case-B"), 0.02)
    b = make_candidate(make_chunk("c2", "p2", 1, "原因", root_id="case-A"), 0.02)
    candidates = {"原因": [a, b]}
    assert select_root(candidates, ["原因"]) == "case-A"
    assert select_root({}, ["原因"]) is None


def test_select_role_chunks_quota_root_filter_and_parent_dedup() -> None:
    records = [
        make_chunk("c1", "p1", 1, "原因"),
        make_chunk("c2", "p1", 2, "原因"),
        make_chunk("c3", "p2", 1, "原因"),
        make_chunk("c4", "p3", 1, "原因"),
        make_chunk("c5", "p4", 1, "原因", root_id="case-other"),
    ]
    candidates = {
        "原因": [
            make_candidate(records[0], 0.05, 1),
            make_candidate(records[1], 0.04, 2),
            make_candidate(records[2], 0.03, 3),
            make_candidate(records[3], 0.02, 4),
            make_candidate(records[4], 0.01, 5),
        ],
        "办法": [],
    }
    selected = select_role_chunks(
        candidates,
        root_id="case-999",
        roles=["原因", "办法"],
        quotas={"原因": 2},
        default_quota=2,
    )
    assert [item["chunk"]["chunk_id"] for item in selected["原因"]] == ["c1", "c3"]
    assert selected["办法"] == []


# ---------------------------------------------------------------------------
# 父块展开 / 证据
# ---------------------------------------------------------------------------


def test_expand_parent_window_and_parent_cap() -> None:
    records = [
        make_chunk(f"c{index}", "p1", index, "原因", text="x" * 100)
        for index in range(1, 8)
    ]
    by_parent = {"p1": records}
    selected = [make_candidate(records[3], 0.05)]
    context = expand_parent(
        by_parent,
        selected,
        window=2,
        parent_max_chars=350,
        role_max_chars=1500,
    )
    assert [item["chunk"]["chunk_id"] for item in context] == ["c3", "c4", "c5"]
    assert [item["is_hit"] for item in context] == [False, True, False]


def test_expand_parent_keeps_all_hits_before_siblings() -> None:
    parent_a = [
        make_chunk(f"a{index}", "pa", index, "原因", text="x" * 100)
        for index in range(1, 5)
    ]
    parent_b = [
        make_chunk(f"b{index}", "pb", index, "原因", text="x" * 100)
        for index in range(1, 4)
    ]
    by_parent = {"pa": parent_a, "pb": parent_b}
    selected = [make_candidate(parent_a[1], 0.05), make_candidate(parent_b[1], 0.04)]
    context = expand_parent(
        by_parent,
        selected,
        window=1,
        parent_max_chars=1500,
        role_max_chars=250,
    )
    # 两个命中块先占 200 字；兄弟块再加就超过 250 字，所以只剩两个 HIT。
    assert [item["chunk"]["chunk_id"] for item in context] == ["a2", "b2"]
    assert all(item["is_hit"] for item in context)


def test_assemble_evidence_quota_and_type_rotation() -> None:
    selected_parent = make_chunk("e1", "p1", 1, "原因", "执行计划")
    stats_other = make_chunk("e2", "p2", 1, "办法", "统计输出")
    plan_other = make_chunk("e3", "p2", 2, "办法", "执行计划")
    plan_far = make_chunk("e4", "p3", 1, "验证", "执行计划")
    stats_far = make_chunk("e5", "p3", 2, "验证", "统计输出")
    by_parent = {
        "p1": [selected_parent],
        "p2": [stats_other, plan_other],
        "p3": [plan_far, stats_far],
    }
    by_root = {"case-999": [selected_parent, stats_other, plan_other, plan_far, stats_far]}
    selected = {"原因": [make_candidate(selected_parent, 0.05)]}
    evidence = assemble_evidence(
        by_parent,
        by_root,
        root_id="case-999",
        selected=selected,
        content_types=["执行计划", "统计输出"],
        quota=3,
    )
    assert [item["chunk_id"] for item in evidence] == ["e1", "e2", "e3"]
    assert evidence[0]["from_selected_parent"] is True
    assert evidence[1]["from_selected_parent"] is False


# ---------------------------------------------------------------------------
# 补角色 / 切词
# ---------------------------------------------------------------------------


def test_backfill_missing_roles_uses_one_root_local_recall(monkeypatch: pytest.MonkeyPatch) -> None:
    selected_chunk = make_chunk("c1", "p1", 1, "原因")
    missing_chunk = make_chunk("c2", "p2", 1, "办法")
    index = SimpleNamespace(by_id={"c1": selected_chunk, "c2": missing_chunk})
    calls = []

    def fake_root_local_recall(*args, **kwargs):
        calls.append(kwargs)
        return [
            {"chunk_id": "c1", "score": 0.03},
            {"chunk_id": "c2", "score": 0.02},
        ]

    monkeypatch.setattr(retrieval, "root_local_recall", fake_root_local_recall)
    selected = {"原因": [make_candidate(selected_chunk, 0.05)], "办法": []}
    result, added = backfill_missing_roles(
        index,
        "测试查询",
        [0.0],
        "case-999",
        ["原因", "办法"],
        selected,
        depth=50,
    )
    assert len(calls) == 1
    assert added == ["办法"]
    assert [item["chunk"]["chunk_id"] for item in result["办法"]] == ["c2"]
    assert result["办法"][0]["source"] == "backfill"


def test_tokenize_text_adds_domain_terms() -> None:
    tokens = tokenize_text("执行计划里 CSCN2 做了 HASH JOIN，索引失效", ["CSCN2", "HASH JOIN", "索引"])
    assert "cscn2" in tokens
    assert "hash join" in tokens
    assert "索引" in tokens
    assert all(token.strip() for token in tokens)

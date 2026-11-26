"""1A 单元测试：锁定切块、标签、父子块和清洗规则（防止改注释时改坏逻辑）。

覆盖的规则：
- 标题树：父标题正文独立成节；纯编号分组标题不进入 section_path；
- 父块：root_id/parent_id 按来源和标题生成，chunk_index 在父块内从 1 连续编号；
- 短块：只在同父块、同角色、同内容类型内合并；合不了就保留并标 short_chunk；
- 长正文：按自然句切分，目标 200/上限 300，生产参数不重叠；
- 代码：短代码跟正文同块；超长代码按行拆段，每段保留围栏，并标 code_split；
- 表格：长表按行拆并重复表头；无表头时打 table_without_header；
- 角色：案例用 section_roles，手册用 manual_section_roles，证据靠 content_type 区分；
- 清洗：纯编号标题补粗体标签；推广 H1 删除；H1 代码算 content；Statistics 输出标 text；
- 标签：主题/操作符/参数/SQL 特征；元数据字段齐全。

运行：
    python -m pytest -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.knowledge.ingest import (
    Chunk,
    Source,
    chunks_from_cleaned,
    classify_h1_intro,
    classify_section_role,
    clean_markdown,
    convert_indented_code,
    detect_content_type,
    extract_h1,
    extract_version,
    load_tags,
    merge_short_chunks,
    parse_sections,
    split_code_unit,
    split_long_text,
    tag_chunk,
)

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "chunk_min_chars": 80,
    "chunk_target_chars": 200,
    "chunk_max_chars": 300,
    "chunk_overlap_chars": 0,
}


@pytest.fixture(scope="module")
def tags_cfg():
    """真实标签词典，保证测试和 config/tags.yaml 一致。"""
    return load_tags(ROOT / "config" / "tags.yaml")


def make_source(
    intro_type: str = "none", source_type: str = "case", source_id: str | None = None
) -> Source:
    """造一条测试来源，字段与 sources.json 的登记结构保持一致。"""
    prefix = "doc" if source_type == "doc" else "case"
    return Source(
        source_id=source_id or f"{prefix}-999",
        source_type=source_type,
        source_name="测试文档",
        source_file="test.md",
        source_url=f"local://dm_{prefix}s/test.md",
        version="DM8 2025Q1",
        acquired_at="2026-11-17",
        intro_type=intro_type,
    )


def build(
    markdown: str,
    intro_type: str = "none",
    settings: dict | None = None,
    source_type: str = "case",
):
    """测试用快捷函数：一段 Markdown 直接走完整的切块+打标签流程。"""
    source = make_source(intro_type, source_type=source_type)
    records, warnings = chunks_from_cleaned(
        markdown,
        source,
        settings or SETTINGS,
        load_tags(ROOT / "config" / "tags.yaml"),
    )
    return records, warnings


# ---------- 标题树与 section_path ----------


def test_parse_sections_keeps_parent_body_and_path():
    markdown = (
        "# T\n\n摘要\n\n"
        "## A\n\nA正文\n\n"
        "### A1\n\nA1正文\n\n"
        "## B\n\nB正文\n"
    )
    sections = parse_sections(markdown)
    assert [section.path for section in sections] == [
        "T",
        "T > A",
        "T > A > A1",
        "T > B",
    ]
    assert sections[1].body == "A正文"
    assert sections[2].body == "A1正文"
    assert sections[3].body == "B正文"


def test_numeric_grouping_heading_is_not_in_section_path():
    markdown = "# T\n\n## 01\n\n### 简介\n\n正文。\n"
    sections = parse_sections(markdown)
    assert len(sections) == 1
    assert sections[0].path == "T > 简介"
    assert sections[0].title == "简介"


# ---------- 父子块与短块合并 ----------


def test_short_section_does_not_merge_across_parents():
    markdown = "# 文档\n\n## 问题一\n\n短文本。\n\n## 问题二\n\n另一个短文本。\n"
    records, _ = build(markdown)
    assert len(records) == 2
    assert records[0]["parent_id"] != records[1]["parent_id"]
    assert all("short_chunk" in record["_flags"] for record in records)
    assert "短文本" in records[0]["text"]
    assert "另一个短文本" in records[1]["text"]


def test_short_chunks_merge_only_within_same_parent_and_role():
    base = dict(title="t", section_path="p", section_role="问题", content_type="文字")
    first = Chunk(text="短。", parent_id="case-999-p001", **base)
    second = Chunk(text="同一父块的短块。", parent_id="case-999-p001", **base)
    other_parent = Chunk(text="不同父块的短块。", parent_id="case-999-p002", **base)
    merged, _ = merge_short_chunks([first, second, other_parent], 80, 300)
    assert len(merged) == 2
    assert "同一父块的短块" in merged[0].text
    assert "不同父块的短块" in merged[1].text
    assert "short_chunk" in merged[1].flags


def test_chunks_have_parent_id_and_contiguous_chunk_index():
    markdown = (
        "# 文档\n\n"
        "## 问题描述\n\n短。\n\n"
        "## 原因分析\n\n" + "这是一段原因分析。" * 60 + "\n"
    )
    records, _ = build(markdown)
    assert {record["root_id"] for record in records} == {"case-999"}
    assert records[0]["parent_id"] == "case-999-p001"
    assert records[0]["chunk_index"] == 1
    reason = [record for record in records if record["parent_id"] == "case-999-p002"]
    assert [record["chunk_index"] for record in reason] == list(
        range(1, len(reason) + 1)
    )
    assert len(reason) > 1


def test_h1_abstract_stays_standalone_even_below_min():
    markdown = (
        "# 测试文档\n\n短摘要。\n\n"
        "## A\n\n短文本。\n\n"
        "## B\n\n" + "长文本。" * 30
    )
    records, _ = build(markdown, intro_type="abstract")
    assert len(records) == 3
    assert records[0]["text"] == "短摘要。"
    assert records[0]["title"] == "测试文档"
    assert records[0]["section_role"] == "概念"
    assert "短文本" in records[1]["text"]
    assert "short_chunk" in records[1]["_flags"]


# ---------- 长正文切分 ----------


def test_long_text_split_without_overlap():
    text = "句子。" * 200
    pieces = split_long_text(text, target=200, max_chars=300, overlap=0)
    assert len(pieces) > 1
    assert all(len(piece) <= 300 for piece in pieces)
    assert sum(len(piece) for piece in pieces) == len(text)


def test_long_prose_chunk_has_text_split_flag():
    markdown = "# 文档\n\n## 正文\n\n" + "这是一句测试文本。" * 60
    records, _ = build(markdown)
    assert len(records) > 1
    assert all("text_split" in record["_flags"] for record in records)
    assert sum(len(record["text"]) for record in records) == len("这是一句测试文本。" * 60)


# ---------- 代码块：短代码同块、长代码拆段、上下文 ----------


def test_long_code_is_split_by_lines_with_balanced_fences():
    code = "\n".join(
        [f"select {index} from t; -- 注释{index}" for index in range(60)]
    )
    fenced = f"```sql\n-- 原SQL\n{code}\n-- 优化后SQL\n```"
    markdown = "# 文档\n\n## 代码\n\n" + fenced
    records, _ = build(markdown)
    assert len(records) > 1
    assert all(len(record["text"]) <= 300 for record in records)
    assert all(record["text"].count("```") == 2 for record in records)
    assert all("code_split" in record["_flags"] for record in records)
    assert "-- 原SQL" in records[0]["text"]
    assert "-- 优化后SQL" in records[-1]["text"]
    assert all(record["parent_id"] == "case-999-p001" for record in records)
    assert [record["chunk_index"] for record in records] == list(
        range(1, len(records) + 1)
    )


def test_split_code_unit_never_merges_last_piece_over_max():
    lines = [f"line{index:02d}" + "x" * 24 for index in range(10)]
    text = "```text\n" + "\n".join(lines) + "\n```"
    pieces = split_code_unit(text, 300, 80)
    assert len(pieces) == 2
    assert all(len(piece) <= 300 for piece, _ in pieces)
    assert all("code_split" in flags for _, flags in pieces)


def test_stats_then_ascii_table_marks_table_rows_separately():
    stats_lines = [
        "Statistics",
        "------------------------------------------------------------",
        "     1  80                   1         1 0  1.000000",
    ]
    header = "OP     TIME(US)   PERCENT RANK  SEQ  N_ENTER  N_ROWS"
    separator = "------ ---------- ------- ----- ------ ------- -------"
    rows = [
        f"PRJT2  {index:<9} {index:>5}%  {index:>5} {index:>5} {index:>7} {index:>7}"
        for index in range(1, 13)
    ]
    text = "```text\n" + "\n".join(stats_lines + [header, separator] + rows) + "\n```"
    pieces = split_code_unit(text, 300, 80)
    flags_by_piece = [flags for _, flags in pieces]
    assert any("stat_code" in flags for flags in flags_by_piece)
    assert any("table_code" in flags for flags in flags_by_piece)
    # 表头那一块也要标 table_code，不能只标分隔线/数据行
    header_pieces = [piece for piece, _ in pieces if header in piece]
    assert header_pieces
    assert all(
        "table_code" in flags
        for piece, flags in pieces
        if header in piece
    )


def test_short_code_and_preceding_explanation_stay_together():
    explanation = "这段说明解释优化前后的 SQL 为什么要改写。" * 4
    fenced = "```sql\nselect * from t where id = 1;\n```"
    markdown = f"# 文档\n\n## 代码\n\n{explanation}\n\n{fenced}"
    records, _ = build(markdown)
    assert len(records) == 1
    assert explanation in records[0]["text"]
    assert "select * from t" in records[0]["text"]


def test_short_lead_in_is_attached_to_following_long_code():
    long_prose = "上一节的长正文。" * 30
    code = "```sql\n" + "\n".join(
        f"select {index} from t; -- 超长代码{index}" for index in range(80)
    ) + "\n```"
    markdown = (
        "# 文档\n\n## A\n\n"
        f"{long_prose}\n\n"
        "## B\n\n短介绍。\n\n"
        f"{code}\n"
    )
    records, _ = build(markdown)
    intro = records[1]
    assert intro["title"] == "B"
    assert intro["text"].startswith("短介绍。\n\n```sql")
    assert "short_chunk" not in intro["_flags"]
    assert intro["content_type"] == "SQL"
    assert all(record["text"] != "短介绍。" for record in records)


def test_short_lead_in_is_attached_to_following_table():
    rows = "\n".join(f"| {index} | name{index} |" for index in range(40))
    table = "| id | name |\n| --- | --- |\n" + rows
    markdown = "# 文档\n\n## 表格\n\n参数列表如下：\n\n" + table
    records, _ = build(markdown)
    assert len(records) > 1
    assert records[0]["text"].startswith("参数列表如下：\n\n| id | name |")
    assert all("table_split" in record["_flags"] for record in records)


def test_code_context_uses_explanation_not_short_label():
    explanation = "这是一段足够长的解释文字，用来说明第二个代码块的必要背景。" * 3
    code_one = "```sql\nselect 1 from dual;\n```"
    code_two = "```text\n" + "\n".join(
        f"#OP{index}: [1, 1, 1]" for index in range(17)
    ) + "\n```"
    markdown = (
        "# 文档\n\n## 代码\n\n"
        f"{explanation}\n\nSQL：\n\n{code_one}\n\n{code_two}\n"
    )
    records, _ = build(markdown)
    assert len(records) == 2
    assert records[1]["text"].startswith("这是一段足够长的")
    assert not records[1]["text"].startswith("SQL：")


# ---------- 表格 ----------


def test_long_table_splits_with_repeated_header():
    rows = "\n".join(f"| {index} | name{index} |" for index in range(40))
    table = "| id | name |\n| --- | --- |\n" + rows
    markdown = "# 文档\n\n## 表格\n\n" + table
    records, _ = build(markdown)
    assert len(records) > 1
    for record in records:
        lines = record["text"].splitlines()
        assert lines[0] == "| id | name |"
        assert "---" in lines[1]
    assert all("table_split" in record["_flags"] for record in records)


def test_long_ascii_table_rows_are_typed_as_table():
    header = "OP     TIME(US)             PERCENT RANK"
    separator = "------ -------------------- ------- ----"
    row = "PRJT2  7                    0%      10"
    markdown = (
        "# 文档\n\n## 表格\n\n```text\n"
        + header
        + "\n"
        + separator
        + "\n"
        + "\n".join([row] * 20)
        + "\n```"
    )
    records, _ = build(markdown)
    assert len(records) > 1
    assert all(record["content_type"] == "表格" for record in records)
    assert all("table_code" in record["_flags"] for record in records)
    assert all("code_split" in record["_flags"] for record in records)


# ---------- 清洗与标题化 ----------


def test_numeric_heading_gets_bold_label():
    raw = "# 文档\n\n## 01\n\n**原始SQL**\n\n内容\n"
    cleaned, _ = clean_markdown(raw, "none")
    assert "## 01 原始SQL" in cleaned


def test_boilerplate_h1_body_is_removed():
    raw = (
        "# 文档\n\n"
        "ps：推荐阅读《某文章》\n"
        "https://mp.weixin.qq.com/s/example\n\n"
        "## 正文\n\n正文内容。\n"
    )
    title, body = extract_h1(raw)
    assert title == "文档"
    assert classify_h1_intro(body) == "boilerplate"
    cleaned, _ = clean_markdown(raw, "boilerplate")
    assert "https://" not in cleaned
    assert "正文内容" in cleaned


def test_h1_code_is_content_not_abstract():
    body = "说明\n\n    select 1 from dual;\n"
    assert classify_h1_intro(body) == "content"


def test_statistics_output_is_text_fence_not_sql():
    lines = [
        "    Statistics",
        "    -------------------------------------------------",
        "    2531 logical reads",
        "    147  exec time(ms)",
    ]
    converted, _ = convert_indented_code(lines)
    assert "```text" in converted
    assert "```sql" not in converted


# ---------- 角色与内容类型 ----------


def test_case_section_roles_follow_title_keywords(tags_cfg):
    assert classify_section_role("问题现象", tags_cfg) == "问题"
    assert classify_section_role("原因分析", tags_cfg) == "原因"
    assert classify_section_role("优化方案", tags_cfg) == "办法"
    assert classify_section_role("优化前后对比", tags_cfg) == "验证"
    assert classify_section_role("任意标题", tags_cfg, is_h1=True) == "概念"
    assert classify_section_role("无关标题", tags_cfg) == "未分类"


def test_manual_section_roles_use_manual_vocabulary(tags_cfg):
    assert classify_section_role("查看执行计划", tags_cfg, source_type="doc") == "步骤"
    assert classify_section_role("参数说明", tags_cfg, source_type="doc") == "参考"
    assert classify_section_role("性能概述", tags_cfg, source_type="doc") == "概念"
    # 手册里的“优化建议”不能当成案例的“办法”，否则会污染报告拼接。
    assert classify_section_role("优化建议", tags_cfg, source_type="doc") == "概念"


def test_manual_chunks_use_doc_source_and_manual_roles():
    markdown = (
        "# DM9 手册\n\n"
        "## 查看执行计划\n\n" + "用 EXPLAIN 查看执行计划。" * 12 + "\n\n"
        "## 参数说明\n\n" + "这里列出与查询优化有关的参数。" * 12 + "\n\n"
        "## 性能概述\n\n" + "性能问题通常需要结合执行计划和统计信息判断。" * 12 + "\n"
    )
    records, _ = build(markdown, source_type="doc")
    assert {record["root_id"] for record in records} == {"doc-999"}
    roles = {record["title"]: record["section_role"] for record in records}
    assert roles["查看执行计划"] == "步骤"
    assert roles["参数说明"] == "参考"
    assert roles["性能概述"] == "概念"


def test_content_type_precedence():
    assert detect_content_type("普通说明文字。") == "文字"
    assert detect_content_type("```sql\nselect 1 from dual;\n```") == "SQL"
    assert detect_content_type("```text\n1 #NSET2: [1, 1, 1]\n```") == "执行计划"
    assert detect_content_type("```text\nStatistics\n1 logical reads\n```") == "统计输出"
    assert detect_content_type("```text\n| A | B |\n| 1 | 2 |\n```") == "表格"
    assert detect_content_type("| a | b |\n| --- | --- |\n| 1 | 2 |") == "表格"
    assert (
        detect_content_type(
            "```text\nOP    TIME\n------ ------ ------\nPRJT2 7\n```"
        )
        == "表格"
    )
    assert detect_content_type("$ disql SYSDBA/SYSDBA") == "命令"


def test_plan_inside_reason_section_does_not_change_role():
    markdown = (
        "# 文档\n\n## 原因分析\n\n"
        + "统计信息过期导致行数估算错误。" * 8
        + "\n\n```text\n1 #NSET2: [1, 1, 1]\n2 #CSCN2: [1, 10, 1]\n```\n"
    )
    records, _ = build(markdown)
    assert all(record["section_role"] == "原因" for record in records)
    assert any(record["content_type"] == "执行计划" for record in records)


# ---------- 版本识别 ----------


def test_extract_version_supports_dm8_dm9_and_missing():
    assert extract_version("DM8（2024）Q3") == "DM8 2024Q3"
    assert extract_version("DM9 系统管理员手册") == "DM9"
    assert extract_version("没有版本号的文章") == "未注明"


# ---------- 元数据与标签 ----------


def test_metadata_fields_and_tags(tags_cfg):
    markdown = (
        "# 测试文档\n\n"
        "## 01 执行计划\n\n"
        "```text\n1 #NSET2: [1, 1, 1]\n```\n\n"
        "参数 TOP_ORDER_OPT_FLAG 用于消除排序，这是测试用的说明文字，需要保证块长度超过八十个字符。\n"
    )
    records, _ = build(markdown)
    assert len(records) == 1
    record = records[0]
    assert set(record) == {
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
        "_flags",
    }
    assert record["chunk_id"] == "case-999-c001"
    assert record["root_id"] == "case-999"
    assert record["parent_id"] == "case-999-p001"
    assert record["chunk_index"] == 1
    assert record["source_type"] == "case"
    assert record["source_name"] == "测试文档"
    assert record["source_url"] == "local://dm_cases/test.md"
    assert record["version"] == "DM8 2025Q1"
    assert record["section_path"] == "测试文档 > 01 执行计划"
    assert record["section_role"] == "未分类"
    assert record["content_type"] == "执行计划"
    assert "执行计划" in record["tags"]
    assert "NSET2" in record["tags"]
    assert "TOP_ORDER_OPT_FLAG" in record["tags"]


def test_tag_rules_for_operator_parameter_and_sql_features(tags_cfg):
    text = (
        "1 #NSET2: [1, 1, 1]\n"
        "2 #CSCN2: [1, 10, 1]\n"
        "TOP_ORDER_OPT_FLAG=1;\n"
        "select * from t where a like '%x';\n"
    )
    tags, warnings = tag_chunk(text, tags_cfg)
    assert warnings == []
    assert tags[0] == "执行计划"
    for expected in ("NSET2", "CSCN2", "TOP_ORDER_OPT_FLAG", "SELECT *", "前导模糊"):
        assert expected in tags


# ---------- 表格缺表头告警 ----------


def test_table_without_header_is_flagged_by_splitter():
    markdown = "# 文档\n\n## 表格\n\n| a | b |\n| 1 | 2 |\n" + "| x | y |\n" * 60
    records, _ = build(markdown)
    assert records
    assert all("table_without_header" in record["_flags"] for record in records)

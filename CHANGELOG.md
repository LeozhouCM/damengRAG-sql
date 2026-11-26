# 变更记录

> 本文件是什么：按时间记录每次数据/规则变更的范围、受影响块和验证结果，供复核和答辩追溯。只记已经发生的变更，不写计划。

依据：实践计划书 §4(六) 要求记录变更日期、范围和受影响块。

## 2026-11-26：1A v2（小块 + 父子块 + 角色）

- 范围：26 篇案例 + 1 篇 DM9 手册样章（系统管理员手册第 22 章“查询优化”）；不包含 embedding/向量库。
- 变更原因：1A v1 单块平均 859 字、最长 15,876 字，检索不精确；组会决定按 DM-RAG 论文改为“小块检索 + 父块补上下文”。
- 切块参数：目标 200 字、上限 300 字、重叠 0；依据论文 128 token 的中文工程换算，不是论文原话。
- 新增元数据：`root_id`、`parent_id`、`chunk_index`、`section_role`、`content_type`。
- 新增规则：
  - 父块 = 标题节，`parent_id=<source_id>-pNNN`；父块不单独入库，运行时按 `parent_id` 还原。
  - 角色只看标题；案例、手册用不同词表；H1 记“概念”；证据用 `content_type=执行计划/统计输出`，不占角色。
  - 长代码按行拆段，每段保留围栏；单行超长保留整行并标 `oversize_line`。
  - 代码里的等宽表格按“表头 + 分隔线 + 数据行”单独成段，整段标 `table_code` 并记为“表格”；Statistics 输出段标 `stat_code`，避免表行被误判成 SQL。
  - 短引导句贴到后面的代码/表格，不再单独成块。
  - 短块只在同父块、同角色、同内容类型内合并。
- 输入：`data/raw/dm_cases/` 26 篇 + `data/raw/dm_docs/DM9系统管理员手册-第22章-查询优化.md`。
- 输出：`data/processed/chunks.jsonl`、`data/processed/stats.json`、`data/processed/spot_check_20.md`；v1 产物已删除，不保留备份。
- 结果：27 个来源（case 26 + doc 1）；1,502 块；240 个父块；平均 211.9 字；p50 223、p90 287；最短 18、最长 594；不足 80 字 39 块（其中 9 个 H1 摘要）；超过 300 字 13 块（单行超长 9、表格行超长 4）。
- 角色覆盖：问题 173、原因 20、办法 311、验证 152、概念 265、步骤 99、参考 1、总结 18、未分类 463。
- 内容类型：文字 278、SQL 465、执行计划 524、统计输出 53、表格 182。
- 重复文本：37 组、48 个多余块，均为原文重复，不是切块器重复。
- 验证：`check_chunks.py` 0 个错误；`pytest` 30 个测试通过。
- 未完成：20 子块 + 10 父块人工勾选；1B 的 embedding、Chroma、BM25/RRF、父块限长展开和评测。

## 2026-11-17：1A 初版

- 范围：26 篇 DM SQL 优化 Markdown 的来源登记、清洗、标题化、切块、标签、校验；不包含 embedding/向量库。
- 输入：`data/raw/dm_cases/` 下 26 篇原文；原始资料按项目理解 表12 不进入 Git。
- 新增：`config/settings.yaml`、`config/tags.yaml`、`src/knowledge/ingest.py`、`scripts/build_chunks.py`、`scripts/check_chunks.py`、`tests/test_ingest.py`。
- 标题化：13 个纯编号标题按下一行粗体标签补齐；无正文的纯编号分组标题不进入 `section_path`。
- 输出：`data/registry/sources.json`、`data/interim/cleaned/*.md`、`data/processed/chunks.jsonl`、`data/processed/stats.json`、`data/processed/spot_check_20.md`。
- 结果：296 块；平均 859.0 字；最短 27 字（H1 摘要），最长 15,876 字（完整执行计划代码块）；重复块 0；主题覆盖为执行计划 226、索引 131、统计信息 17、SQL写法 233。
- 质量标记：94 块，其中 `oversize_code` 51、`code_context_repeated` 38、`merged_short` 5。
- 验证：`check_chunks.py` 0 个错误；`pytest` 18 个测试通过。
- 未完成：20 块人工勾选；1B 的 embedding、Chroma、BM25/RRF 和评测。

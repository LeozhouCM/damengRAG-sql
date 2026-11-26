"""1A 知识库清洗、切块与标签（本文件是 1A 的数据处理核心）。

本文件做什么：
- 识别 H1 是摘要、推广、纯正文还是带代码内容；
- 清洗 Markdown，把缩进代码转成三个反引号围栏，并做标题化；
- 按文章标题结构切出父块，再切 200/300 字子块，处理长正文、长表格和长代码；
- 给每个子块标 parent_id/chunk_index/section_role/content_type；
- 按固定词典打主题/操作符/参数/SQL 特征标签；
- 输出带来源、章节、父子块、角色、版本、标签的 chunk 记录和统计信息。

调用链：
build_chunks.py -> chunks_from_cleaned()
                -> parse_sections() -> pack_section() -> merge_short_chunks()
                -> tag_chunk()
check_chunks.py 只读取本文件产出的 chunks.jsonl，不反向修改数据。

依据：
- DM-RAG 论文 §V：小块检索 + 父块补上下文；检索块 128 token；重叠 0 最好。
- 中文工程换算：目标 200 字、上限 300 字；字符数不是论文原话。
- 实践计划书 §4(二)：清洗与标题化；代码块用三个反引号；表格保留表头。
- 实践计划书 §4(一)：四类主题；§4(五)：构建流程与统计；§4(六)：人工抽看。
- 项目计划书 §4：chunk 元数据包含来源、章节、主题、SQL 特征/操作符。
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```")
PLAN_LINE_RE = re.compile(r"^\s*(?:\d+\s+)?#\s*[A-Z]")
SQL_START_RE = re.compile(
    r"^\s*(select|with|update|insert|delete|create|alter|drop|call|explain|merge|truncate|comment|grant|revoke)\b",
    re.I,
)
STAT_START_RE = re.compile(
    r"^\s*(Statistics|data pages changed|undo pages changed|logical reads|physical reads|"
    r"redo size|rows processed|exec time\(ms\)|已用时间|执行号)",
    re.I,
)
BOLD_LABEL_RE = re.compile(r"^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*$")
URL_RE = re.compile(r"https?://\S+")
HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
ASCII_TABLE_SEPARATOR_RE = re.compile(r"^\s*-{3,}(?:\s+-{3,}){2,}\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")

TOPIC_ORDER = ("执行计划", "索引", "统计信息", "SQL写法")
ROLE_ORDER = ("问题", "原因", "办法", "验证", "概念", "步骤", "参考", "总结", "未分类")
CONTENT_TYPE_ORDER = ("执行计划", "统计输出", "SQL", "表格", "命令", "文字")
COMMAND_RE = re.compile(r"^\s*(?:\$|\./|disql\b|sp_[A-Za-z0-9_]+\b|monitor\b)", re.I)


@dataclass
class Source:
    """来源登记信息，对应 data/registry/sources.json 的一条记录。"""

    source_id: str
    source_type: str
    source_name: str
    source_file: str
    source_url: str
    version: str
    acquired_at: str
    intro_type: str
    copyright_note: str = "公开技术文章，仅本地学习使用；原文不进入 Git"


@dataclass
class Section:
    """标题树中的一节：level/title/path/body，body 只包含本级标题自己的正文。"""

    level: int
    title: str
    path: str
    body: str
    is_h1: bool = False
    root_id: str = ""
    parent_id: str = ""
    section_role: str = "未分类"


@dataclass
class Unit:
    """切块前的最小单元：正文段落、列表、代码块或表格。"""

    kind: str  # text / list / code / table
    text: str


@dataclass
class Chunk:
    """内部 chunk；flags/h1_abstract 只用于 QC，不写入 chunks.jsonl。"""

    text: str
    title: str
    section_path: str
    flags: set[str] = field(default_factory=set)
    h1_abstract: bool = False
    root_id: str = ""
    parent_id: str = ""
    section_role: str = "未分类"
    content_type: str = "文字"
    chunk_index: int = 0


def classify_h1_intro(body: str) -> str:
    """把 H1 下的正文分成 none / boilerplate / abstract / content。

    - none：H1 后直接是 H2，没有正文。
    - boilerplate：只有“ps/推荐文章/公众号”等推广链接，没有案例内容。
    - content：含代码块或表格，按普通正文切块。
    - abstract：真正的摘要/概述，短于 80 字也保留独立块。
    """
    stripped = body.strip()
    if not stripped:
        return "none"

    lines = body.splitlines()
    has_indented_code = any(line.startswith("    ") and line.strip() for line in lines)
    has_fenced_code = any(line.strip().startswith("```") for line in lines)
    has_table = any(line.lstrip().startswith("|") for line in lines)
    if has_indented_code or has_fenced_code or has_table:
        return "content"

    if URL_RE.search(body):
        without_url = URL_RE.sub("", body)
        without_url = re.sub(r"(?is)ps[：:].*", "", without_url)
        without_url = re.sub(r"[*_`#>\[\]()（）]", "", without_url).strip()
        looks_promotional = bool(
            re.search(r"(?i)ps[：:]|推荐|公众号|关注|往期|相关文章", body)
        )
        if looks_promotional and len(without_url) < 120:
            return "boilerplate"

    return "abstract"


def extract_h1(raw: str) -> tuple[str, str]:
    """返回 (H1 标题, H1 下的正文)。没有 H1 时用第一行非空文本。"""
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(.*)$", line)
        if m:
            title = m.group(1).strip()
            body_lines: list[str] = []
            for nxt in lines[i + 1 :]:
                if re.match(r"^##\s+", nxt):
                    break
                body_lines.append(nxt)
            return title, "\n".join(body_lines).strip()
    first = next((line.strip() for line in lines if line.strip()), "")
    return first.lstrip("# ").strip(), ""


def extract_version(raw: str) -> str:
    """从原文提取 DM8/DM9 年份季度；找不到就写“未注明”，不猜版本。"""
    m = re.search(
        r"DM\s*(8|9)\s*[（(]?\s*(20\d{2})\s*[）)]?\s*Q([1-4])", raw, re.I
    )
    if m:
        return f"DM{m.group(1)} {m.group(2)}Q{m.group(3)}"
    m = re.search(r"DM\s*(8|9)\b", raw, re.I)
    if m:
        return f"DM{m.group(1)}"
    return "未注明"


def _remove_boilerplate_h1(lines: list[str]) -> tuple[list[str], int]:
    """删除 H1 与第一个 H2 之间的推广正文。返回行和删除字符数。"""
    h1_idx = next((i for i, line in enumerate(lines) if re.match(r"^#\s+", line)), None)
    if h1_idx is None:
        return lines, 0
    h2_idx = next(
        (i for i in range(h1_idx + 1, len(lines)) if re.match(r"^##\s+", lines[i])),
        None,
    )
    if h2_idx is None:
        removed = sum(len(line) + 1 for line in lines[h1_idx + 1 :])
        return lines[: h1_idx + 1], removed
    removed = sum(len(line) + 1 for line in lines[h1_idx + 1 : h2_idx])
    return lines[: h1_idx + 1] + lines[h2_idx:], removed


def _looks_like_list_or_quote(line: str) -> bool:
    """避免把缩进列表/引用误当成缩进代码。"""
    stripped = line.lstrip()
    return bool(LIST_ITEM_RE.match(stripped) or stripped.startswith(">"))


def classify_code_block(lines: list[str]) -> str:
    """把围栏内容分成 sql / plan / stat / table；表格输出不再被当成 SQL。"""
    if any(PLAN_LINE_RE.match(line) for line in lines):
        return "plan"
    first = next(
        (
            line
            for line in lines
            if line.strip() and not line.lstrip().startswith("--")
        ),
        "",
    )
    if STAT_START_RE.match(first):
        return "stat"
    nonempty = [line for line in lines if line.strip()]
    if nonempty and all(line.lstrip().startswith("|") for line in nonempty):
        return "table"
    # 控制台输出的等宽表格：表头下面是一串“---- ---- ----”分隔线。
    if any(ASCII_TABLE_SEPARATOR_RE.match(line) for line in lines):
        return "table"
    return "sql"


def _iter_fenced_blocks(text: str) -> Iterable[list[str]]:
    """逐段返回围栏代码块（含首尾 ``` 行）；未闭合时返回到文本末尾。"""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not FENCE_RE.match(lines[index]):
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not FENCE_RE.match(lines[end]):
            end += 1
        yield lines[index : end + 1] if end < len(lines) else lines[index:]
        index = end + 1


def detect_content_type(text: str) -> str:
    """判断一个块“包含什么”：执行计划 > 统计输出 > SQL > 表格 > 命令 > 文字。"""
    kinds: set[str] = set()
    for block in _iter_fenced_blocks(text):
        inner = block[1:-1] if len(block) >= 2 and FENCE_RE.match(block[-1]) else block[1:]
        kind = classify_code_block(inner)
        if kind == "plan":
            kinds.add("执行计划")
        elif kind == "stat":
            kinds.add("统计输出")
        elif kind == "table":
            kinds.add("表格")
        else:
            kinds.add("SQL")
        if any(COMMAND_RE.match(line) for line in inner):
            kinds.add("命令")
    for name in CONTENT_TYPE_ORDER:
        if name in kinds:
            return name
    lines = text.splitlines()
    if any(line.lstrip().startswith("|") for line in lines):
        return "表格"
    if any(COMMAND_RE.match(line) for line in lines):
        return "命令"
    return "文字"


def split_code_region(lines: list[str]) -> list[tuple[str, list[str]]]:
    """把一段缩进代码按 SQL / 执行计划分成子块；不按注释切。"""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] = []
    current_kind: str | None = None

    def emit() -> None:
        nonlocal current
        while current and not current[-1].strip():
            current.pop()
        if current:
            blocks.append((classify_code_block(current), current[:]))
        current = []

    for line in lines:
        if not line.strip():
            current.append("")
            continue
        new_kind = "plan" if PLAN_LINE_RE.match(line) else "sql"
        if current_kind is None:
            current_kind = new_kind
        elif new_kind != current_kind:
            emit()
            current_kind = new_kind
        elif current_kind == "sql" and current and not current[-1].strip():
            previous = next((item for item in reversed(current) if item.strip()), "")
            if previous.rstrip().endswith(";") and SQL_START_RE.match(line):
                emit()
                current_kind = "sql"
        current.append(line)
    emit()
    return blocks


def convert_indented_code(lines: list[str]) -> tuple[list[str], list[str]]:
    """把 4 空格缩进代码转换成三个反引号围栏；SQL 与计划分开。"""
    out: list[str] = []
    warnings: list[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue
        is_indented = line.startswith("    ") and bool(line.strip())
        if is_indented and _looks_like_list_or_quote(line):
            out.append(line)
            i += 1
            continue
        if not is_indented:
            out.append(line)
            i += 1
            continue

        region: list[str] = []
        j = i
        while j < len(lines):
            candidate = lines[j]
            if candidate.startswith("    ") and candidate.strip():
                region.append(candidate[4:])
                j += 1
            elif not candidate.strip():
                region.append("")
                j += 1
            else:
                break
        while region and not region[-1].strip():
            region.pop()

        blocks = split_code_region(region)
        if not blocks:
            warnings.append("检测到缩进代码但内容为空")
        for kind, block_lines in blocks:
            block_text = "\n".join(block_lines).strip("\n")
            if not block_text.strip():
                continue
            lang = "sql" if kind == "sql" else "text"
            out.append(f"```{lang}")
            out.extend(block_lines)
            out.append("```")
            out.append("")
        i = j
    return out, warnings


def titleize_numeric_headings(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """给“## 01”这类纯编号标题补上下一行粗体标签。"""
    out = lines[:]
    warnings: list[str] = []
    changed: list[str] = []
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^(#{2,6})\s+(\d+)\s*$", line)
        if not m:
            continue
        next_idx = i + 1
        while next_idx < len(lines) and not lines[next_idx].strip():
            next_idx += 1
        if next_idx >= len(lines):
            warnings.append(f"纯编号标题无法补标题：{line.strip()}")
            continue
        if re.match(r"^#{2,6}\s+", lines[next_idx]):
            # 纯编号分组标题，下面直接是子标题；parse_sections 不把它放进 section_path。
            continue
        label = BOLD_LABEL_RE.match(lines[next_idx])
        if label:
            new_title = f"{m.group(1)} {m.group(2)} {label.group(1).strip()}"
            out[i] = new_title
            changed.append(f"{line.strip()} -> {new_title.strip()}")
        else:
            warnings.append(f"纯编号标题下一行不是粗体标签：{line.strip()}")
    return out, warnings, changed


def remove_horizontal_rules(lines: list[str]) -> list[str]:
    """删除 Markdown 水平线；代码围栏内和表格分隔行（带 |）不动。"""
    out: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and HORIZONTAL_RULE_RE.match(line):
            continue
        out.append(line)
    return out


def collapse_blank_lines(lines: list[str]) -> list[str]:
    """压缩连续空行，最多保留两行，避免清洗结果出现大片空白。"""
    out: list[str] = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            out.append(line.rstrip())
        else:
            blank += 1
            if blank <= 2:
                out.append("")
    while out and not out[-1].strip():
        out.pop()
    return out


def clean_markdown(raw: str, intro_type: str) -> tuple[str, list[str]]:
    """清洗一篇原始 Markdown，返回 (清洗后文本, 告警列表)。

    顺序很重要：先删推广正文，再转代码围栏，最后标题化和压缩空行；
    代码转围栏后，标题化不会误改代码里的 # 行。
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    warnings: list[str] = []

    # [STEP 1] 只对“纯推广”H1 删除 H1 与第一个 H2 之间的链接/ps 文案。
    if intro_type == "boilerplate":
        lines, removed = _remove_boilerplate_h1(lines)
        if removed:
            warnings.append(f"删除 H1 推广正文 {removed} 字符")

    # [STEP 2] 缩进代码 -> 三个反引号；SQL/执行计划/统计输出分别标语言。
    lines, code_warnings = convert_indented_code(lines)
    warnings.extend(code_warnings)

    # [STEP 3] “## 01 + 下一行粗体标签”补成可读标题；纯编号分组标题不改原文。
    lines, title_warnings, changed = titleize_numeric_headings(lines)
    warnings.extend(title_warnings)
    for item in changed:
        warnings.append(f"标题化：{item}")

    # [STEP 4] 去水平线、压缩空行。
    lines = remove_horizontal_rules(lines)
    lines = collapse_blank_lines(lines)
    return "\n".join(lines).strip() + "\n", warnings


def parse_sections(cleaned: str) -> list[Section]:
    """按 Markdown 标题树切出 Section；自动跳过代码围栏里的“#”行。"""
    lines = cleaned.splitlines()
    # [STEP 1] 只收集围栏外的标题；代码里的“#”注释不算标题。
    headings: list[tuple[int, int, str]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    # [STEP 2] 每个标题的正文 = 到下一个标题之前的行；父标题只拿自己的正文。
    rows: list[tuple[int, int, str, str]] = []
    for idx, (line_no, level, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[line_no + 1 : end]).strip()
        rows.append((level, line_no, title, body))

    # [STEP 3] 用栈维护标题层级；没有正文的标题只当路径的一部分，不单独产块。
    sections: list[Section] = []
    stack: list[tuple[int, str, bool]] = []
    for level, line_no, title, body in rows:
        while stack and stack[-1][0] >= level:
            stack.pop()
        # 纯编号且无正文的标题只是分组标记，不作为 section_path 的语义层级。
        path_titles = [
            item[1]
            for item in stack
            if not (re.fullmatch(r"\d+", item[1]) and not item[2])
        ] + [title]
        path = " > ".join(path_titles)
        stack.append((level, title, bool(body.strip())))
        if body:
            sections.append(
                Section(level=level, title=title, path=path, body=body, is_h1=(level == 1))
            )
    return sections


def classify_section_role(
    title: str,
    tags_cfg: dict[str, Any],
    source_type: str = "case",
    is_h1: bool = False,
) -> str:
    """按标题给一节定角色；手册和案例用不同的词表，标题看不出就返回未分类/概念。

    - 案例：问题/原因/办法/验证/概念/步骤/总结；
    - 手册：步骤/参考/总结，其余默认概念；
    - H1 是文档标题，不参与关键词匹配，统一按概念处理。
    """
    if is_h1:
        return "概念"
    mapping_key = "manual_section_roles" if source_type == "doc" else "section_roles"
    for role, pattern in tags_cfg.get(mapping_key, {}).items():
        try:
            if re.search(pattern, title, re.I):
                return role
        except re.error:
            continue
    return "概念" if source_type == "doc" else "未分类"


def split_units(body: str) -> list[Unit]:
    """把一节正文拆成最小单元：代码块、表格、列表或普通段落。"""
    lines = body.splitlines()
    units: list[Unit] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if FENCE_RE.match(line):
            j = i + 1
            while j < len(lines) and not FENCE_RE.match(lines[j]):
                j += 1
            if j >= len(lines):
                units.append(Unit("code", "\n".join(lines[i:])))
                break
            units.append(Unit("code", "\n".join(lines[i : j + 1])))
            i = j + 1
            continue

        if line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            units.append(Unit("table", "\n".join(lines[i:j])))
            i = j
            continue

        if LIST_ITEM_RE.match(line):
            j = i
            while j < len(lines):
                if not lines[j].strip():
                    break
                if LIST_ITEM_RE.match(lines[j]) or lines[j].startswith("  "):
                    j += 1
                else:
                    break
            units.append(Unit("list", "\n".join(lines[i:j])))
            i = j
            continue

        j = i
        while j < len(lines):
            candidate = lines[j]
            if not candidate.strip():
                break
            if FENCE_RE.match(candidate) or candidate.lstrip().startswith("|"):
                break
            if LIST_ITEM_RE.match(candidate):
                break
            j += 1
        units.append(Unit("text", "\n".join(lines[i:j])))
        i = j
    return units


def _join_text(left: str, right: str) -> str:
    """用两个换行拼接两段文本；任一方为空时直接返回另一方。"""
    if not left:
        return right
    if not right:
        return left
    return left.rstrip() + "\n\n" + right.lstrip()


def hard_split(text: str, max_chars: int) -> list[str]:
    """兜底按字符硬切；只用于连句号都没有的超长单句。"""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def split_long_text(text: str, target: int, max_chars: int, overlap: int) -> list[str]:
    """长正文按自然句切分：目标 target 字，单块不超过 max_chars，相邻重叠 overlap 字。"""
    if len(text) <= max_chars:
        return [text]
    parts = [part for part in re.split(r"(?<=[。！？；.!?])\s*", text) if part]
    if not parts:
        parts = [text]

    pieces: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(hard_split(part, max_chars))
            continue
        candidate = current + part
        if len(candidate) > target and current:
            pieces.append(current)
            current = part
        else:
            current = candidate
    if current:
        pieces.append(current)

    if overlap > 0 and len(pieces) > 1:
        pieces = [pieces[0]] + [pieces[i - 1][-overlap:] + pieces[i] for i in range(1, len(pieces))]
    return pieces


def _group_code_lines(
    lines: list[str],
    budget: int,
    min_chars: int,
    max_chars: int,
    frame_len: int,
) -> tuple[list[list[str]], set[str]]:
    """把一段代码按行贪心分组，并处理最后一段过短的问题。

    max_chars 是这一段每块允许的字符上限（已经把围栏和引导句算进去），
    所以挪行/并回都不能超过它。
    """
    flags: set[str] = set()
    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if line_len > budget:
            if current:
                groups.append(current)
                current = []
                current_len = 0
            groups.append([line])
            flags.add("oversize_line")
            continue
        if current and current_len + line_len > budget:
            groups.append(current)
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        groups.append(current)

    def wrapped_len(group: list[str]) -> int:
        return frame_len + sum(len(line) + 1 for line in group)

    # 最后一段太短时，从上一段末尾挪几行过来，让最后一块也够读；
    # 只在长度不超过 max_chars 时挪，不制造超长块。
    while len(groups) > 1 and wrapped_len(groups[-1]) < min_chars:
        previous = groups[-2]
        if len(previous) <= 1:
            break
        line = previous[-1]
        if wrapped_len(groups[-1] + [line]) > max_chars:
            break
        previous.pop()
        groups[-1].insert(0, line)

    # 挪行后仍然太短就并回上一段；并回后仍不能超过 max_chars。
    if len(groups) > 1 and wrapped_len(groups[-1]) < min_chars:
        if wrapped_len(groups[-2] + groups[-1]) <= max_chars:
            groups[-2].extend(groups[-1])
            groups.pop()
    return groups, flags


def split_code_unit(
    text: str, max_chars: int, min_chars: int, prefix: str = ""
) -> list[tuple[str, set[str]]]:
    """把超长代码按行分组，每组重新包上 ``` 围栏；返回 (块文本, 块标记)。

    - prefix 是紧挨代码前的引导句，只放进第一块并提前预留长度；
    - 代码里的等宽表格按“表头 + 分隔线 + 数据行”单独成段，整段标 table_code；
    - Statistics 输出整段标 stat_code，避免后面的行被误判成 SQL；
    - 单行超长时保留整行并标 oversize_line，不把一行从中间切断。
    """
    prefix_len = len(prefix) + 2 if prefix else 0
    if len(text) + prefix_len <= max_chars:
        return [(_join_text(prefix, text) if prefix else text, set())]

    lines = text.splitlines()
    has_fence = bool(lines and FENCE_RE.match(lines[0]))
    opening = lines[0] if has_fence else ""
    closing = "```" if has_fence else ""
    if has_fence:
        end = next(
            (index for index in range(1, len(lines)) if FENCE_RE.match(lines[index])),
            len(lines),
        )
        inner = lines[1:end]
    else:
        inner = lines

    frame_len = len(opening) + len(closing) + 2

    # 找到等宽表格的起点：分隔线的上一行是表头，后面是数据行。
    table_start: int | None = None
    for index, line in enumerate(inner):
        if ASCII_TABLE_SEPARATOR_RE.match(line):
            table_start = max(0, index - 1)
            break
    if table_start is None:
        segments: list[tuple[list[str], bool]] = [(inner, False)]
    else:
        segments = []
        if table_start > 0:
            segments.append((inner[:table_start], False))
        segments.append((inner[table_start:], True))

    result: list[tuple[str, set[str]]] = []
    for segment_index, (segment_lines, is_table) in enumerate(segments):
        extra = prefix_len if (segment_index == 0 and prefix) else 0
        budget = max(max_chars - frame_len - extra, 1)
        groups, flags = _group_code_lines(
            segment_lines, budget, min_chars, max_chars - extra, frame_len
        )
        if is_table:
            flags.add("table_code")
        elif classify_code_block(segment_lines) == "stat":
            flags.add("stat_code")
        for group in groups:
            if has_fence:
                piece = opening + "\n" + "\n".join(group) + "\n" + closing
            else:
                piece = "\n".join(group)
            if not result and prefix:
                piece = _join_text(prefix, piece)
            result.append((piece, set(flags)))

    if len(result) > 1:
        for _, piece_flags in result:
            piece_flags.add("code_split")
    return result


def split_table(text: str, target: int, max_chars: int) -> tuple[list[str], set[str]]:
    """长表格按行拆块，每块重复表头；引导句和表标题只放在第一块。

    单行超长时保留整行并打标记，不把一行表格从中间切断。
    """
    lines = text.splitlines()
    flags: set[str] = set()
    start = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("|")),
        None,
    )
    if start is None or len(lines) < start + 2:
        flags.add("table_without_header")
        return [text], flags
    if not TABLE_SEPARATOR_RE.match(lines[start + 1]):
        flags.add("table_without_header")
        return [text], flags

    prefix_lines = lines[:start]
    table_header = lines[start : start + 2]
    rows = lines[start + 2 :]
    if not rows:
        return [text], flags

    def render(piece_rows: list[str], first: bool) -> str:
        """第一块带引导句/表标题，后续块只重复表格表头。"""
        head = prefix_lines + table_header if first else table_header
        return "\n".join(head + piece_rows)

    pieces: list[str] = []
    current_rows: list[str] = []
    full_header_len = len("\n".join(prefix_lines + table_header))
    bare_header_len = len("\n".join(table_header))
    current_len = full_header_len
    for row in rows:
        if len(row) > max_chars:
            if current_rows:
                pieces.append(render(current_rows, not pieces))
                current_rows = []
            pieces.append(render([row], not pieces))
            current_len = bare_header_len
            flags.add("table_row_oversize")
            continue
        candidate_len = current_len + len(row) + 1
        if candidate_len > target and current_rows:
            pieces.append(render(current_rows, not pieces))
            current_rows = [row]
            current_len = bare_header_len + len(row) + 1
        else:
            current_rows.append(row)
            current_len = candidate_len
    if current_rows:
        pieces.append(render(current_rows, not pieces))
    return pieces, flags


def _is_label_text(text: str) -> bool:
    """“SQL：”“执行计划：”这类短标签不作为代码块的前文解释。"""
    stripped = text.strip()
    return len(stripped) <= 20 and bool(re.search(r"[：:]\s*$", stripped))


def _context_for_code(code: str, last_text: str, max_chars: int) -> str:
    """代码另起一块时，只重复放得下的前文，避免把块撑得更大。"""
    if not last_text:
        return ""
    budget = max_chars - len(code) - 2
    if budget <= 0:
        return ""
    text = last_text.strip()
    if len(text) <= budget:
        return text
    parts = [part for part in re.split(r"(?<=[。！？；.!?])\s*", text) if part]
    context = ""
    for part in parts:
        candidate = _join_text(context, part)
        if len(candidate) > budget:
            break
        context = candidate
    if context:
        return context
    first = parts[0] if parts else text
    return first[:budget].rstrip()


def pack_section(
    section: Section, target: int, max_chars: int, overlap: int, min_chars: int = 80
) -> list[Chunk]:
    """把一节的正文单元打包成块。

    规则顺序：
    1. 代码块短则跟正文同块；超过 max_chars 按行拆成多段，每段保留围栏；
       单行超长保留整行并标 oversize_line。
    2. 表格超过 max_chars 才按行拆，并重复表头。
    3. 正文超过 max_chars 才按句子拆；生产参数下不重叠。
    """
    units = split_units(section.body)
    # current 是正在攒的块；last_text 记住上一段正文，供后续代码块重复上下文。
    chunks: list[Chunk] = []
    current = ""
    current_flags: set[str] = set()
    last_text = ""
    pending = ""  # 暂存“构造数据”“执行计划：”这类短引导句，贴到下一个单元上
    last_index = len(units) - 1

    def make_chunk(text: str, flags: set[str]) -> Chunk:
        """统一填写父块、角色和内容类型，避免每个分支漏字段。"""
        stripped = text.strip()
        return Chunk(
            text=stripped,
            title=section.title,
            section_path=section.path,
            flags=set(flags),
            root_id=section.root_id,
            parent_id=section.parent_id,
            section_role=section.section_role,
            content_type=(
                "表格" if "table_code" in flags else detect_content_type(stripped)
            ),
        )

    def emit() -> None:
        """把 current 收尾成一个块；current 为空时不产块。"""
        nonlocal current, current_flags
        text = current.strip()
        if text:
            chunks.append(make_chunk(text, current_flags))
        current = ""
        current_flags = set()

    for index, unit in enumerate(units):
        # 短正文/短列表且后面还有内容：先攒着，避免“构造数据”四个字单独成块。
        if (
            unit.kind in {"text", "list"}
            and index < last_index
            and len(unit.text) < min_chars
        ):
            pending = _join_text(pending, unit.text)
            continue

        # 分支 1：代码块。短代码可与正文同块；超长代码拆段，引导句只跟第一段。
        if unit.kind == "code":
            prefix = pending
            pending = ""
            if prefix:
                combined = _join_text(prefix, unit.text)
                if len(combined) <= max_chars:
                    candidate = _join_text(current, combined)
                    if len(candidate) > max_chars:
                        emit()
                        current = combined
                    else:
                        current = candidate
                    continue
                emit()
                pieces = split_code_unit(
                    unit.text, max_chars, min_chars, prefix=prefix
                )
                for piece, piece_flags in pieces:
                    chunks.append(make_chunk(piece, piece_flags))
                continue
            if len(unit.text) > max_chars:
                emit()
                pieces = split_code_unit(unit.text, max_chars, min_chars)
                for piece, piece_flags in pieces:
                    chunks.append(make_chunk(piece, piece_flags))
                continue
            candidate = _join_text(current, unit.text)
            if len(candidate) > max_chars or (not current and last_text):
                emit()
                context = _context_for_code(unit.text, last_text, max_chars)
                with_context = _join_text(context, unit.text) if context else unit.text
                if len(with_context) <= max_chars:
                    current = with_context
                    current_flags = {"code_context_repeated"} if context else set()
                else:
                    current = unit.text
                    current_flags = set()
            else:
                current = candidate
            continue

        # 表格/正文先把暂存的引导句贴上，再判断长度。
        unit_text = _join_text(pending, unit.text) if pending else unit.text
        pending = ""

        # 分支 2：表格只有超过 max_chars 才拆，拆的时候每块重复表头和引导句。
        if unit.kind == "table":
            if len(unit_text) > max_chars:
                emit()
                pieces, flags = split_table(unit_text, target, max_chars)
                for piece in pieces:
                    chunks.append(make_chunk(piece, flags | {"table_split"}))
                continue
            candidate = _join_text(current, unit_text)
            if len(candidate) > max_chars:
                emit()
                current = unit_text
            else:
                current = candidate
            continue

        # 分支 3：正文/列表；超长才按句子切，否则继续往 current 里攒。
        if not _is_label_text(unit_text):
            last_text = unit_text
        if len(unit_text) > max_chars:
            emit()
            for piece in split_long_text(unit_text, target, max_chars, overlap):
                chunks.append(make_chunk(piece, {"text_split"}))
            continue
        candidate = _join_text(current, unit_text)
        if len(candidate) > max_chars:
            emit()
            current = unit_text
        else:
            current = candidate

    # 节末尾还攒着引导句：能并进当前块就并，并不了就保留成短块交给校验。
    if pending:
        candidate = _join_text(current, pending)
        if len(candidate) <= max_chars:
            current = candidate
        else:
            emit()
            current = pending
    emit()
    return chunks


def merge_short_chunks(
    chunks: list[Chunk], min_chars: int, max_chars: int
) -> tuple[list[Chunk], list[str]]:
    """不足 min_chars 的块只在同父块、同角色、同内容类型内合并；H1 摘要除外。

    不跨节合并，避免把“问题”并进“办法”；也不把代码碎片并进正文。
    找不到合适邻居的短块保留，并打 short_chunk，交给校验和人工抽看。
    """
    warnings: list[str] = []

    def can_merge(left: Chunk, right: Chunk) -> bool:
        if left.h1_abstract or right.h1_abstract:
            return False
        if not (
            left.parent_id == right.parent_id
            and left.section_role == right.section_role
            and left.content_type == right.content_type
        ):
            return False
        return len(_join_text(left.text, right.text)) <= max_chars

    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            len(chunk.text) < min_chars
            and merged
            and can_merge(merged[-1], chunk)
        ):
            merged[-1].text = _join_text(merged[-1].text, chunk.text)
            merged[-1].flags.update(chunk.flags)
            merged[-1].flags.add("merged_short")
            merged[-1].flags.discard("short_chunk")
            warnings.append(
                f"短块并入上一块：{chunk.section_path} ({len(chunk.text)} 字)"
            )
            continue
        merged.append(chunk)

    index = 0
    while index < len(merged):
        chunk = merged[index]
        if (
            len(chunk.text) < min_chars
            and index + 1 < len(merged)
            and can_merge(chunk, merged[index + 1])
        ):
            nxt = merged[index + 1]
            nxt.text = _join_text(chunk.text, nxt.text)
            nxt.flags.update(chunk.flags)
            nxt.flags.add("merged_short")
            nxt.flags.discard("short_chunk")
            warnings.append(
                f"短块并入下一块：{chunk.section_path} ({len(chunk.text)} 字)"
            )
            merged.pop(index)
            continue
        index += 1

    for chunk in merged:
        if len(chunk.text) < min_chars and not chunk.h1_abstract:
            chunk.flags.add("short_chunk")
            warnings.append(
                f"短块保留（无同父同角色邻居）：{chunk.section_path} ({len(chunk.text)} 字)"
            )
    return merged, warnings


def _keyword_pattern(keyword: str) -> re.Pattern[str] | None:
    """中文按子串；英文按宽松词边界，允许操作符后的数字。"""
    if re.search(r"[一-鿿]", keyword):
        return None
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")
    return re.compile(rf"(?i)(?<![A-Za-z_]){escaped}(?![A-Za-z_])")


def keyword_in_text(keyword: str, text: str) -> bool:
    """按 config/tags.yaml 的词典判断关键词是否出现；不做同义词扩展。"""
    pattern = _keyword_pattern(keyword)
    if pattern is None:
        return keyword in text
    return bool(pattern.search(text))


def load_tags(path: Path) -> dict[str, Any]:
    """读取 config/tags.yaml：主题、操作符、参数、SQL 特征四类词典。"""
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def tag_chunk(text: str, tags_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """返回 (tags, warnings)。注释可以参与关键词匹配，但不是切块边界。"""
    warnings: list[str] = []
    # [STEP 1] 主题：词典命中即打标签，允许一块多主题。
    topic_tags: list[str] = []
    for topic in TOPIC_ORDER:
        keywords = tags_cfg.get("topics", {}).get(topic, [])
        if any(keyword_in_text(keyword, text) for keyword in keywords):
            topic_tags.append(topic)

    # [STEP 2] 操作符：只认执行计划行里 "#OP" 开头且已登记的名字。
    operator_set = set(tags_cfg.get("operators", []))
    operator_tags: list[str] = []
    for match in re.finditer(r"#\s*([A-Z][A-Z0-9_ ]*?)(?=[:：\[\(;]|$)", text, re.M):
        name = " ".join(match.group(1).split())
        if name in operator_set and name not in operator_tags:
            operator_tags.append(name)

    # [STEP 3] 参数：按完整词匹配，避免把参数名的一部分误当参数。
    parameter_tags: list[str] = []
    for parameter in tags_cfg.get("parameters", []):
        if re.search(rf"(?<![A-Z0-9_]){re.escape(parameter)}(?![A-Z0-9_])", text):
            parameter_tags.append(parameter)

    # [STEP 4] SQL 特征：只跑 tags.yaml 里写好的显式正则。
    feature_tags: list[str] = []
    for feature, pattern in tags_cfg.get("sql_features", {}).items():
        try:
            if re.search(pattern, text, re.M):
                feature_tags.append(feature)
        except re.error as exc:
            warnings.append(f"SQL 特征正则错误 {feature}: {exc}")

    if operator_tags or parameter_tags:
        if "执行计划" not in topic_tags:
            topic_tags.insert(0, "执行计划")

    # 只按显式信号补主题：SQL 代码块或已命中的 SQL 特征。
    if ("```sql" in text or feature_tags) and "SQL写法" not in topic_tags:
        topic_tags.append("SQL写法")

    if not topic_tags:
        warnings.append("没有命中四类主题")

    # [STEP 5] 固定顺序输出并去重：主题 -> 操作符 -> 参数 -> SQL 特征。
    ordered: list[str] = []
    for tag in topic_tags + operator_tags + parameter_tags + feature_tags:
        if tag not in ordered:
            ordered.append(tag)
    return ordered, warnings


def chunks_from_cleaned(
    cleaned: str,
    source: Source,
    settings: dict[str, Any],
    tags_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """单篇文档的主流程：标题树 -> 打包 -> 合并短块 -> 打标签 -> 输出记录。

    返回的 warnings 只用于 data/logs/build_warnings.md，不改变块内容。
    """
    warnings: list[str] = []
    # [STEP 1] 标题树 -> 每节一个父块；父块角色只看标题，手册和案例用不同词表。
    sections = parse_sections(cleaned)
    raw_chunks: list[Chunk] = []
    for parent_index, section in enumerate(sections, start=1):
        section.root_id = source.source_id
        section.parent_id = f"{source.source_id}-p{parent_index:03d}"
        section.section_role = classify_section_role(
            section.title,
            tags_cfg,
            source_type=source.source_type,
            is_h1=section.is_h1,
        )
        section_chunks = pack_section(
            section,
            target=int(settings["chunk_target_chars"]),
            max_chars=int(settings["chunk_max_chars"]),
            overlap=int(settings["chunk_overlap_chars"]),
            min_chars=int(settings["chunk_min_chars"]),
        )
        if section.is_h1 and source.intro_type == "abstract":
            for chunk in section_chunks:
                chunk.h1_abstract = True
        raw_chunks.extend(section_chunks)

    # [STEP 2] 只在同父块、同角色、同内容类型内合并短块。
    raw_chunks, merge_warnings = merge_short_chunks(
        raw_chunks,
        int(settings["chunk_min_chars"]),
        int(settings["chunk_max_chars"]),
    )
    warnings.extend(merge_warnings)

    # [STEP 3] 同一父块内从 1 编号；合并后重新判断内容类型。
    parent_counts: defaultdict[str, int] = defaultdict(int)
    for chunk in raw_chunks:
        parent_counts[chunk.parent_id] += 1
        chunk.chunk_index = parent_counts[chunk.parent_id]
        chunk.content_type = (
            "表格" if "table_code" in chunk.flags else detect_content_type(chunk.text)
        )

    # [STEP 4] 逐块打标签，组装成对外记录的固定字段。
    records: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw_chunks, start=1):
        tags, tag_warnings = tag_chunk(chunk.text, tags_cfg)
        warnings.extend(f"{source.source_id}-c{index:03d}: {item}" for item in tag_warnings)
        records.append(
            {
                "chunk_id": f"{source.source_id}-c{index:03d}",
                "root_id": source.source_id,
                "parent_id": chunk.parent_id,
                "chunk_index": chunk.chunk_index,
                "source_type": source.source_type,
                "source_name": source.source_name,
                "source_url": source.source_url,
                "section_path": chunk.section_path,
                "title": chunk.title,
                "section_role": chunk.section_role,
                "content_type": chunk.content_type,
                "tags": tags,
                "version": source.version,
                "text": chunk.text,
                "_flags": sorted(chunk.flags),
            }
        )
    return records, warnings


def make_stats(
    records: list[dict[str, Any]],
    sources: list[Source],
    warnings_by_source: dict[str, list[str]],
    min_chars: int = 80,
    max_chars: int = 300,
) -> dict[str, Any]:
    """汇总总块数、父块、角色、内容类型、长度、重复和告警，供 1A 验收。"""
    lengths = [len(record["text"]) for record in records]
    sorted_lengths = sorted(lengths)

    def percentile(values: list[int], p: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
        return values[index]

    text_counts = Counter(record["text"] for record in records)
    duplicate_chunks = sum(count - 1 for count in text_counts.values() if count > 1)

    topic_coverage = {
        topic: sum(1 for record in records if topic in record["tags"]) for topic in TOPIC_ORDER
    }
    tag_counts: Counter[str] = Counter()
    for record in records:
        tag_counts.update(record["tags"])

    source_type_counts = Counter(record["source_type"] for record in records)
    per_source = Counter(record["source_name"] for record in records)
    parent_counts = Counter(record["parent_id"] for record in records)
    parent_char_lengths: defaultdict[str, int] = defaultdict(int)
    for record in records:
        parent_char_lengths[record["parent_id"]] += len(record["text"])
    role_coverage = {
        role: sum(1 for record in records if record["section_role"] == role)
        for role in ROLE_ORDER
    }
    content_type_counts = Counter(record["content_type"] for record in records)

    flags: list[dict[str, Any]] = []
    for record in records:
        if record.get("_flags"):
            flags.append(
                {
                    "chunk_id": record["chunk_id"],
                    "flags": record["_flags"],
                    "length": len(record["text"]),
                }
            )

    return {
        "generated_at": date.today().isoformat(),
        "total_chunks": len(records),
        "source_type_distribution": dict(source_type_counts),
        "topic_coverage": topic_coverage,
        "role_coverage": role_coverage,
        "role_unclear_chunks": role_coverage.get("未分类", 0),
        "content_type_distribution": dict(content_type_counts),
        "parent_count": len(parent_counts),
        "avg_chunks_per_parent": round(len(records) / len(parent_counts), 2) if parent_counts else 0,
        "max_chunks_per_parent": max(parent_counts.values()) if parent_counts else 0,
        "parents_over_1500_chars": sum(
            1 for length in parent_char_lengths.values() if length > 1500
        ),
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "p50_chunk_chars": percentile(sorted_lengths, 0.50),
        "p90_chunk_chars": percentile(sorted_lengths, 0.90),
        "under_min_chunks": sum(1 for length in lengths if length < min_chars),
        "over_max_chunks": sum(1 for length in lengths if length > max_chars),
        "short_chunk_count": sum(
            1 for record in records if "short_chunk" in record.get("_flags", [])
        ),
        "code_split_chunks": sum(
            1 for record in records if "code_split" in record.get("_flags", [])
        ),
        "duplicate_chunks": duplicate_chunks,
        "tag_counts": dict(tag_counts.most_common()),
        "per_source_chunks": dict(per_source.most_common()),
        "flagged_chunks": flags,
        "parse_warnings": [
            {"source_id": source.source_id, "source_file": source.source_file, "warnings": warnings_by_source.get(source.source_id, [])}
            for source in sources
            if warnings_by_source.get(source.source_id)
        ],
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """写 JSONL；以下划线开头的内部字段（如 _flags）不写入 chunks.jsonl。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            public = {key: value for key, value in record.items() if not key.startswith("_")}
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    """写 UTF-8 JSON，保留中文，固定缩进，便于 Git 查看差异。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

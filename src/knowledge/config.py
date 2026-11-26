# 本文件做什么：1B 三个入口脚本共用的配置读取与相对路径解析。
# 被谁使用：scripts/build_index.py、scripts/search.py、scripts/eval_retrieval.py、
#            src/knowledge/retrieval.py（加载 settings.yaml 里的 retrieval 段）。
# 依据：实践计划书 §3(四) 要求路径和参数从 config/settings.yaml 读取，不写死在代码里。
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_settings(path: Path) -> dict[str, Any]:
    """读取 YAML 配置；文件不存在时直接报错，避免用默认值掩盖配置错误。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射：{path}")
    return data


def resolve_path(root: Path, value: str | Path) -> Path:
    """把配置里的相对路径转成以项目根目录为基准的绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else root / path


def retrieval_config(settings: dict[str, Any]) -> dict[str, Any]:
    """取出 1B 的 retrieval 段；缺少该段时明确报错，不静默退回 1A 参数。"""
    config = settings.get("retrieval")
    if not isinstance(config, dict):
        raise ValueError("settings.yaml 缺少 retrieval 段")
    return config


def chroma_relative_path(chroma_dir: Path) -> Path:
    """把 Chroma 目录转成相对当前工作目录的 ASCII 路径。

    本机实测（Chroma 1.5.9 + Windows）：把 `E:\\毕业设计\\...` 这种含中文的绝对路径
    传给 PersistentClient，建库时 HNSW 文件写不出来，换进程重开报
    `Error loading hnsw index`；同一目录改用相对路径后正常。这里不猜底层编码细节，
    只把限制固定下来：脚本必须从项目根目录运行，路径中不能出现中文。
    """
    cwd = Path.cwd().resolve()
    try:
        relative = chroma_dir.resolve().relative_to(cwd)
    except ValueError as exc:
        raise ValueError(
            f"Chroma 目录不在当前工作目录下：{chroma_dir}；请在项目根目录运行脚本"
        ) from exc
    if not relative.as_posix().isascii():
        raise ValueError(
            f"Chroma 相对路径含非 ASCII 字符：{relative}；请从项目根目录运行脚本"
        )
    return relative

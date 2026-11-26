# 本文件做什么：1B 建索引入口。把 1A 的 chunks.jsonl 变成 Chroma 向量库 + BM25 词表 + 索引元数据。
# 被谁使用：人工执行；scripts/search.py / scripts/eval_retrieval.py 只读它产出的 data/index。
# 运行：
#   python scripts/build_index.py            # 首次建索引
#   python scripts/build_index.py --force    # 覆盖旧索引（chunks/tags/模型变化后必须重建）
# 产物（data/index/ 不进 Git）：
#   chroma/                  Chroma 持久化目录（子块向量，父块不入库）
#   bm25_tokens.jsonl        BM25 用的子块 token，索引端/查询端同一份
#   index_meta.json          模型、集合、哈希、块数、角色分布
# 依据：实践计划书 表41（sentence-transformers + Chroma、jieba + rank-bm25）；
#       DM-RAG 论文 §V（BM25 + 向量 + RRF）；bge-base-zh-v1.5 模型卡（768 维、512 token）。
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge.config import (  # noqa: E402
    chroma_relative_path,
    load_settings,
    resolve_path,
    retrieval_config,
)
from src.knowledge.retrieval import (  # noqa: E402
    collect_terms,
    file_sha256,
    read_jsonl,
    tokenize_text,
    write_jsonl,
)

REQUIRED_FIELDS = {
    "chunk_id",
    "root_id",
    "parent_id",
    "chunk_index",
    "source_type",
    "source_name",
    "section_path",
    "title",
    "section_role",
    "content_type",
    "text",
}


def validate_chunks(records: list[dict[str, Any]]) -> dict[str, Any]:
    """建库前做硬校验：字段齐全、ID 唯一、父块只属于一个 root/一个角色、索引连续。"""
    if not records:
        raise SystemExit("chunks.jsonl 为空，无法建索引")
    seen_ids: set[str] = set()
    parent_roots: dict[str, set[str]] = {}
    parent_roles: dict[str, set[str]] = {}
    parent_indexes: dict[str, list[int]] = {}
    for record in records:
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise SystemExit(f"块 {record.get('chunk_id', '?')} 缺少字段：{sorted(missing)}")
        chunk_id = str(record["chunk_id"])
        if chunk_id in seen_ids:
            raise SystemExit(f"chunk_id 重复：{chunk_id}")
        seen_ids.add(chunk_id)
        parent_id = str(record["parent_id"])
        parent_roots.setdefault(parent_id, set()).add(str(record["root_id"]))
        parent_roles.setdefault(parent_id, set()).add(str(record["section_role"]))
        parent_indexes.setdefault(parent_id, []).append(int(record["chunk_index"]))
        if not str(record["text"]).strip():
            raise SystemExit(f"块 {chunk_id} 文本为空")

    for parent_id, roots in parent_roots.items():
        if len(roots) != 1:
            raise SystemExit(f"父块 {parent_id} 属于多个 root：{sorted(roots)}")
    for parent_id, roles in parent_roles.items():
        if len(roles) != 1:
            raise SystemExit(f"父块 {parent_id} 有多个角色：{sorted(roles)}")
    for parent_id, indexes in parent_indexes.items():
        ordered = sorted(indexes)
        if ordered != list(range(1, len(ordered) + 1)):
            raise SystemExit(f"父块 {parent_id} 的 chunk_index 不是从 1 连续编号：{ordered[:10]}")

    return {
        "chunk_count": len(records),
        "parent_count": len(parent_roots),
        "root_count": len({record["root_id"] for record in records}),
        "role_counts": dict(Counter(record["section_role"] for record in records)),
        "content_type_counts": dict(Counter(record["content_type"] for record in records)),
    }


def embedding_dimension(model: Any) -> int:
    """兼容 sentence-transformers 6.3 的新旧方法名，避免 FutureWarning。"""
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = getattr(model, "get_sentence_embedding_dimension")
    return int(getter())


def build_collection(client: Any, name: str) -> Any:
    """创建余弦距离集合；兼容 Chroma 0.x 的 metadata 与 1.x 的 configuration 写法。"""
    try:
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        # Chroma 1.x 新接口；只有旧接口失败时才走这里，错误原因会保留在最终失败里。
        return client.create_collection(
            name=name,
            configuration={"hnsw": {"space": "cosine"}},
        )


def build_index(settings: dict[str, Any], force: bool, batch_size_override: int | None) -> None:
    """完整建索引流程：校验 -> 编码 -> 写 Chroma/BM25 -> 写元数据 -> 打印完整性统计。"""
    started = time.perf_counter()
    cfg = retrieval_config(settings)
    chunks_file = resolve_path(ROOT, settings["processed_dir"]) / "chunks.jsonl"
    tags_file = ROOT / "config" / "tags.yaml"
    index_dir = resolve_path(ROOT, cfg["index_dir"])
    chroma_dir = index_dir / "chroma"
    meta_path = index_dir / "index_meta.json"
    bm25_path = index_dir / "bm25_tokens.jsonl"

    if not chunks_file.exists():
        raise SystemExit(f"找不到切块产物：{chunks_file}；先运行 scripts/build_chunks.py")
    records = read_jsonl(chunks_file)
    integrity = validate_chunks(records)

    with tags_file.open("r", encoding="utf-8") as handle:
        tags_cfg = yaml.safe_load(handle)
    terms = collect_terms(tags_cfg)

    index_dir.mkdir(parents=True, exist_ok=True)
    import chromadb  # noqa: PLC0415

    # 必须传相对路径：Chroma 1.5.9 在 Windows 下对含中文的绝对路径写不出 HNSW。
    chroma_path = chroma_relative_path(chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection_name = str(cfg["chroma_collection"])
    existing = None
    try:
        existing = client.get_collection(name=collection_name)
    except Exception:
        existing = None
    if existing is not None and not force:
        raise SystemExit(
            f"集合 {collection_name} 已存在；确认覆盖请加 --force（会删除旧集合）"
        )
    if existing is not None:
        client.delete_collection(name=collection_name)
    collection = build_collection(client, collection_name)

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    device = str(cfg.get("device", "cpu"))
    model = SentenceTransformer(str(cfg["embedding_model"]), device=device)
    model.max_seq_length = int(cfg["max_seq_length"])
    texts = [str(record["text"]) for record in records]
    batch_size = int(batch_size_override or cfg.get("batch_size", 32))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 分批写 Chroma，避免一次请求过大；metadata 只放可过滤的扁平字段。
    for start in range(0, len(records), 128):
        batch = records[start : start + 128]
        collection.add(
            ids=[str(record["chunk_id"]) for record in batch],
            embeddings=embeddings[start : start + 128].tolist(),
            documents=[str(record["text"]) for record in batch],
            metadatas=[
                {
                    "chunk_id": str(record["chunk_id"]),
                    "root_id": str(record["root_id"]),
                    "parent_id": str(record["parent_id"]),
                    "chunk_index": int(record["chunk_index"]),
                    "section_role": str(record["section_role"]),
                    "content_type": str(record["content_type"]),
                    "source_type": str(record["source_type"]),
                    "source_name": str(record["source_name"]),
                    "title": str(record["title"]),
                }
                for record in batch
            ],
        )

    token_rows = [
        {
            "chunk_id": str(record["chunk_id"]),
            "tokens": tokenize_text(str(record["text"]), terms),
        }
        for record in records
    ]
    write_jsonl(bm25_path, token_rows)

    elapsed = time.perf_counter() - started
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": str(cfg["embedding_model"]),
        "embedding_dim": embedding_dimension(model),
        "max_seq_length": int(cfg["max_seq_length"]),
        "query_instruction": str(cfg.get("query_instruction", "")),
        "chroma_collection": collection_name,
        "chroma_dir": str(chroma_dir),
        "chunk_count": integrity["chunk_count"],
        "parent_count": integrity["parent_count"],
        "root_count": integrity["root_count"],
        "chunks_sha256": file_sha256(chunks_file),
        "tags_sha256": file_sha256(tags_file),
        "bm25_k1": float(cfg["bm25_k1"]),
        "bm25_b": float(cfg["bm25_b"]),
        "bm25_doc_count": len(token_rows),
        "role_counts": integrity["role_counts"],
        "content_type_counts": integrity["content_type_counts"],
        "elapsed_sec": round(elapsed, 3),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    collection_count = collection.count()
    if collection_count != len(records):
        raise SystemExit(
            f"Chroma 实际向量数 {collection_count} 与 chunks {len(records)} 不一致"
        )
    if len(token_rows) != len(records):
        raise SystemExit("BM25 token 行数与 chunks 不一致")
    if embedding_dimension(model) != int(meta["embedding_dim"]):
        raise SystemExit("模型维度与元数据不一致")

    # Chroma 1.5.x 的 HNSW 段是后台异步落盘的。只 count() 不够：进程退出后可能只剩
    # index_metadata.pickle，重新打开会报 "Error loading hnsw index"。这里先查一次，
    # 强制 HNSW 段构建；close() 后再用新 client 重开，模拟下一次检索的真实路径。
    probe_vector = embeddings[0].tolist()
    collection.query(query_embeddings=[probe_vector], n_results=1)
    client.close()

    reopened = chromadb.PersistentClient(path=str(chroma_path))
    reopened_collection = reopened.get_collection(name=collection_name)
    if reopened_collection.count() != len(records):
        raise SystemExit("重开 Chroma 后向量数不一致，HNSW 索引未完整落盘")
    reopened_collection.query(query_embeddings=[probe_vector], n_results=1)
    reopened.close()

    index_size_mb = sum(
        path.stat().st_size for path in index_dir.rglob("*") if path.is_file()
    ) / (1024 * 1024)
    print(f"chunks: {integrity['chunk_count']}")
    print(f"parents: {integrity['parent_count']}")
    print(f"roots: {integrity['root_count']}")
    print(f"model: {cfg['embedding_model']}")
    print(f"embedding_dim: {meta['embedding_dim']}")
    print(f"collection: {collection_name} ({collection_count} vectors)")
    print(f"bm25_docs: {len(token_rows)}")
    print(f"role_counts: {integrity['role_counts']}")
    print(f"content_type_counts: {integrity['content_type_counts']}")
    print(f"index_dir: {index_dir}")
    print(f"index_size_mb: {index_size_mb:.2f}")
    print(f"elapsed_sec: {elapsed:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="1B 建索引：Chroma + BM25")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--force", action="store_true", help="覆盖同名 Chroma 集合")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    settings = load_settings(config_path)
    build_index(settings, force=args.force, batch_size_override=args.batch_size)


if __name__ == "__main__":
    main()

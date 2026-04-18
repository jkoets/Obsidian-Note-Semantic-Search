"""Persistence layer: JSON for metadata + .npz for vectors.

The index lives next to the vaults so it syncs via Resilio. Two files keep
things inspectable: you can open index.json and eyeball what's been indexed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .chunker import Chunk


INDEX_SCHEMA_VERSION = 1


@dataclass
class Index:
    """In-memory representation of the search index."""
    model_name: str
    model_dim: int
    schema_version: int = INDEX_SCHEMA_VERSION
    # Mapping: vault-qualified path "Projects/foo.md" -> {"hash": ..., "chunk_ids": [...]}
    files: dict[str, dict] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    # vectors[i] corresponds to chunks[i]
    vectors: np.ndarray | None = None

    def chunk_id_to_row(self) -> dict[str, int]:
        return {c.id: i for i, c in enumerate(self.chunks)}


def index_dir(notes_root: Path) -> Path:
    return notes_root / ".note-search"


def save_index(idx: Index, notes_root: Path) -> None:
    """Write index.json + vectors.npz atomically."""
    d = index_dir(notes_root)
    d.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": idx.schema_version,
        "model_name": idx.model_name,
        "model_dim": idx.model_dim,
        "files": idx.files,
        "chunks": [c.to_dict() for c in idx.chunks],
    }

    tmp_json = d / "index.json.tmp"
    tmp_npz = d / "vectors.npz.tmp"

    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # Pass an open file handle so numpy doesn't auto-append .npz to our temp path
    vecs_to_save = (
        idx.vectors
        if (idx.vectors is not None and len(idx.vectors) > 0)
        else np.zeros((0, idx.model_dim), dtype=np.float32)
    )
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(f, vectors=vecs_to_save)

    # Atomic-ish replace (Windows requires the target not exist, so unlink first)
    final_json = d / "index.json"
    final_npz = d / "vectors.npz"
    if final_json.exists():
        final_json.unlink()
    if final_npz.exists():
        final_npz.unlink()
    tmp_json.replace(final_json)
    tmp_npz.replace(final_npz)


def load_index(notes_root: Path) -> Optional[Index]:
    """Load the index if it exists. Returns None if absent."""
    d = index_dir(notes_root)
    idx_path = d / "index.json"
    vec_path = d / "vectors.npz"

    if not idx_path.exists():
        return None

    with open(idx_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chunks = [Chunk.from_dict(c) for c in meta.get("chunks", [])]
    vectors = None
    if vec_path.exists():
        with np.load(vec_path) as data:
            vectors = data["vectors"]

    return Index(
        model_name=meta["model_name"],
        model_dim=meta["model_dim"],
        schema_version=meta.get("schema_version", 1),
        files=meta.get("files", {}),
        chunks=chunks,
        vectors=vectors,
    )

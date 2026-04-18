"""Incremental indexer.

Strategy:
  - Walk each configured vault for *.md files
  - For each file, compute content hash
  - If hash matches prior index: keep existing chunks/vectors for that file
  - Otherwise: re-chunk + re-embed
  - Files that disappeared: drop their chunks
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
from tqdm import tqdm

from .chunker import Chunk, chunk_note, hash_file_content
from .embedder import Embedder
from .storage import Index, load_index, save_index


def _iter_markdown_files(vault_root: Path) -> Iterable[Path]:
    for p in vault_root.rglob("*.md"):
        # Skip Obsidian's own config dir and our index dir
        parts = set(p.parts)
        if ".obsidian" in parts or ".trash" in parts or ".note-search" in parts:
            continue
        yield p


def _vault_relative(notes_root: Path, vault_name: str, vault_root: Path, file_path: Path) -> str:
    """Return path relative to notes_root, using forward slashes so it
    survives cross-OS syncing cleanly."""
    rel = file_path.relative_to(notes_root)
    return rel.as_posix()


def build_or_update_index(
    notes_root: Path,
    vaults: dict[str, str],  # name -> vault folder name (relative to notes_root)
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    force: bool = False,
    max_chars: int = 2000,
    min_chars: int = 40,
    verbose: bool = True,
) -> Index:
    """Build or incrementally update the search index.

    Returns the updated Index.
    """
    existing = None if force else load_index(notes_root)

    # If model changed, force a rebuild (embeddings aren't comparable)
    if existing is not None and existing.model_name != model_name:
        if verbose:
            print(
                f"Model changed ({existing.model_name} -> {model_name}); rebuilding from scratch."
            )
        existing = None

    old_files = existing.files if existing else {}
    old_chunks_by_id: dict[str, Chunk] = (
        {c.id: c for c in existing.chunks} if existing else {}
    )
    old_vec_row: dict[str, int] = existing.chunk_id_to_row() if existing else {}
    old_vectors = existing.vectors if existing else None

    # 1. Scan current files
    current_files: dict[str, Path] = {}
    for vault_name, vault_subdir in vaults.items():
        vault_root = notes_root / vault_subdir
        if not vault_root.exists():
            if verbose:
                print(f"WARNING: vault '{vault_name}' not found at {vault_root}; skipping")
            continue
        for fp in _iter_markdown_files(vault_root):
            rel = _vault_relative(notes_root, vault_name, vault_root, fp)
            current_files[rel] = fp

    # 2. Classify: unchanged / changed / new / deleted
    unchanged_paths: list[str] = []
    changed_or_new_paths: list[str] = []

    file_contents: dict[str, str] = {}
    file_hashes: dict[str, str] = {}

    scan_iter = current_files.items()
    if verbose:
        scan_iter = tqdm(list(scan_iter), desc="Scanning notes", unit="file")

    for rel, fp in scan_iter:
        try:
            content = fp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fall back, very rare in markdown
            content = fp.read_text(encoding="utf-8", errors="replace")
        fhash = hash_file_content(content)
        file_contents[rel] = content
        file_hashes[rel] = fhash

        prior = old_files.get(rel)
        if prior and prior.get("hash") == fhash:
            unchanged_paths.append(rel)
        else:
            changed_or_new_paths.append(rel)

    deleted_paths = [p for p in old_files if p not in current_files]

    if verbose:
        print(
            f"  {len(unchanged_paths)} unchanged, "
            f"{len(changed_or_new_paths)} new/changed, "
            f"{len(deleted_paths)} deleted"
        )

    # 3. Gather kept chunks from unchanged files
    new_files_meta: dict[str, dict] = {}
    kept_chunks: list[Chunk] = []
    kept_vector_rows: list[int] = []

    for rel in unchanged_paths:
        meta = old_files[rel]
        new_files_meta[rel] = meta
        for cid in meta.get("chunk_ids", []):
            c = old_chunks_by_id.get(cid)
            if c is None:
                continue
            row = old_vec_row.get(cid)
            if row is None:
                continue
            kept_chunks.append(c)
            kept_vector_rows.append(row)

    # 4. Chunk new/changed files
    new_chunks: list[Chunk] = []
    for rel in changed_or_new_paths:
        content = file_contents[rel]
        fhash = file_hashes[rel]
        vault_name = rel.split("/", 1)[0]
        path_in_vault = rel
        chunks = chunk_note(
            text=content,
            vault=vault_name,
            path=path_in_vault,
            file_hash=fhash,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        new_chunks.extend(chunks)
        new_files_meta[rel] = {
            "hash": fhash,
            "chunk_ids": [c.id for c in chunks],
        }

    # 5. Embed new chunks
    embedder: Embedder | None = None
    new_vectors: np.ndarray | None = None
    model_dim: int
    if new_chunks:
        if verbose:
            print(f"Loading embedding model '{model_name}'...")
        embedder = Embedder(model_name=model_name)
        model_dim = embedder.dim
        if verbose:
            print(f"Embedding {len(new_chunks)} new/changed chunks on {embedder.device}...")
        texts = [c.embedding_text() for c in new_chunks]
        new_vectors = embedder.embed_documents(texts)
    else:
        # No embedding needed - but we still need a dim for saving
        if existing is not None:
            model_dim = existing.model_dim
        else:
            # Unusual case: first-ever run with zero notes. Load model anyway.
            embedder = Embedder(model_name=model_name)
            model_dim = embedder.dim

    # 6. Assemble final vectors array
    if kept_vector_rows and old_vectors is not None:
        kept_vectors = old_vectors[kept_vector_rows]
    else:
        kept_vectors = np.zeros((0, model_dim), dtype=np.float32)

    if new_vectors is not None and len(new_vectors) > 0:
        combined_vectors = np.vstack([kept_vectors, new_vectors])
    else:
        combined_vectors = kept_vectors

    all_chunks = kept_chunks + new_chunks

    idx = Index(
        model_name=model_name,
        model_dim=model_dim,
        files=new_files_meta,
        chunks=all_chunks,
        vectors=combined_vectors,
    )

    save_index(idx, notes_root)
    if verbose:
        print(f"Index saved: {len(all_chunks)} chunks across {len(new_files_meta)} notes.")
    return idx

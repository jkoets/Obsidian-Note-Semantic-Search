"""Hybrid search: semantic (dense) + BM25 (lexical) + optional wikilink graph boost.

Fusion method: Reciprocal Rank Fusion. A chunk's combined score is
    sum over rankings r of: 1 / (k + rank_in_r)
where k is a smoothing constant (60 is the canonical choice from Cormack et al).

Why RRF over weighted score sum: semantic cosines and BM25 scores aren't
on the same scale, and normalizing them is fiddly. RRF only uses ranks,
which are scale-invariant.

Wikilink graph boost (optional):
    After initial hybrid ranking, look at the top-K candidates. Any note those
    candidates [[link to]] is considered "relevant neighborhood" - chunks from
    those linked notes get an additional RRF contribution. This catches the
    case where the top hit says "see also [[Design Notes]]" and Design Notes
    is itself a great match but didn't rank high on lexical/semantic alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .chunker import Chunk
from .embedder import Embedder
from .storage import Index


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _note_basename(path: str) -> str:
    """'Projects/Daily/2024-01-15.md' -> '2024-01-15' (Obsidian wikilink target form)."""
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".md"):
        name = name[:-3]
    return name


@dataclass
class SearchResult:
    chunk: Chunk
    score: float             # combined RRF score
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    link_rank: int | None = None  # rank among chunks promoted by graph boost
    semantic_score: float | None = None
    lexical_score: float | None = None


class Searcher:
    """Holds the loaded index and lazy BM25 tokenization."""

    def __init__(self, index: Index):
        self.index = index
        self._bm25 = None
        self._embedder: Optional[Embedder] = None
        self._basename_index: dict[str, list[str]] | None = None  # basename -> paths

    def _build_bm25(self):
        """Build BM25 on chunk embedding-text (heading path + body). Cheap: seconds for thousands."""
        from rank_bm25 import BM25Okapi

        corpus_tokens = [_tokenize(c.embedding_text()) for c in self.index.chunks]
        self._bm25 = BM25Okapi(corpus_tokens)

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(model_name=self.index.model_name)
        return self._embedder

    def _get_basename_index(self) -> dict[str, list[str]]:
        """Map note basename (lowercase) -> list of vault-qualified paths.

        Built once, cached. Lowercased keys because Obsidian wikilinks are
        case-insensitive in practice on Windows/Mac filesystems.
        """
        if self._basename_index is not None:
            return self._basename_index
        bi: dict[str, list[str]] = {}
        seen_paths: set[str] = set()
        for c in self.index.chunks:
            if c.path in seen_paths:
                continue
            seen_paths.add(c.path)
            key = _note_basename(c.path).lower()
            bi.setdefault(key, []).append(c.path)
        self._basename_index = bi
        return bi

    def _resolve_wikilinks(self, wikilinks: list[str]) -> set[str]:
        """Wikilink targets -> set of indexed paths.

        Handles both bare names ([[Design Notes]]) and path-qualified
        names ([[folder/Design Notes]]). Unresolved links are dropped.
        """
        bi = self._get_basename_index()
        resolved: set[str] = set()
        for wl in wikilinks:
            # Take just the tail if the link has a path component
            tail = wl.rsplit("/", 1)[-1].strip().lower()
            if not tail:
                continue
            # Try basename match first
            if tail in bi:
                resolved.update(bi[tail])
                continue
            # Fall back: try matching the full link as a path
            wl_lower = wl.lower()
            for c in self.index.chunks:
                if c.path.lower().rstrip(".md") == wl_lower or \
                   c.path.lower() == wl_lower + ".md":
                    resolved.add(c.path)
        return resolved

    def search(
        self,
        query: str,
        top_n: int = 10,
        vault: str | None = None,
        tag: str | None = None,
        candidate_k: int = 50,
        rrf_k: int = 60,
        mode: str = "hybrid",  # "hybrid" | "semantic" | "lexical"
        graph_boost: bool = True,
        graph_seed_k: int = 5,  # how many top results seed the link expansion
    ) -> list[SearchResult]:
        if len(self.index.chunks) == 0:
            return []

        # Build a filter mask once
        def keep(c: Chunk) -> bool:
            if vault is not None and c.vault != vault:
                return False
            if tag is not None:
                # allow match on exact tag or prefix (e.g. "project" matches "project/alpha")
                if not any(t == tag or t.startswith(tag + "/") for t in c.tags):
                    return False
            return True

        filtered_indices = [i for i, c in enumerate(self.index.chunks) if keep(c)]
        if not filtered_indices:
            return []
        filtered_set = set(filtered_indices)

        semantic_ranking: list[tuple[int, float]] = []
        lexical_ranking: list[tuple[int, float]] = []

        if mode in ("hybrid", "semantic"):
            embedder = self._get_embedder()
            qvec = embedder.embed_query(query)
            # Vectors are already L2-normalized, so dot product == cosine similarity
            vectors = self.index.vectors
            sims = vectors @ qvec
            # Mask out filtered-out chunks
            mask = np.full(len(self.index.chunks), -np.inf, dtype=np.float32)
            mask[filtered_indices] = 0.0
            masked_sims = sims + mask
            k = min(candidate_k, len(filtered_indices))
            top_idx = np.argpartition(-masked_sims, k - 1)[:k]
            # Sort those k by actual score
            top_idx = top_idx[np.argsort(-masked_sims[top_idx])]
            semantic_ranking = [(int(i), float(masked_sims[i])) for i in top_idx]

        if mode in ("hybrid", "lexical"):
            if self._bm25 is None:
                self._build_bm25()
            query_tokens = _tokenize(query)
            if query_tokens:
                scores = self._bm25.get_scores(query_tokens)
                # Mask filtered
                mask = np.full(len(self.index.chunks), -np.inf, dtype=np.float32)
                mask[filtered_indices] = 0.0
                masked_scores = scores + mask
                k = min(candidate_k, len(filtered_indices))
                top_idx = np.argpartition(-masked_scores, k - 1)[:k]
                top_idx = top_idx[np.argsort(-masked_scores[top_idx])]
                lexical_ranking = [(int(i), float(masked_scores[i])) for i in top_idx]

        # Combine via RRF
        combined: dict[int, dict] = {}
        for rank, (idx, score) in enumerate(semantic_ranking):
            entry = combined.setdefault(idx, {"score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank + 1)
            entry["semantic_rank"] = rank + 1
            entry["semantic_score"] = score
        for rank, (idx, score) in enumerate(lexical_ranking):
            entry = combined.setdefault(idx, {"score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank + 1)
            entry["lexical_rank"] = rank + 1
            entry["lexical_score"] = score

        # Optional wikilink graph boost.
        # Take the top-K candidates after initial fusion, collect their wikilinks,
        # resolve to target notes, and give chunks from those notes an RRF bump.
        if graph_boost and combined:
            pre_ranked = sorted(combined.items(), key=lambda kv: kv[1]["score"], reverse=True)
            seed_indices = [i for i, _ in pre_ranked[:graph_seed_k]]
            seed_wikilinks: list[str] = []
            for si in seed_indices:
                seed_wikilinks.extend(self.index.chunks[si].wikilinks)

            if seed_wikilinks:
                linked_paths = self._resolve_wikilinks(seed_wikilinks)
                # Exclude paths already in seeds so the boost reveals *new* notes
                seed_paths = {self.index.chunks[si].path for si in seed_indices}
                linked_paths -= seed_paths

                if linked_paths:
                    # Rank chunks in linked notes by their dot product with the
                    # combined query signal (use semantic sims if available,
                    # else BM25 scores, else by chunk position).
                    link_candidates: list[tuple[int, float]] = []
                    for i, c in enumerate(self.index.chunks):
                        if i not in filtered_set:
                            continue
                        if c.path not in linked_paths:
                            continue
                        # Score proxy: use whatever we already computed for this chunk.
                        # If it wasn't in either top-K, use 0 so ranking falls back to chunk order.
                        sscore = next(
                            (s for idx_s, s in semantic_ranking if idx_s == i), 0.0
                        )
                        lscore = next(
                            (s for idx_l, s in lexical_ranking if idx_l == i), 0.0
                        )
                        link_candidates.append((i, sscore + lscore))

                    link_candidates.sort(key=lambda x: x[1], reverse=True)
                    # Apply boost. Use a higher rrf_k so link hits contribute less
                    # than direct semantic/lexical matches of the same rank.
                    link_rrf_k = rrf_k * 2
                    for rank, (idx, _) in enumerate(link_candidates):
                        entry = combined.setdefault(idx, {"score": 0.0})
                        entry["score"] += 1.0 / (link_rrf_k + rank + 1)
                        entry["link_rank"] = rank + 1

        ranked = sorted(combined.items(), key=lambda kv: kv[1]["score"], reverse=True)
        results: list[SearchResult] = []
        for idx, entry in ranked[:top_n]:
            c = self.index.chunks[idx]
            results.append(
                SearchResult(
                    chunk=c,
                    score=entry["score"],
                    semantic_rank=entry.get("semantic_rank"),
                    lexical_rank=entry.get("lexical_rank"),
                    link_rank=entry.get("link_rank"),
                    semantic_score=entry.get("semantic_score"),
                    lexical_score=entry.get("lexical_score"),
                )
            )
        return results

    def related(self, path: str, top_n: int = 10) -> list[SearchResult]:
        """Find chunks semantically related to an existing note.

        Averages the chunks of the source note into a single query vector.
        """
        source_rows = [i for i, c in enumerate(self.index.chunks) if c.path == path]
        if not source_rows:
            return []

        src_vecs = self.index.vectors[source_rows]
        qvec = src_vecs.mean(axis=0)
        # Renormalize
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        sims = self.index.vectors @ qvec
        # Exclude chunks from the source note
        for i in source_rows:
            sims[i] = -np.inf

        k = min(top_n * 3, len(self.index.chunks))  # slight over-fetch so we can dedupe by note
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        seen_paths: set[str] = set()
        results: list[SearchResult] = []
        for i in top_idx:
            c = self.index.chunks[int(i)]
            if c.path in seen_paths:
                continue
            seen_paths.add(c.path)
            results.append(
                SearchResult(
                    chunk=c,
                    score=float(sims[i]),
                    semantic_score=float(sims[i]),
                )
            )
            if len(results) >= top_n:
                break
        return results

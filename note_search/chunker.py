"""Chunk markdown notes by heading structure, preserving context.

Each chunk carries its heading path (e.g. ["Project Alpha", "Design Notes"])
so both the embedding and the search result display are context-rich.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Iterator


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[#|][^\]]*)?\]\]")
INLINE_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


@dataclass
class Chunk:
    """A single searchable slice of a note."""
    id: str                      # content-addressed hash
    vault: str                   # e.g. "Projects"
    path: str                    # vault-relative, e.g. "Daily/2024-01-15.md"
    heading_path: list[str]      # e.g. ["Meeting", "Action Items"]
    text: str                    # raw chunk text (no heading prefix)
    tags: list[str]              # from frontmatter + inline #tags
    wikilinks: list[str]         # [[target]] references in this chunk
    file_hash: str               # hash of source file content
    position: int                # 0-based index of chunk within file

    def embedding_text(self) -> str:
        """Text actually passed to the embedder.

        Prepending the heading path gives the embedding model topical context
        that bare chunk text often lacks - especially for short bullet-point
        sections under meaningful headings.
        """
        if self.heading_path:
            path_str = " > ".join(self.heading_path)
            return f"{path_str}\n\n{self.text}"
        return self.text

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _parse_frontmatter(text: str) -> tuple[dict, str, list[str]]:
    """Pull YAML frontmatter tags and return (frontmatter_tags, body_text, all_tags_list).

    We only care about tags here - full YAML parsing would pull in an extra
    dep for minimal value in a retrieval context.
    """
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text, []

    fm_text = m.group(1)
    body = text[m.end():]

    # Cheap tag extraction - catches `tags: [a, b]`, `tags:\n  - a\n  - b`, `tags: a`
    tags: list[str] = []
    in_tags_block = False
    for line in fm_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("tags:"):
            rest = stripped[5:].strip()
            if rest.startswith("[") and rest.endswith("]"):
                tags.extend(t.strip().strip("\"'") for t in rest[1:-1].split(",") if t.strip())
                in_tags_block = False
            elif rest:
                tags.append(rest.strip("\"'"))
                in_tags_block = False
            else:
                in_tags_block = True
        elif in_tags_block and stripped.startswith("-"):
            tags.append(stripped[1:].strip().strip("\"'"))
        elif in_tags_block and not line.startswith(" ") and not line.startswith("\t"):
            in_tags_block = False

    tags = [t for t in tags if t]
    return {"tags": tags}, body, tags


def _hash_chunk(vault: str, path: str, position: int, text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(vault.encode("utf-8"))
    h.update(b"\0")
    h.update(path.encode("utf-8"))
    h.update(b"\0")
    h.update(str(position).encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _split_oversize(text: str, max_chars: int) -> list[str]:
    """Split a too-long section into paragraph-aligned sub-chunks."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        plen = len(para)
        if buf_len + plen + 2 > max_chars and buf:
            pieces.append("\n\n".join(buf))
            buf, buf_len = [], 0
        if plen > max_chars:
            # Single paragraph is too big - hard-split on sentence-ish boundaries
            if buf:
                pieces.append("\n\n".join(buf))
                buf, buf_len = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sbuf: list[str] = []
            slen = 0
            for s in sentences:
                if slen + len(s) + 1 > max_chars and sbuf:
                    pieces.append(" ".join(sbuf))
                    sbuf, slen = [], 0
                sbuf.append(s)
                slen += len(s) + 1
            if sbuf:
                pieces.append(" ".join(sbuf))
        else:
            buf.append(para)
            buf_len += plen + 2

    if buf:
        pieces.append("\n\n".join(buf))
    return pieces


def chunk_note(
    text: str,
    vault: str,
    path: str,
    file_hash: str,
    max_chars: int = 2000,
    min_chars: int = 40,
) -> list[Chunk]:
    """Slice a markdown note into heading-aligned chunks.

    Rules:
      - Each heading (any level) starts a new chunk.
      - Heading path tracks ancestors only (a new H2 clears its own level + deeper).
      - Chunks shorter than `min_chars` are dropped (unless the file is entirely tiny).
      - Chunks longer than `max_chars` are paragraph-split.
    """
    _, body, fm_tags = _parse_frontmatter(text)

    lines = body.split("\n")
    heading_stack: list[tuple[int, str]] = []
    current_heading_path: list[str] = []
    current_lines: list[str] = []
    sections: list[tuple[list[str], str]] = []  # (heading_path, text)

    def flush():
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((list(current_heading_path), section_text))

    in_code_fence = False
    for line in lines:
        # Don't treat #-lines inside code fences as headings
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence
            current_lines.append(line)
            continue

        if not in_code_fence:
            m = HEADING_RE.match(line)
            if m:
                flush()
                current_lines = []
                level = len(m.group(1))
                heading_text = m.group(2).strip()
                # Pop stack down to level-1
                heading_stack = [(l, t) for l, t in heading_stack if l < level]
                heading_stack.append((level, heading_text))
                current_heading_path = [t for _, t in heading_stack]
                continue

        current_lines.append(line)

    flush()

    # If the entire note produced no sections (e.g. empty body), skip it
    if not sections:
        return []

    chunks: list[Chunk] = []
    position = 0
    for heading_path, section_text in sections:
        pieces = _split_oversize(section_text, max_chars)
        for piece in pieces:
            piece_stripped = piece.strip()
            # Keep short chunks if they have a heading path (headings can be meaningful)
            if len(piece_stripped) < min_chars and not heading_path:
                continue
            if not piece_stripped:
                continue

            # Extract metadata specific to this piece
            inline_tags = {m.group(1) for m in INLINE_TAG_RE.finditer(piece_stripped)}
            wikilinks = [m.group(1).strip() for m in WIKILINK_RE.finditer(piece_stripped)]
            all_tags = sorted(set(fm_tags) | inline_tags)

            cid = _hash_chunk(vault, path, position, piece_stripped)
            chunks.append(
                Chunk(
                    id=cid,
                    vault=vault,
                    path=path,
                    heading_path=heading_path,
                    text=piece_stripped,
                    tags=all_tags,
                    wikilinks=wikilinks,
                    file_hash=file_hash,
                    position=position,
                )
            )
            position += 1

    return chunks


def hash_file_content(content: str) -> str:
    """Content hash for change detection. Blake2b is fast and collision-safe here."""
    return hashlib.blake2b(content.encode("utf-8"), digest_size=16).hexdigest()

"""Command-line interface for note-search.

Installed entry point: `ns` (see pyproject.toml).
Also runnable as `python -m note_search`.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

import click

from .indexer import build_or_update_index
from .search import Searcher, SearchResult
from .storage import load_index


DEFAULT_CONFIG_NAMES = ["note-search.config.json", "config.json"]


def _find_config() -> Optional[Path]:
    """Search for config in: env var, CWD, user home, script directory."""
    env = os.environ.get("NOTE_SEARCH_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    candidates = []
    cwd = Path.cwd()
    home = Path.home()
    script_dir = Path(__file__).resolve().parent.parent

    for d in (cwd, home, script_dir):
        for name in DEFAULT_CONFIG_NAMES:
            candidates.append(d / name)

    for p in candidates:
        if p.exists():
            return p
    return None


def _load_config(explicit_path: Optional[str]) -> dict:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.exists():
            raise click.ClickException(f"Config not found: {p}")
    else:
        p = _find_config()
        if p is None:
            raise click.ClickException(
                "No config file found. Create note-search.config.json "
                "(see config.example.json) or set NOTE_SEARCH_CONFIG."
            )
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Expand ~ in notes_root
    cfg["notes_root"] = str(Path(cfg["notes_root"]).expanduser())
    return cfg


def _obsidian_url(vault: str, path_in_notes: str) -> str:
    """Build obsidian:// URL. path_in_notes is like 'Projects/Daily/2024.md';
    we strip the leading vault folder since Obsidian expects vault-relative path."""
    prefix = vault + "/"
    if path_in_notes.startswith(prefix):
        vault_rel = path_in_notes[len(prefix):]
    else:
        vault_rel = path_in_notes
    # Drop .md extension (Obsidian doesn't need it)
    if vault_rel.endswith(".md"):
        vault_rel = vault_rel[:-3]
    return (
        f"obsidian://open?vault={urllib.parse.quote(vault)}"
        f"&file={urllib.parse.quote(vault_rel)}"
    )


def _format_result(r: SearchResult, max_snippet: int = 240) -> str:
    c = r.chunk
    heading = " > ".join(c.heading_path) if c.heading_path else "(no heading)"
    snippet = c.text.replace("\n", " ").strip()
    if len(snippet) > max_snippet:
        snippet = snippet[: max_snippet - 1] + "…"
    ranks = []
    if r.semantic_rank is not None:
        ranks.append(f"sem #{r.semantic_rank}")
    if r.lexical_rank is not None:
        ranks.append(f"lex #{r.lexical_rank}")
    if r.link_rank is not None:
        ranks.append(f"link #{r.link_rank}")
    rank_str = f" [{', '.join(ranks)}]" if ranks else ""

    url = _obsidian_url(c.vault, c.path)

    return (
        f"  {click.style(c.vault + '/' + c.path.split('/', 1)[-1], fg='cyan', bold=True)}"
        f"  {click.style(f'(score {r.score:.4f}{rank_str})', fg='white', dim=True)}\n"
        f"    {click.style(heading, fg='yellow')}\n"
        f"    {snippet}\n"
        f"    {click.style(url, fg='blue', underline=True)}"
    )


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config JSON.")
@click.pass_context
def cli(ctx, config_path):
    """Semantic + lexical search over Obsidian vaults."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    # Config is loaded lazily per-command so --help works without a config file.


def _config(ctx) -> dict:
    """Load config on first access, cache it on the context."""
    if "config" not in ctx.obj:
        ctx.obj["config"] = _load_config(ctx.obj.get("config_path"))
    return ctx.obj["config"]


@cli.command()
@click.option("--force", is_flag=True, help="Rebuild from scratch (ignore existing index).")
@click.pass_context
def index(ctx, force):
    """Build or incrementally update the search index."""
    cfg = _config(ctx)
    notes_root = Path(cfg["notes_root"])
    build_or_update_index(
        notes_root=notes_root,
        vaults=cfg["vaults"],
        model_name=cfg.get("model", "nomic-ai/nomic-embed-text-v1.5"),
        force=force,
        max_chars=cfg.get("chunk_max_chars", 2000),
        min_chars=cfg.get("chunk_min_chars", 40),
    )


@cli.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-n", "--top-n", default=10, help="Number of results to return.")
@click.option("--vault", default=None, help="Restrict to a single vault.")
@click.option("--tag", default=None, help="Restrict to chunks with this tag (or children).")
@click.option("--mode", type=click.Choice(["hybrid", "semantic", "lexical"]), default="hybrid")
@click.option("--no-graph-boost", is_flag=True, help="Disable wikilink graph boost.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of formatted text.")
@click.pass_context
def search(ctx, query, top_n, vault, tag, mode, no_graph_boost, as_json):
    """Search the index. Example: ns search neural network architecture"""
    cfg = _config(ctx)
    notes_root = Path(cfg["notes_root"])
    idx = load_index(notes_root)
    if idx is None:
        raise click.ClickException(f"No index found under {notes_root}. Run `ns index` first.")

    q = " ".join(query)
    searcher = Searcher(idx)
    results = searcher.search(
        query=q, top_n=top_n, vault=vault, tag=tag, mode=mode,
        graph_boost=not no_graph_boost,
    )

    if as_json:
        out = []
        for r in results:
            out.append({
                "score": r.score,
                "semantic_rank": r.semantic_rank,
                "lexical_rank": r.lexical_rank,
                "link_rank": r.link_rank,
                "vault": r.chunk.vault,
                "path": r.chunk.path,
                "heading_path": r.chunk.heading_path,
                "text": r.chunk.text,
                "tags": r.chunk.tags,
                "obsidian_url": _obsidian_url(r.chunk.vault, r.chunk.path),
            })
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if not results:
        click.echo(click.style("No results.", fg="red"))
        return

    click.echo(click.style(f"\nResults for: {q}\n", bold=True))
    for r in results:
        click.echo(_format_result(r))
        click.echo()


@cli.command()
@click.argument("note_path")
@click.option("-n", "--top-n", default=10)
@click.pass_context
def related(ctx, note_path, top_n):
    """Find notes related to the given note. NOTE_PATH is vault-relative,
    e.g. 'Projects/Ideas/new-thing.md'"""
    cfg = _config(ctx)
    notes_root = Path(cfg["notes_root"])
    idx = load_index(notes_root)
    if idx is None:
        raise click.ClickException(f"No index found under {notes_root}. Run `ns index` first.")

    searcher = Searcher(idx)
    # Accept either 'Vault/file.md' or absolute path
    p = note_path
    if Path(note_path).is_absolute():
        try:
            p = Path(note_path).resolve().relative_to(notes_root).as_posix()
        except ValueError:
            raise click.ClickException(f"{note_path} is not under {notes_root}")

    results = searcher.related(p, top_n=top_n)
    if not results:
        click.echo(click.style(f"No related notes found (is '{p}' indexed?).", fg="red"))
        return

    click.echo(click.style(f"\nRelated to {p}:\n", bold=True))
    for r in results:
        click.echo(_format_result(r))
        click.echo()


@cli.command()
@click.pass_context
def stats(ctx):
    """Print index statistics."""
    cfg = _config(ctx)
    notes_root = Path(cfg["notes_root"])
    idx = load_index(notes_root)
    if idx is None:
        click.echo("No index exists yet.")
        return

    vault_counts: dict[str, int] = {}
    for c in idx.chunks:
        vault_counts[c.vault] = vault_counts.get(c.vault, 0) + 1

    click.echo(click.style("Index stats", bold=True))
    click.echo(f"  Notes root:   {notes_root}")
    click.echo(f"  Model:        {idx.model_name} (dim {idx.model_dim})")
    click.echo(f"  Files:        {len(idx.files)}")
    click.echo(f"  Chunks:       {len(idx.chunks)}")
    click.echo(f"  Vector bytes: "
               f"{(idx.vectors.nbytes if idx.vectors is not None else 0):,}")
    click.echo("  Chunks per vault:")
    for v, n in sorted(vault_counts.items()):
        click.echo(f"    {v:20s} {n}")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()

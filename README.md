# note-search

Local, private, semantic + lexical search over your Obsidian vaults. Runs
embeddings on your own GPU (or CPU) and stores the index as JSON + numpy next
to your notes, so it syncs across machines through whatever you're using (Resilio in my case)
along with the notes themselves. 

## What it does

- Chunks your markdown notes by heading structure (so chunks carry their
  heading path as context)
- Embeds each chunk with `nomic-embed-text-v1.5` locally
- Hybrid search at query time: dense (semantic) + BM25 (lexical), combined
  via Reciprocal Rank Fusion - helps when you remember an exact weird phrase
  *and* when you only remember the shape of an idea
- Incremental re-indexing - only changed files get re-embedded
- `related` command - find notes semantically similar to any given note
- Results include `obsidian://` deep links so you can click straight into the note

## Install

From the repo directory, on each machine:

```
pip install -e .
```

That gives you a `ns` command. Python 3.10+.

On first run the nomic model (~550 MB) downloads from HuggingFace and caches
under `~/.cache/huggingface/`.

### Optional: flash-attention

The nomic model will log a warning about flash-attention not being installed.
It still works fine without it, just slightly slower. Safe to ignore.


## Configure

Copy `config.example.json` somewhere the tool can find it. In order of
precedence:

1. `NOTE_SEARCH_CONFIG` env var
2. `./note-search.config.json` in current directory
3. `~/note-search.config.json` in home directory
4. Next to the installed package

Minimal config:

```json
{
  "notes_root": "C:\\Users\\user\\Notes",
  "vaults": {
    "Projects": "Projects",
    "Work": "Work",
    "AI_Notes": "AI_Notes"
  }
}
```
Replace the path there with the path to where your vault(s) are, and replace the vault names with yours. I have one folder called Notes (the one that syncs across all my machines) that has multiple vault folders in it. 

`notes_root` is the parent directory that contains your vault folders. The
index lives at `<notes_root>/.note-search/` and syncs along with your notes.

## Make it easy to get to
I added it to my PowerShell profile, so from any PowerShell window I can call it.
To find your profile, from PowerShell type $PROFILE to get the path. 
Add this, and then restart PowerShell:

```
function ns {
    & "C:\path\to\your\python.exe" -m note_search @args
}
```

## Use

```
ns index                          # build / update the index
ns search neural network ideas    # hybrid search (semantic + BM25 + graph boost)
ns search "state machines" -n 20  # more results
ns search quantum --vault Projects
ns search deploy --tag project/alpha
ns search foo --mode semantic     # or --mode lexical
ns search foo --no-graph-boost    # disable wikilink promotion
ns search foo --json              # machine-readable output
ns related Projects/Ideas/foo.md  # notes similar to this one
ns stats                          # how big is the index?
```

First index run on a few thousand notes with a 3090 (24GB VRAM) should take a couple
minutes at most. Subsequent runs only re-embed changed files, so typically
seconds.

## How hybrid search works

Three ranking signals are combined via Reciprocal Rank Fusion:

1. **Semantic** - cosine similarity between your query vector and each chunk
   vector. Handles conceptual matches ("growing tomatoes" finds "planting
   vegetables").
2. **Lexical (BM25)** - classic keyword scoring. Handles exact-phrase recall
   when you remember a specific word.
3. **Wikilink graph boost** (optional, on by default) - the top-K candidates'
   outgoing `[[wikilinks]]` promote their targets. This surfaces notes that
   are topologically adjacent to great matches, even if the linked note
   doesn't use the query words itself.

You can see which signals contributed to each result - look for the
`[sem #1, lex #3, link #2]` annotation in the output.

## How the index syncs across machines

The plan:
1. Run `ns index` on your 3090 machine whenever you remember to (or on a schedule)
2. Resilio syncs `.note-search/index.json` and `.note-search/vectors.npz` to
   your other machines along with the notes
3. On the other machines, just run `ns search ...` - no indexing needed

If two machines index at the same time you'll get a last-write-wins conflict.
Since embeddings are deterministic given the same model + same notes, this is
usually fine, but avoid racing. If you want to re-index from a different
machine, just run `ns index` there - it'll detect only what's actually changed.

If you change the embedding model in config, the index is rebuilt from scratch
automatically (old embeddings aren't comparable to new ones).

## Files

```
note_search/
  chunker.py    # heading-aware markdown chunking
  embedder.py   # nomic-embed-text-v1.5 wrapper
  indexer.py    # incremental indexing
  search.py     # hybrid search (semantic + BM25 + RRF)
  storage.py    # JSON + numpy persistence
  cli.py        # click CLI
```

## Troubleshooting

- **"No config file found"** - see Configure section above
- **"No index found"** - run `ns index` first
- **Slow first run** - model download is ~550 MB, one-time
- **Out of VRAM** - reduce `batch_size` in
  `embedder.py` (default 32). Even 8 works fine for indexing.

"""Local embedder using nomic-embed-text-v1.5.

Nomic's v1.5 model uses task-specific prefixes. We use:
  - "search_document: " when indexing chunks
  - "search_query: "    when the user is searching
Skipping the prefixes works but gives noticeably worse retrieval.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        device: str | None = None,
        batch_size: int = 32,
    ):
        # Imported lazily so the CLI can start without loading torch
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        # trust_remote_code is required because nomic ships a custom model class
        self._model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device
        )
        # Resolve actual embedding dimension from the model
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Embed chunks for indexing. Returns L2-normalized float32 array."""
        texts = [f"search_document: {t}" for t in texts]
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 64,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns L2-normalized float32 vector."""
        vec = self._model.encode(
            [f"search_query: {text}"],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

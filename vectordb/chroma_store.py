"""
vectordb/chroma_store.py
-------------------------
ChromaDB backend — persistent on disk (or in-memory for tests).

Install:
    pip install chromadb

Persistent usage (recommended):
    store = ChromaStore(collection="movies", persist_dir="./chroma_data")

In-memory usage (tests / CI):
    store = ChromaStore(collection="movies", persist_dir=None)

ChromaDB runs entirely in-process — no separate server needed.
Data is stored in a local SQLite + parquet layout under persist_dir.
"""

import numpy as np
from typing import Optional

from vectordb.base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """
    Vector store backed by ChromaDB (persistent, embedded mode).

    Parameters
    ----------
    collection : str
        Name of the ChromaDB collection (think: table name).
    persist_dir : str or None
        Path to the directory where ChromaDB stores its data on disk.
        Pass None for an ephemeral in-memory instance.
    distance : str
        Distance metric: "cosine" | "l2" | "ip" (inner product).
    """

    def __init__(
        self,
        collection: str = "recsys",
        persist_dir: Optional[str] = "./chroma_data",
        distance: str = "cosine",
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed.\n"
                "Run:  pip install chromadb"
            )

        self.collection_name = collection
        self.persist_dir = persist_dir
        self.distance = distance

        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
            print(f"[ChromaStore] Persistent client at: {persist_dir}")
        else:
            self._client = chromadb.EphemeralClient()
            print("[ChromaStore] Ephemeral (in-memory) client.")

        # get_or_create: safe for both first run and resume
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": distance},
        )
        print(f"[ChromaStore] Collection '{collection}' ready. "
              f"Existing vectors: {self._col.count()}")

    # ── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        item_id: str,
        vector: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        self._col.upsert(
            ids=[str(item_id)],
            embeddings=[vector.tolist()],
            metadatas=[metadata or {}],
        )

    def add_batch(
        self,
        item_ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict] | None = None,
    ) -> None:
        ids = [str(i) for i in item_ids]
        embeddings = [v.tolist() for v in vectors]
        metas = metadatas if metadatas else [{} for _ in ids]
        self._col.upsert(ids=ids, embeddings=embeddings, metadatas=metas)

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Query ChromaDB for nearest neighbours.

        Parameters
        ----------
        filters : dict or None
            ChromaDB `where` clause, e.g. {"genre": "sci-fi"}.
            Passed directly to collection.query().

        Returns
        -------
        list of (item_id, similarity_score) — similarity in [0, 1] for cosine.
        """
        kwargs = dict(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self._col.count()),
            include=["distances"],
        )
        if filters:
            kwargs["where"] = filters

        results = self._col.query(**kwargs)

        ids = results["ids"][0]
        distances = results["distances"][0]

        # ChromaDB returns distance, not similarity.
        # For cosine:  similarity = 1 - distance
        # For l2:      similarity = 1 / (1 + distance)
        if self.distance == "cosine":
            scores = [1.0 - d for d in distances]
        elif self.distance == "ip":
            scores = distances  # inner product IS the score
        else:  # l2
            scores = [1.0 / (1.0 + d) for d in distances]

        return list(zip(ids, scores))

    def count(self) -> int:
        return self._col.count()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )
        print(f"[ChromaStore] Collection '{self.collection_name}' reset.")

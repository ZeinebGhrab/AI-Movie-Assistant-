"""
utils/vector_store.py
----------------------
Qdrant-backed vector store — persistent, production-grade.

Replaces the previous in-memory numpy store.
Each collection in Qdrant maps to one "index" (movies, users, etc.).

Requires Qdrant running via Docker:
    docker compose up -d
"""

import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)


class VectorStore:
    """
    Qdrant-backed vector store with cosine similarity search.

    Parameters
    ----------
    collection_name : str
        Name of the Qdrant collection (e.g. "movies", "users").
    dim : int
        Vector dimensionality.
    host : str
        Qdrant host. Default: "localhost".
    port : int
        Qdrant REST port. Default: 6333.
    recreate : bool
        If True, drop and recreate the collection on init (useful for fresh runs).
    """

    def __init__(
        self,
        collection_name: str,
        dim: int,
        host: str = "localhost",
        port: int = 6333,
        recreate: bool = False,
    ):
        self.collection_name = collection_name
        self.dim = dim
        self._id_map: dict = {}        # item_id (any) → qdrant uint id
        self._reverse_map: dict = {}   # qdrant uint id → item_id

        self.client = QdrantClient(host=host, port=port)

        existing = [c.name for c in self.client.get_collections().collections]

        if recreate and collection_name in existing:
            self.client.delete_collection(collection_name)
            existing = []

        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"[VectorStore] Created collection '{collection_name}' (dim={dim})")
        else:
            print(f"[VectorStore] Using existing collection '{collection_name}'")
            # Rebuild id maps from stored payloads
            self._rebuild_maps()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_uint(self, item_id) -> int:
        """Map any item_id to a stable positive integer for Qdrant."""
        if item_id not in self._id_map:
            uid = len(self._id_map) + 1
            self._id_map[item_id] = uid
            self._reverse_map[uid] = item_id
        return self._id_map[item_id]

    def _rebuild_maps(self):
        """Re-populate id maps from Qdrant payload on startup."""
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                orig_id = rec.payload.get("item_id")
                if orig_id is not None:
                    self._id_map[orig_id] = rec.id
                    self._reverse_map[rec.id] = orig_id
            if offset is None:
                break

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add(self, item_id, vector: np.ndarray, payload: dict = None) -> None:
        """Upsert a single vector into the collection."""
        uid = self._to_uint(item_id)
        base_payload = {"item_id": item_id}
        if payload:
            base_payload.update(payload)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=uid,
                    vector=vector.astype(np.float32).tolist(),
                    payload=base_payload,
                )
            ],
        )

    def add_batch(
        self,
        item_ids: list,
        vectors: np.ndarray,
        payloads: list[dict] = None,
        batch_size: int = 128,
    ) -> None:
        """Upsert a batch of vectors. Splits into sub-batches automatically."""
        vectors = np.array(vectors, dtype=np.float32)
        n = len(item_ids)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            points = []
            for i in range(start, end):
                uid = self._to_uint(item_ids[i])
                base_payload = {"item_id": item_ids[i]}
                if payloads:
                    base_payload.update(payloads[i])
                points.append(
                    PointStruct(
                        id=uid,
                        vector=vectors[i].tolist(),
                        payload=base_payload,
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"[VectorStore] Upserted {n} vectors into '{self.collection_name}'")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_payload: dict = None,
    ) -> list[tuple]:
        """
        Cosine similarity search in Qdrant.

        Parameters
        ----------
        query_vector : np.ndarray  shape (dim,)
        top_k : int
        filter_payload : dict, optional
            e.g. {"genre": "sci-fi"} — filters on stored payload fields.

        Returns
        -------
        list of (item_id, score) tuples, sorted descending.
        """
        qdrant_filter = None
        if filter_payload:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_payload.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results: list[ScoredPoint] = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.astype(np.float32).tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            (self._reverse_map.get(r.id, r.payload.get("item_id", r.id)), r.score)
            for r in results
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self.client.count(self.collection_name).count

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._id_map.clear()
        self._reverse_map.clear()

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"VectorStore(qdrant, collection='{self.collection_name}', dim={self.dim}, n={self.count()})"

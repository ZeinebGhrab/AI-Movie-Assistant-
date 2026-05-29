"""
vectordb/qdrant_store.py
-------------------------
Qdrant backend — connects to a running Qdrant instance (local Docker or cloud).

Install:
    pip install qdrant-client

Start Qdrant with Docker (persistent volume):
    docker run -d --name qdrant \
        -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_data:/qdrant/storage \
        qdrant/qdrant

Or via docker-compose (see docker-compose.yml at project root).

Usage:
    store = QdrantStore(collection="movies", host="localhost", port=6333, dim=384)
"""

import numpy as np
from typing import Optional

from vectordb.base import BaseVectorStore


class QdrantStore(BaseVectorStore):
    """
    Vector store backed by Qdrant (gRPC + REST, persistent via Docker volume).

    Parameters
    ----------
    collection : str
        Qdrant collection name.
    host : str
        Qdrant host (default: localhost).
    port : int
        Qdrant REST port (default: 6333).
    dim : int
        Vector dimensionality — MUST match the embedding model output.
    distance : str
        "Cosine" | "Euclid" | "Dot"  (Qdrant naming convention).
    recreate : bool
        If True, drop and recreate the collection on init (fresh start).
        If False (default), reuse existing collection if it exists.
    """

    def __init__(
        self,
        collection: str = "recsys",
        host: str = "localhost",
        port: int = 6333,
        dim: int = 384,
        distance: str = "Cosine",
        recreate: bool = False,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is not installed.\n"
                "Run:  pip install qdrant-client"
            )

        from qdrant_client.models import Distance, VectorParams

        self.collection_name = collection
        self.dim = dim
        self._distance_str = distance

        self._client = QdrantClient(host=host, port=port)
        print(f"[QdrantStore] Connected to Qdrant at {host}:{port}")

        # Map string → Qdrant Distance enum
        dist_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot":    Distance.DOT,
        }
        qdrant_distance = dist_map.get(distance, Distance.COSINE)

        existing = [c.name for c in self._client.get_collections().collections]

        if recreate and collection in existing:
            self._client.delete_collection(collection)
            existing.remove(collection)
            print(f"[QdrantStore] Collection '{collection}' dropped (recreate=True).")

        if collection not in existing:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=qdrant_distance),
            )
            print(f"[QdrantStore] Collection '{collection}' created (dim={dim}, dist={distance}).")
        else:
            info = self._client.get_collection(collection)
            n = info.points_count
            print(f"[QdrantStore] Reusing collection '{collection}'. Existing points: {n}")

    # ── Write ────────────────────────────────────────────────────────────────

    def add(
        self,
        item_id: str,
        vector: np.ndarray,
        metadata: dict | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        # Qdrant point IDs must be unsigned integers or UUIDs.
        # We store the original string ID in payload for retrieval.
        numeric_id = self._to_numeric_id(item_id)
        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=numeric_id,
                    vector=vector.tolist(),
                    payload={"original_id": str(item_id), **(metadata or {})},
                )
            ],
        )

    def add_batch(
        self,
        item_ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict] | None = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for i, (item_id, vec) in enumerate(zip(item_ids, vectors)):
            meta = (metadatas[i] if metadatas else {}) or {}
            points.append(
                PointStruct(
                    id=self._to_numeric_id(item_id),
                    vector=vec.tolist(),
                    payload={"original_id": str(item_id), **meta},
                )
            )
        # Qdrant recommends batches of ≤ 100 for stability
        batch_size = 100
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

    # ── Read ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Parameters
        ----------
        filters : dict or None
            Qdrant filter dict, e.g. {"must": [{"key": "genre", "match": {"value": "sci-fi"}}]}.
            Converted to a qdrant_client Filter object internally.

        Returns
        -------
        list of (original_item_id, score).
        """
        from qdrant_client.models import Filter

        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(**filters)

        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            (hit.payload.get("original_id", str(hit.id)), hit.score)
            for hit in results
        ]

    def count(self) -> int:
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        dist_map = {"Cosine": Distance.COSINE, "Euclid": Distance.EUCLID, "Dot": Distance.DOT}
        self._client.delete_collection(self.collection_name)
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dim, distance=dist_map.get(self._distance_str, Distance.COSINE)
            ),
        )
        print(f"[QdrantStore] Collection '{self.collection_name}' reset.")

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_numeric_id(item_id) -> int:
        """
        Convert any item_id to a positive integer for Qdrant.
        Uses Python's built-in hash, masked to 63 bits to stay positive.
        The original string ID is preserved in the payload.
        """
        return abs(hash(str(item_id))) % (2**63)

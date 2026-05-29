"""
vectordb/base.py
-----------------
Abstract base class that every vector store backend must implement.
This interface is intentionally minimal so backends (ChromaDB, Qdrant,
or the in-memory fallback) are drop-in replacements for each other.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseVectorStore(ABC):
    """
    Minimal interface for a vector store.

    All IDs are expected to be strings.  Callers should str()-cast integer IDs
    before calling add / search, and cast back on the way out if needed.
    """

    # ── Write ────────────────────────────────────────────────────────────────

    @abstractmethod
    def add(self, item_id: str, vector: np.ndarray, metadata: dict | None = None) -> None:
        """Insert or upsert a single vector with optional metadata."""

    @abstractmethod
    def add_batch(
        self,
        item_ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict] | None = None,
    ) -> None:
        """Batch insert / upsert."""

    # ── Read ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return top_k (item_id, score) pairs sorted by descending similarity.
        `filters` is backend-specific metadata filtering (optional).
        """

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors currently stored."""

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """Delete all vectors (useful between experiments)."""

    # ── Dunder helpers ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.count()})"

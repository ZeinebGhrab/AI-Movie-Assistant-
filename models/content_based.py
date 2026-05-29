"""
models/content_based.py
------------------------
Content-Based Filtering — Qdrant-backed.

Each movie description is encoded → stored in Qdrant collection "movies".
Genre is stored in the payload for optional server-side filtering.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_generator import EmbeddingGenerator
from utils.vector_store import VectorStore


class ContentBasedRecommender:
    """
    Parameters
    ----------
    model_name : str        SentenceTransformer model.
    min_rating : float      Minimum rating to consider a movie "liked".
    qdrant_host : str       Qdrant host (default localhost).
    qdrant_port : int       Qdrant REST port (default 6333).
    recreate : bool         Recreate Qdrant collection on fit().
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_rating: float = 4.0,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        recreate: bool = True,
    ):
        self.encoder = EmbeddingGenerator(model_name)
        self.min_rating = min_rating
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.recreate = recreate

        self.store: VectorStore = None
        self.movies_df: pd.DataFrame = None
        self.interactions_df: pd.DataFrame = None

    def fit(self, movies_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        self.movies_df = movies_df.reset_index(drop=True)
        self.interactions_df = interactions_df

        self.store = VectorStore(
            collection_name="recsys_movies",
            dim=self.encoder.dim,
            host=self.qdrant_host,
            port=self.qdrant_port,
            recreate=self.recreate,
        )

        print("[ContentBased] Encoding movie descriptions …")
        descriptions = self.movies_df["description"].tolist()
        embeddings = self.encoder.encode_batch(descriptions, show_progress=True)

        # Store genre in payload for server-side filtering
        payloads = [
            {"genre": row.get("genre", "unknown"), "title": row.get("title", "")}
            for _, row in self.movies_df.iterrows()
        ]

        self.store.add_batch(
            item_ids=self.movies_df["movie_id"].tolist(),
            vectors=embeddings,
            payloads=payloads,
        )

        # Keep embeddings in memory for user profile computation
        self.movies_df = self.movies_df.copy()
        self.movies_df["embedding"] = list(embeddings)
        print(f"[ContentBased] {len(self.store)} movies indexed in Qdrant.")

    def _user_profile(self, user_id) -> np.ndarray:
        liked = self.interactions_df[
            (self.interactions_df["user_id"] == user_id)
            & (self.interactions_df["rating"] >= self.min_rating)
        ]["movie_id"].tolist()

        if not liked:
            raise ValueError(f"User {user_id} has no liked movies (rating >= {self.min_rating}).")

        liked_embeddings = self.movies_df[
            self.movies_df["movie_id"].isin(liked)
        ]["embedding"].tolist()

        return np.mean(liked_embeddings, axis=0)

    def recommend(
        self,
        user_id,
        top_k: int = 5,
        exclude_seen: bool = True,
        genre_filter: str = None,
    ) -> list[tuple]:
        """
        Parameters
        ----------
        genre_filter : str, optional
            If set, Qdrant filters by payload genre — server-side, no client overhead.
        """
        user_vec = self._user_profile(user_id)

        seen = set()
        if exclude_seen:
            seen = set(
                self.interactions_df[self.interactions_df["user_id"] == user_id][
                    "movie_id"
                ].tolist()
            )

        filter_payload = {"genre": genre_filter} if genre_filter else None

        candidates = self.store.search(
            query_vector=user_vec,
            top_k=top_k + len(seen) + 5,
            filter_payload=filter_payload,
        )

        results = [(mid, score) for mid, score in candidates if mid not in seen]
        return results[:top_k]

    def get_movie_title(self, movie_id) -> str:
        row = self.movies_df[self.movies_df["movie_id"] == movie_id]
        return row.iloc[0]["title"] if not row.empty else str(movie_id)

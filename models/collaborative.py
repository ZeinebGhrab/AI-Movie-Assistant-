"""
models/collaborative.py
------------------------
User-based Collaborative Filtering — Qdrant-backed.

User embeddings are stored in a dedicated Qdrant collection "users".
Nearest-neighbour search finds similar users → aggregate their liked items.
"""

import numpy as np
import pandas as pd
import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.embedding_generator import EmbeddingGenerator
from utils.vector_store import VectorStore


class CollaborativeRecommender:
    """
    Parameters
    ----------
    model_name : str        SentenceTransformer model.
    min_rating : float      Minimum rating to consider "liked".
    n_neighbours : int      Number of similar users to consult.
    qdrant_host : str
    qdrant_port : int
    recreate : bool
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_rating: float = 4.0,
        n_neighbours: int = 3,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        recreate: bool = True,
    ):
        self.encoder = EmbeddingGenerator(model_name)
        self.min_rating = min_rating
        self.n_neighbours = n_neighbours
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.recreate = recreate

        self.user_store: VectorStore = None
        self.movies_df: pd.DataFrame = None
        self.interactions_df: pd.DataFrame = None
        self._movie_embeddings: dict = {}   # movie_id → np.ndarray
        self._user_embeddings: dict = {}    # user_id → np.ndarray

    def fit(self, movies_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        self.movies_df = movies_df.reset_index(drop=True)
        self.interactions_df = interactions_df

        # --- Encode movies ---
        print("[Collaborative] Encoding movies …")
        descriptions = self.movies_df["description"].tolist()
        movie_ids = self.movies_df["movie_id"].tolist()
        embeddings = self.encoder.encode_batch(descriptions, show_progress=True)
        self._movie_embeddings = {mid: emb for mid, emb in zip(movie_ids, embeddings)}

        # --- Build user profiles ---
        print("[Collaborative] Building user profiles …")
        user_ids, user_vecs, user_payloads = [], [], []

        for user_id in interactions_df["user_id"].unique():
            liked = interactions_df[
                (interactions_df["user_id"] == user_id)
                & (interactions_df["rating"] >= self.min_rating)
            ]["movie_id"].tolist()

            liked_vecs = [
                self._movie_embeddings[mid]
                for mid in liked
                if mid in self._movie_embeddings
            ]
            if not liked_vecs:
                continue

            user_vec = np.mean(liked_vecs, axis=0)
            self._user_embeddings[user_id] = user_vec
            user_ids.append(user_id)
            user_vecs.append(user_vec)
            user_payloads.append({"user_id": int(user_id)})

        # --- Index user vectors in Qdrant ---
        self.user_store = VectorStore(
            collection_name="recsys_users",
            dim=self.encoder.dim,
            host=self.qdrant_host,
            port=self.qdrant_port,
            recreate=self.recreate,
        )
        self.user_store.add_batch(
            item_ids=user_ids,
            vectors=np.array(user_vecs),
            payloads=user_payloads,
        )
        print(f"[Collaborative] {len(self.user_store)} user profiles indexed in Qdrant.")

    def recommend(
        self,
        user_id,
        top_k: int = 5,
        exclude_seen: bool = True,
    ) -> list[tuple]:
        if user_id not in self._user_embeddings:
            raise ValueError(f"User {user_id} not found. Did you call fit()?")

        user_vec = self._user_embeddings[user_id]

        # Find neighbours (fetch n+1 to exclude self)
        neighbours = self.user_store.search(user_vec, top_k=self.n_neighbours + 1)
        neighbours = [(uid, score) for uid, score in neighbours if uid != user_id]

        seen = set()
        if exclude_seen:
            seen = set(
                self.interactions_df[self.interactions_df["user_id"] == user_id][
                    "movie_id"
                ].tolist()
            )

        # Weighted aggregation: score = Σ similarity * (rating / 5)
        item_scores: dict = defaultdict(float)
        for neighbour_id, similarity in neighbours:
            nb_liked = self.interactions_df[
                (self.interactions_df["user_id"] == neighbour_id)
                & (self.interactions_df["rating"] >= self.min_rating)
            ][["movie_id", "rating"]].values

            for movie_id, rating in nb_liked:
                if movie_id not in seen:
                    item_scores[movie_id] += similarity * (rating / 5.0)

        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    def get_movie_title(self, movie_id) -> str:
        row = self.movies_df[self.movies_df["movie_id"] == movie_id]
        return row.iloc[0]["title"] if not row.empty else str(movie_id)

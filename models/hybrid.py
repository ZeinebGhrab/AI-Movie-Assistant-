"""
models/hybrid.py
-----------------
Hybrid Recommender — blends Content-Based + Collaborative scores.

Both sub-recommenders share the same Qdrant instance but use
different collections: "recsys_movies" and "recsys_users".

Score formula:
    hybrid = alpha * collab_score + (1 - alpha) * content_score

    alpha = 1.0  → pure collaborative
    alpha = 0.0  → pure content-based
    alpha = 0.5  → equal blend (default)
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Parameters
    ----------
    alpha : float           Weight for collaborative score ∈ [0, 1].
    model_name : str
    min_rating : float
    n_neighbours : int
    qdrant_host : str
    qdrant_port : int
    recreate : bool
    """

    def __init__(
        self,
        alpha: float = 0.5,
        model_name: str = "all-MiniLM-L6-v2",
        min_rating: float = 4.0,
        n_neighbours: int = 3,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        recreate: bool = True,
    ):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]."
        self.alpha = alpha

        shared = dict(
            model_name=model_name,
            min_rating=min_rating,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            recreate=recreate,
        )

        self.content_rec = ContentBasedRecommender(**shared)
        self.collab_rec = CollaborativeRecommender(n_neighbours=n_neighbours, **shared)

        self.movies_df: pd.DataFrame = None
        self.interactions_df: pd.DataFrame = None

    def fit(self, movies_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
        self.movies_df = movies_df
        self.interactions_df = interactions_df

        print(f"\n{'='*52}")
        print("  [Hybrid] Fitting Content-Based Recommender …")
        print(f"{'='*52}")
        self.content_rec.fit(movies_df, interactions_df)

        print(f"\n{'='*52}")
        print("  [Hybrid] Fitting Collaborative Recommender …")
        print(f"{'='*52}")
        self.collab_rec.fit(movies_df, interactions_df)
        print()

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
            Qdrant server-side payload filter — restricts content-based candidates.
        """
        n_cand = top_k * 4

        # --- Content-based scores (with optional Qdrant payload filter) ---
        content_results = self.content_rec.recommend(
            user_id,
            top_k=n_cand,
            exclude_seen=exclude_seen,
            genre_filter=genre_filter,
        )
        content_scores = {mid: s for mid, s in content_results}

        # --- Collaborative scores ---
        try:
            collab_results = self.collab_rec.recommend(
                user_id, top_k=n_cand, exclude_seen=exclude_seen
            )
            collab_scores = {mid: s for mid, s in collab_results}
        except ValueError:
            print(f"[Hybrid] User {user_id} not in collaborative index → content-only fallback.")
            collab_scores = {}

        # --- Min-max normalize to [0, 1] ---
        def _norm(d: dict) -> dict:
            if not d:
                return d
            mx = max(d.values()) + 1e-10
            return {k: v / mx for k, v in d.items()}

        content_scores = _norm(content_scores)
        collab_scores = _norm(collab_scores)

        # --- Merge & score ---
        all_candidates = set(content_scores) | set(collab_scores)
        hybrid_scores = {
            mid: self.alpha * collab_scores.get(mid, 0.0)
                 + (1 - self.alpha) * content_scores.get(mid, 0.0)
            for mid in all_candidates
        }

        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def get_movie_title(self, movie_id) -> str:
        row = self.movies_df[self.movies_df["movie_id"] == movie_id]
        return row.iloc[0]["title"] if not row.empty else str(movie_id)

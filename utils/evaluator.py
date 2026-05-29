"""
utils/evaluator.py
-------------------
Evaluation metrics for recommendation systems:
  - Precision@K
  - Recall@K
  - NDCG@K (Normalized Discounted Cumulative Gain)
"""

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of the top-K recommendations that are relevant.

    Parameters
    ----------
    recommended : list
        Ordered list of recommended item IDs (top-K first).
    relevant : set
        Set of ground-truth relevant item IDs.
    k : int
        Cut-off.

    Returns
    -------
    float in [0, 1]
    """
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fraction of relevant items found in the top-K recommendations.

    Returns
    -------
    float in [0, 1]
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            score += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
    return score


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    Penalizes relevant items placed lower in the ranking.

    Returns
    -------
    float in [0, 1]
    """
    ideal = sorted(relevant, key=lambda x: x in relevant, reverse=True)
    idcg = dcg_at_k(list(relevant)[:k], relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg


def evaluate(recommender, interactions_df, movies_df, k: int = 5) -> dict:
    """
    Run evaluation across all users with a leave-one-out split.

    Parameters
    ----------
    recommender : object
        Must implement .recommend(user_id, top_k) -> list of movie_ids.
    interactions_df : pd.DataFrame
        Columns: user_id, movie_id, rating.
    movies_df : pd.DataFrame
        Columns: movie_id, ...
    k : int
        Evaluation cut-off.

    Returns
    -------
    dict with mean Precision@K, Recall@K, NDCG@K.
    """
    precisions, recalls, ndcgs = [], [], []

    for user_id in interactions_df["user_id"].unique():
        user_items = interactions_df[
            (interactions_df["user_id"] == user_id) & (interactions_df["rating"] >= 4)
        ]["movie_id"].tolist()

        if len(user_items) < 2:
            continue

        # Leave-one-out: hold out the last interaction as ground truth
        relevant = {user_items[-1]}

        recs = recommender.recommend(user_id=user_id, top_k=k)
        rec_ids = [r[0] for r in recs]

        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

    return {
        f"Precision@{k}": round(np.mean(precisions), 4) if precisions else 0.0,
        f"Recall@{k}": round(np.mean(recalls), 4) if recalls else 0.0,
        f"NDCG@{k}": round(np.mean(ndcgs), 4) if ndcgs else 0.0,
    }

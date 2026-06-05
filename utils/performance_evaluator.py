"""
utils/performance_evaluator.py
--------------------------------
Lesson 6 — Evaluating and Tuning Vector Search Performance.

Covers:
  - Query Latency (ms per query)
  - Throughput / QPS (Queries Per Second)
  - Recall@K  (proportion of true neighbours recovered)
  - Precision@K and NDCG@K at a given latency budget
  - F1-Score (harmonic mean of Precision and Recall)
  - Full benchmark report across multiple top_k values

Usage (standalone):
    python utils/performance_evaluator.py

Usage (from code):
    from utils.performance_evaluator import PerformanceEvaluator
    pe = PerformanceEvaluator(vector_store, ground_truth_fn)
    report = pe.run(queries, top_k_values=[5, 10, 20])
    pe.print_report(report)

Requirements:
    - A fitted VectorStore instance (utils/vector_store.py)
    - A callable that returns the ground-truth relevant item IDs for a query vector
"""

import time
import statistics
import numpy as np
import math
from typing import Callable


# ---------------------------------------------------------------------------
# Individual metric helpers  (mirrors evaluator.py but with timing awareness)
# ---------------------------------------------------------------------------

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of top-K recommendations that are relevant."""
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Fraction of relevant items recovered in the top-K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def dcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            score += 1.0 / math.log2(i + 2)
    return score


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Normalised DCG at K — penalises relevant items ranked lower."""
    idcg = dcg_at_k(list(relevant)[:k], relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg


# ---------------------------------------------------------------------------
# PerformanceEvaluator
# ---------------------------------------------------------------------------

class PerformanceEvaluator:
    """
    Benchmarks a VectorStore instance across accuracy *and* speed dimensions.

    Parameters
    ----------
    vector_store : VectorStore
        A fitted VectorStore (utils/vector_store.py).
    ground_truth_fn : Callable[[np.ndarray], set]
        Given a query vector, returns the set of relevant item IDs.
        For recommendation evaluation this is typically the set of
        items a user has positively rated that are NOT in the index query.
    n_warmup : int
        Number of warm-up queries executed before timing starts.
        Warm-up lets Qdrant's HNSW graph cache settle, giving stable numbers.
    """

    def __init__(
        self,
        vector_store,
        ground_truth_fn: Callable[[np.ndarray], set],
        n_warmup: int = 3,
    ):
        self.store = vector_store
        self.ground_truth_fn = ground_truth_fn
        self.n_warmup = n_warmup

    # ------------------------------------------------------------------
    # Core benchmark loop
    # ------------------------------------------------------------------

    def benchmark(
        self,
        query_vectors: list[np.ndarray],
        top_k: int = 5,
        filter_payload: dict = None,
    ) -> dict:
        """
        Run a single benchmark at a fixed top_k.

        Returns
        -------
        dict with keys:
            top_k, n_queries,
            latency_mean_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms,
            qps,
            precision_mean, recall_mean, f1_mean, ndcg_mean
        """
        # Warm-up queries (not timed)
        for vec in query_vectors[: self.n_warmup]:
            self.store.search(vec, top_k=top_k, filter_payload=filter_payload)

        latencies_ms = []
        precisions, recalls, f1s, ndcgs = [], [], [], []

        for query_vec in query_vectors:
            relevant = self.ground_truth_fn(query_vec)

            t0 = time.perf_counter()
            results = self.store.search(query_vec, top_k=top_k, filter_payload=filter_payload)
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1_000)

            rec_ids = [item_id for item_id, _ in results]
            precisions.append(precision_at_k(rec_ids, relevant, top_k))
            recalls.append(recall_at_k(rec_ids, relevant, top_k))
            f1s.append(f1_at_k(rec_ids, relevant, top_k))
            ndcgs.append(ndcg_at_k(rec_ids, relevant, top_k))

        total_time_s = sum(latencies_ms) / 1_000
        qps = len(query_vectors) / total_time_s if total_time_s > 0 else float("inf")

        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)

        return {
            "top_k":          top_k,
            "n_queries":      n,
            "latency_mean_ms": round(statistics.mean(latencies_ms), 3),
            "latency_p50_ms":  round(sorted_lat[int(n * 0.50)], 3),
            "latency_p95_ms":  round(sorted_lat[min(int(n * 0.95), n - 1)], 3),
            "latency_p99_ms":  round(sorted_lat[min(int(n * 0.99), n - 1)], 3),
            "qps":             round(qps, 1),
            "precision_mean":  round(statistics.mean(precisions), 4),
            "recall_mean":     round(statistics.mean(recalls), 4),
            "f1_mean":         round(statistics.mean(f1s), 4),
            "ndcg_mean":       round(statistics.mean(ndcgs), 4),
        }

    # ------------------------------------------------------------------
    # Multi-K report
    # ------------------------------------------------------------------

    def run(
        self,
        query_vectors: list[np.ndarray],
        top_k_values: list[int] = (1, 5, 10, 20),
        filter_payload: dict = None,
    ) -> list[dict]:
        """
        Run benchmarks for each value in top_k_values.

        Returns
        -------
        list of benchmark dicts, one per top_k value.
        """
        report = []
        for k in top_k_values:
            print(f"  Benchmarking top_k={k} over {len(query_vectors)} queries …", end=" ")
            result = self.benchmark(query_vectors, top_k=k, filter_payload=filter_payload)
            report.append(result)
            print(f"done  (latency_mean={result['latency_mean_ms']} ms, "
                  f"recall={result['recall_mean']}, qps={result['qps']})")
        return report

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: list[dict]) -> None:
        """Print a formatted table of benchmark results."""
        header = (
            f"{'K':>4}  {'Queries':>7}  "
            f"{'Lat mean':>9}  {'P50':>7}  {'P95':>7}  {'P99':>7}  "
            f"{'QPS':>7}  "
            f"{'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'NDCG':>6}"
        )
        sep = "─" * len(header)

        print()
        print("  Vector Search Performance Report")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")

        for r in report:
            print(
                f"  {r['top_k']:>4}  "
                f"{r['n_queries']:>7}  "
                f"{r['latency_mean_ms']:>8.3f}ms  "
                f"{r['latency_p50_ms']:>6.3f}ms  "
                f"{r['latency_p95_ms']:>6.3f}ms  "
                f"{r['latency_p99_ms']:>6.3f}ms  "
                f"{r['qps']:>7.1f}  "
                f"{r['precision_mean']:>6.4f}  "
                f"{r['recall_mean']:>6.4f}  "
                f"{r['f1_mean']:>6.4f}  "
                f"{r['ndcg_mean']:>6.4f}"
            )

        print(f"  {sep}")
        print()

        # Interpretation hints (Lesson 6 guidance)
        recalls = [r["recall_mean"] for r in report]
        lats = [r["latency_mean_ms"] for r in report]

        best_recall_k = report[recalls.index(max(recalls))]["top_k"]
        fastest_k = report[lats.index(min(lats))]["top_k"]

        print("  Interpretation")
        print(f"  • Best recall achieved at top_k={best_recall_k}  "
              f"(recall={max(recalls):.4f})")
        print(f"  • Lowest latency at top_k={fastest_k}  "
              f"({min(lats):.3f} ms/query)")

        # Recall / latency tradeoff warning
        if len(report) >= 2:
            recall_gain = recalls[-1] - recalls[0]
            lat_cost = lats[-1] - lats[0]
            print(f"  • Increasing K from {report[0]['top_k']} → {report[-1]['top_k']}: "
                  f"recall +{recall_gain:.4f}, latency +{lat_cost:.3f} ms  "
                  f"({'good tradeoff' if recall_gain > 0.05 else 'marginal gain'})")
        print()


# ---------------------------------------------------------------------------
# Standalone demo — wires into a live VectorStore + sample user queries
# ---------------------------------------------------------------------------

def _build_demo_ground_truth(interactions_df, movies_df, min_rating: float = 4.0):
    """
    Returns a closure: given a query vector (user profile), returns the set
    of movie IDs that user positively rated.

    In a leave-one-out setup the caller would exclude the held-out item from
    the index before querying; here we just demonstrate the API.
    """
    import pandas as pd

    # Build a mapping: user_id → set of liked movie_ids
    liked_per_user = {}
    for uid in interactions_df["user_id"].unique():
        liked = set(
            interactions_df[
                (interactions_df["user_id"] == uid)
                & (interactions_df["rating"] >= min_rating)
            ]["movie_id"].tolist()
        )
        liked_per_user[uid] = liked

    # We need to map query_vector → user. We store user_id in the closure
    # by building a list of (user_id, query_vec) pairs externally.
    # The ground_truth_fn therefore accepts user_id instead of raw vector.
    def ground_truth(user_id) -> set:
        return liked_per_user.get(user_id, set())

    return ground_truth, liked_per_user


if __name__ == "__main__":
    """
    Demo: fits ContentBasedRecommender, then benchmarks the movie VectorStore.

    Run:
        docker compose up -d          # start Qdrant + MongoDB
        python utils/performance_evaluator.py --seed
    """
    import sys, os, argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    from utils.db import MongoDataLoader
    from models.content_based import ContentBasedRecommender
    from embeddings.embedding_generator import EmbeddingGenerator

    parser = argparse.ArgumentParser(description="VecRecSys — Performance Evaluator")
    parser.add_argument("--seed",       action="store_true")
    parser.add_argument("--mongo_uri",  default="mongodb://localhost:27017")
    parser.add_argument("--mongo_db",   default="recsys")
    parser.add_argument("--qdrant_host", default="localhost")
    parser.add_argument("--qdrant_port", type=int, default=6333)
    parser.add_argument("--top_k_values", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    loader = MongoDataLoader(uri=args.mongo_uri, db_name=args.mongo_db)
    if args.seed:
        loader.seed()
    movies_df, interactions_df = loader.load()
    loader.close()

    # ── Fit recommender (indexes movies into Qdrant) ───────────────────────
    cb = ContentBasedRecommender(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        recreate=True,
    )
    cb.fit(movies_df, interactions_df)

    # ── Build user query vectors ───────────────────────────────────────────
    encoder = EmbeddingGenerator()
    user_ids = interactions_df["user_id"].unique().tolist()

    query_items = []   # list of (user_id, np.ndarray)
    for uid in user_ids:
        try:
            vec = cb._user_profile(uid)
            query_items.append((uid, vec))
        except ValueError:
            pass

    # ── Ground-truth closure ───────────────────────────────────────────────
    gt_fn, liked_map = _build_demo_ground_truth(interactions_df, movies_df)

    # We adapt the API: ground_truth_fn receives the query_vector and must
    # return a set.  We embed user_id lookup into the closure via a dict.
    vec_to_uid = {id(vec): uid for uid, vec in query_items}

    def ground_truth_for_vector(query_vec: np.ndarray) -> set:
        uid = vec_to_uid.get(id(query_vec), None)
        return liked_map.get(uid, set())

    # ── Run benchmark ──────────────────────────────────────────────────────
    pe = PerformanceEvaluator(
        vector_store=cb.store,
        ground_truth_fn=ground_truth_for_vector,
        n_warmup=2,
    )

    query_vectors = [vec for _, vec in query_items]

    print("\n" + "═" * 60)
    print("  VecRecSys — Lesson 6: Vector Search Performance Benchmark")
    print("═" * 60)
    print(f"  Collection : recsys_movies")
    print(f"  Queries    : {len(query_vectors)} user profile vectors")
    print(f"  top_k sweep: {args.top_k_values}")
    print("═" * 60 + "\n")

    report = pe.run(query_vectors, top_k_values=args.top_k_values)
    PerformanceEvaluator.print_report(report)
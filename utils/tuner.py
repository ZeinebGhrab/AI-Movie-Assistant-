"""
utils/tuner.py
---------------
Lesson 6 — Tuning Vector Search Performance.

Provides two tuning workflows:

1. HNSWTuner
   Sweeps the HNSW `ef` (efSearch) parameter by passing it to Qdrant's
   search API.  Higher ef → better recall, higher latency.

2. TopKTuner
   Sweeps top_k to show the precision / recall / latency tradeoff curve.
   Useful for choosing K before deploying a recommender.

3. FilterOverheadProfiler
   Compares search latency with and without a payload filter to quantify
   the server-side filtering cost.

Usage (standalone):
    python utils/tuner.py

Usage (from code):
    from utils.tuner import HNSWTuner, TopKTuner
    tuner = HNSWTuner(vector_store, ground_truth_fn)
    results = tuner.sweep(query_vectors, ef_values=[64, 128, 256, 512])
    tuner.print_table(results)

Theoretical background — Lesson 6 (Vector Databases Basics):
    "HNSW efSearch: Controls the search effort during query time.
     Higher values of efSearch improve recall but also increase query latency."
"""

import time
import statistics
import math
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def _recall(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def _precision(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def _ndcg(recommended: list, relevant: set, k: int) -> float:
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    if idcg == 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    return dcg / idcg


def _timed_search(store, query_vec: np.ndarray, top_k: int, filter_payload: dict = None):
    """Run a single search and return (results, latency_ms)."""
    t0 = time.perf_counter()
    results = store.search(query_vec, top_k=top_k, filter_payload=filter_payload)
    t1 = time.perf_counter()
    return results, (t1 - t0) * 1_000


def _mean_metrics(
    store,
    query_items: list[tuple],   # (query_vec, relevant_set)
    top_k: int,
    filter_payload: dict = None,
    n_warmup: int = 2,
) -> dict:
    """Run all queries and return aggregated metrics."""
    # Warm-up
    for vec, _ in query_items[:n_warmup]:
        store.search(vec, top_k=top_k, filter_payload=filter_payload)

    latencies, recalls, precisions, ndcgs = [], [], [], []
    for vec, relevant in query_items:
        results, lat_ms = _timed_search(store, vec, top_k, filter_payload)
        rec_ids = [item_id for item_id, _ in results]
        latencies.append(lat_ms)
        recalls.append(_recall(rec_ids, relevant, top_k))
        precisions.append(_precision(rec_ids, relevant, top_k))
        ndcgs.append(_ndcg(rec_ids, relevant, top_k))

    total_s = sum(latencies) / 1_000
    return {
        "n":          len(query_items),
        "lat_mean":   round(statistics.mean(latencies), 3),
        "lat_p95":    round(sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)], 3),
        "qps":        round(len(query_items) / total_s if total_s > 0 else float("inf"), 1),
        "recall":     round(statistics.mean(recalls), 4),
        "precision":  round(statistics.mean(precisions), 4),
        "ndcg":       round(statistics.mean(ndcgs), 4),
    }


# ---------------------------------------------------------------------------
# 1. HNSWTuner — sweep efSearch (accuracy vs. speed tradeoff)
# ---------------------------------------------------------------------------

class HNSWTuner:
    """
    Sweeps Qdrant's HNSW `ef` (search-time efSearch equivalent) by
    fetching progressively larger candidate lists and measuring how
    recall and latency change.

    Because Qdrant v1.x exposes efSearch via the `params` argument of
    `search()`, this tuner monkey-patches the VectorStore.search() method
    temporarily to inject the param — no schema changes needed.

    Lesson 6 reference:
        "efSearch — Controls the search effort during query time.
         Higher values improve recall but increase query latency."

    Parameters
    ----------
    vector_store : VectorStore
        A fitted VectorStore (utils/vector_store.py).
    ground_truth_fn : Callable[[np.ndarray], set]
        Given a query vector, returns the ground-truth relevant item IDs.
    """

    def __init__(self, vector_store, ground_truth_fn: Callable):
        self.store = vector_store
        self.gt_fn = ground_truth_fn

    def sweep(
        self,
        query_vectors: list[np.ndarray],
        top_k: int = 5,
        ef_values: list[int] = (32, 64, 128, 256, 512),
        filter_payload: dict = None,
    ) -> list[dict]:
        """
        For each ef value, run all queries and record recall + latency.

        Returns
        -------
        list of dicts, one per ef value, with keys:
            ef, top_k, lat_mean, lat_p95, qps, recall, precision, ndcg
        """
        from qdrant_client.models import SearchParams

        query_items = [(vec, self.gt_fn(vec)) for vec in query_vectors]
        results = []

        for ef in ef_values:
            print(f"  [HNSWTuner] ef={ef} … ", end="", flush=True)

            # Warm-up
            for vec, _ in query_items[:2]:
                self.store.client.search(
                    collection_name=self.store.collection_name,
                    query_vector=vec.astype(np.float32).tolist(),
                    limit=top_k,
                    search_params=SearchParams(hnsw_ef=ef, exact=False),
                )

            latencies, recalls, precisions, ndcgs = [], [], [], []

            for vec, relevant in query_items:
                t0 = time.perf_counter()
                raw = self.store.client.search(
                    collection_name=self.store.collection_name,
                    query_vector=vec.astype(np.float32).tolist(),
                    limit=top_k,
                    search_params=SearchParams(hnsw_ef=ef, exact=False),
                )
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1_000
                latencies.append(lat_ms)

                rec_ids = [
                    self.store._reverse_map.get(r.id, r.payload.get("item_id", r.id))
                    for r in raw
                ]
                recalls.append(_recall(rec_ids, relevant, top_k))
                precisions.append(_precision(rec_ids, relevant, top_k))
                ndcgs.append(_ndcg(rec_ids, relevant, top_k))

            n = len(latencies)
            sorted_lat = sorted(latencies)
            total_s = sum(latencies) / 1_000
            row = {
                "ef":        ef,
                "top_k":     top_k,
                "lat_mean":  round(statistics.mean(latencies), 3),
                "lat_p95":   round(sorted_lat[min(int(n * 0.95), n - 1)], 3),
                "qps":       round(n / total_s if total_s > 0 else float("inf"), 1),
                "recall":    round(statistics.mean(recalls), 4),
                "precision": round(statistics.mean(precisions), 4),
                "ndcg":      round(statistics.mean(ndcgs), 4),
            }
            print(f"recall={row['recall']:.4f}  lat_mean={row['lat_mean']:.3f}ms")
            results.append(row)

        return results

    @staticmethod
    def print_table(results: list[dict]) -> None:
        """Print a formatted HNSW ef sweep table."""
        header = (
            f"{'ef':>6}  {'K':>4}  {'Lat mean':>9}  {'P95':>7}  "
            f"{'QPS':>7}  {'Recall':>7}  {'Prec':>7}  {'NDCG':>7}"
        )
        sep = "─" * len(header)
        print()
        print("  HNSW efSearch Sweep — Recall vs. Latency Tradeoff")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")
        for r in results:
            print(
                f"  {r['ef']:>6}  {r['top_k']:>4}  "
                f"{r['lat_mean']:>8.3f}ms  {r['lat_p95']:>6.3f}ms  "
                f"{r['qps']:>7.1f}  {r['recall']:>7.4f}  "
                f"{r['precision']:>7.4f}  {r['ndcg']:>7.4f}"
            )
        print(f"  {sep}")

        # Suggest optimal ef based on recall saturation
        recalls = [r["recall"] for r in results]
        max_recall = max(recalls)
        for r in results:
            if r["recall"] >= max_recall * 0.98:
                print(
                    f"\n  Recommended ef: {r['ef']}  "
                    f"(reaches 98% of max recall={max_recall:.4f} "
                    f"at {r['lat_mean']:.3f} ms/query)"
                )
                break
        print()


# ---------------------------------------------------------------------------
# 2. TopKTuner — sweep K to show precision / recall tradeoff
# ---------------------------------------------------------------------------

class TopKTuner:
    """
    Measures how Precision@K, Recall@K, and latency change across K values.

    Lesson 6 reference:
        "Of the K items recommended, what fraction were actually relevant?
         Of all items the user actually liked, what fraction appeared in the top K?"

    Parameters
    ----------
    vector_store : VectorStore
    ground_truth_fn : Callable[[np.ndarray], set]
    """

    def __init__(self, vector_store, ground_truth_fn: Callable):
        self.store = vector_store
        self.gt_fn = ground_truth_fn

    def sweep(
        self,
        query_vectors: list[np.ndarray],
        k_values: list[int] = (1, 3, 5, 10, 20),
        filter_payload: dict = None,
    ) -> list[dict]:
        """Sweep K values and measure metrics at each cut-off."""
        query_items = [(vec, self.gt_fn(vec)) for vec in query_vectors]
        results = []

        for k in k_values:
            print(f"  [TopKTuner] K={k} … ", end="", flush=True)
            row = _mean_metrics(self.store, query_items, top_k=k, filter_payload=filter_payload)
            row["top_k"] = k
            print(f"recall={row['recall']:.4f}  precision={row['precision']:.4f}  "
                  f"lat_mean={row['lat_mean']:.3f}ms")
            results.append(row)

        return results

    @staticmethod
    def print_table(results: list[dict]) -> None:
        """Print precision / recall / latency table across K values."""
        header = (
            f"{'K':>4}  {'Lat mean':>9}  {'P95':>7}  {'QPS':>7}  "
            f"{'Recall':>7}  {'Prec':>7}  {'NDCG':>7}"
        )
        sep = "─" * len(header)
        print()
        print("  Top-K Sweep — Precision / Recall / Latency Tradeoff")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")
        for r in results:
            print(
                f"  {r['top_k']:>4}  "
                f"{r['lat_mean']:>8.3f}ms  {r['lat_p95']:>6.3f}ms  "
                f"{r['qps']:>7.1f}  {r['recall']:>7.4f}  "
                f"{r['precision']:>7.4f}  {r['ndcg']:>7.4f}"
            )
        print(f"  {sep}")

        # Precision / recall tradeoff observation
        prec = [r["precision"] for r in results]
        rec  = [r["recall"]    for r in results]
        print(
            f"\n  Note: As K increases, Recall {'increases' if rec[-1] >= rec[0] else 'varies'} "
            f"({rec[0]:.4f} → {rec[-1]:.4f}) while "
            f"Precision {'decreases' if prec[-1] <= prec[0] else 'varies'} "
            f"({prec[0]:.4f} → {prec[-1]:.4f})."
        )
        print("  This is the classic Precision–Recall tradeoff.\n")


# ---------------------------------------------------------------------------
# 3. FilterOverheadProfiler — cost of server-side payload filtering
# ---------------------------------------------------------------------------

class FilterOverheadProfiler:
    """
    Quantifies the latency overhead introduced by Qdrant server-side
    payload filtering vs. unfiltered search.

    Lesson 6 reference:
        "Efficient filtering and metadata-based queries can significantly
         improve search performance by reducing the number of vectors that
         need to be compared to the query vector."

    Parameters
    ----------
    vector_store : VectorStore
    """

    def __init__(self, vector_store):
        self.store = vector_store

    def profile(
        self,
        query_vectors: list[np.ndarray],
        top_k: int = 5,
        filter_payload: dict = None,
        n_warmup: int = 2,
    ) -> dict:
        """
        Run both filtered and unfiltered searches and compare latencies.

        Returns
        -------
        dict with:
            unfiltered_mean_ms, filtered_mean_ms, overhead_ms, overhead_pct,
            unfiltered_qps, filtered_qps
        """
        # Warm-up
        for vec in query_vectors[:n_warmup]:
            self.store.search(vec, top_k=top_k)
            if filter_payload:
                self.store.search(vec, top_k=top_k, filter_payload=filter_payload)

        unfiltered_lats, filtered_lats = [], []

        for vec in query_vectors:
            _, lat = _timed_search(self.store, vec, top_k)
            unfiltered_lats.append(lat)

            if filter_payload:
                _, lat_f = _timed_search(self.store, vec, top_k, filter_payload=filter_payload)
                filtered_lats.append(lat_f)

        uf_mean = statistics.mean(unfiltered_lats)
        f_mean  = statistics.mean(filtered_lats) if filtered_lats else None

        result = {
            "unfiltered_mean_ms": round(uf_mean, 3),
            "unfiltered_qps":     round(len(unfiltered_lats) / (sum(unfiltered_lats) / 1_000), 1),
        }
        if f_mean is not None:
            overhead = f_mean - uf_mean
            result.update({
                "filtered_mean_ms": round(f_mean, 3),
                "filtered_qps":     round(len(filtered_lats) / (sum(filtered_lats) / 1_000), 1),
                "overhead_ms":      round(overhead, 3),
                "overhead_pct":     round(overhead / uf_mean * 100, 1) if uf_mean > 0 else 0.0,
            })

        return result

    @staticmethod
    def print_result(result: dict, filter_payload: dict = None) -> None:
        print()
        print("  Filter Overhead Profiler")
        print(f"  Filter: {filter_payload}")
        print(f"  {'Unfiltered':>12}: {result['unfiltered_mean_ms']:>8.3f} ms/query  "
              f"QPS={result['unfiltered_qps']}")
        if "filtered_mean_ms" in result:
            print(f"  {'Filtered':>12}: {result['filtered_mean_ms']:>8.3f} ms/query  "
                  f"QPS={result['filtered_qps']}")
            overhead = result.get("overhead_ms", 0)
            pct = result.get("overhead_pct", 0)
            status = "acceptable" if pct < 20 else "high — consider indexing this field"
            print(f"  Overhead    : {overhead:+.3f} ms  ({pct:+.1f}%)  → {status}")
        print()


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os, argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)

    from utils.db import MongoDataLoader
    from models.content_based import ContentBasedRecommender

    parser = argparse.ArgumentParser(description="VecRecSys — Tuner")
    parser.add_argument("--seed",         action="store_true")
    parser.add_argument("--mongo_uri",    default="mongodb://localhost:27017")
    parser.add_argument("--mongo_db",     default="recsys")
    parser.add_argument("--qdrant_host",  default="localhost")
    parser.add_argument("--qdrant_port",  type=int, default=6333)
    parser.add_argument("--genre_filter", default=None,
                        help="Optional genre payload filter for FilterOverheadProfiler")
    args = parser.parse_args()

    # Load data
    loader = MongoDataLoader(uri=args.mongo_uri, db_name=args.mongo_db)
    if args.seed:
        loader.seed()
    movies_df, interactions_df = loader.load()
    loader.close()

    # Fit recommender
    cb = ContentBasedRecommender(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        recreate=True,
    )
    cb.fit(movies_df, interactions_df)

    # Build query vectors and ground truth
    uid_list = interactions_df["user_id"].unique().tolist()
    query_items = []
    for uid in uid_list:
        try:
            vec = cb._user_profile(uid)
            liked = set(
                interactions_df[
                    (interactions_df["user_id"] == uid)
                    & (interactions_df["rating"] >= 4.0)
                ]["movie_id"].tolist()
            )
            query_items.append((uid, vec, liked))
        except ValueError:
            pass

    vec_to_liked = {id(vec): liked for _, vec, liked in query_items}
    query_vectors = [vec for _, vec, _ in query_items]

    def gt_fn(query_vec: np.ndarray) -> set:
        return vec_to_liked.get(id(query_vec), set())

    print("\n" + "═" * 60)
    print("  VecRecSys — Lesson 6: Parameter Tuning")
    print("═" * 60 + "\n")

    # 1. HNSW efSearch sweep
    print("1 / 3  HNSW efSearch sweep")
    hnsw_tuner = HNSWTuner(cb.store, gt_fn)
    hnsw_results = hnsw_tuner.sweep(query_vectors, top_k=5, ef_values=[32, 64, 128, 256])
    HNSWTuner.print_table(hnsw_results)

    # 2. Top-K sweep
    print("2 / 3  Top-K sweep")
    topk_tuner = TopKTuner(cb.store, gt_fn)
    topk_results = topk_tuner.sweep(query_vectors, k_values=[1, 3, 5, 10])
    TopKTuner.print_table(topk_results)

    # 3. Filter overhead
    print("3 / 3  Filter overhead profiler")
    profiler = FilterOverheadProfiler(cb.store)
    filter_payload = {"genre": args.genre_filter} if args.genre_filter else {"genre": "sci-fi"}
    overhead = profiler.profile(query_vectors, top_k=5, filter_payload=filter_payload)
    FilterOverheadProfiler.print_result(overhead, filter_payload=filter_payload)
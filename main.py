"""
main.py
--------
CLI entry point for VecRecSys (Qdrant backend).

Usage:
    python main.py --user_id 1 --top_k 5
    python main.py --user_id 1 --mode content --genre sci-fi
    python main.py --user_id 1 --mode hybrid --alpha 0.7
    python main.py --user_id 1 --mode all --top_k 5

Qdrant must be running:
    docker compose up -d
"""

import argparse
import pandas as pd
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender


def load_data(data_dir: str):
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    interactions = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    return movies, interactions


def print_recs(title: str, recs: list, movies_df: pd.DataFrame) -> None:
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}")
    if not recs:
        print("  No recommendations found.")
        return
    for rank, (movie_id, score) in enumerate(recs, start=1):
        row = movies_df[movies_df["movie_id"] == movie_id]
        if row.empty:
            print(f"  {rank:>2}. movie_id={movie_id}  score={score:.4f}")
        else:
            r = row.iloc[0]
            print(
                f"  {rank:>2}. {r['title']:<32} ({r.get('year','?')})  "
                f"[{r.get('genre','?')}]  score={score:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="VecRecSys — Qdrant-backed Recommendation System")
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["content", "collaborative", "hybrid", "all"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid alpha: 0=content-only, 1=collaborative-only")
    parser.add_argument("--genre", type=str, default=None,
                        help="Filter recommendations by genre (Qdrant payload filter)")
    parser.add_argument("--qdrant_host", type=str, default="localhost")
    parser.add_argument("--qdrant_port", type=int, default=6333)
    parser.add_argument("--data_dir", type=str, default=os.path.join(BASE_DIR, "data"))
    args = parser.parse_args()

    print("\n" + "═" * 52)
    print("       VecRecSys  —  Qdrant Backend")
    print("═" * 52)
    print(f"  User     : {args.user_id}")
    print(f"  Mode     : {args.mode}")
    print(f"  Top-K    : {args.top_k}")
    print(f"  Qdrant   : {args.qdrant_host}:{args.qdrant_port}")
    if args.genre:
        print(f"  Genre    : {args.genre}")
    if args.mode in ("hybrid", "all"):
        print(f"  Alpha    : {args.alpha}")
    print("═" * 52)

    movies_df, interactions_df = load_data(args.data_dir)
    qdrant_cfg = dict(qdrant_host=args.qdrant_host, qdrant_port=args.qdrant_port)

    if args.mode in ("content", "all"):
        cb = ContentBasedRecommender(**qdrant_cfg)
        cb.fit(movies_df, interactions_df)
        recs = cb.recommend(args.user_id, top_k=args.top_k, genre_filter=args.genre)
        print_recs("Content-Based Filtering", recs, movies_df)

    if args.mode in ("collaborative", "all"):
        col = CollaborativeRecommender(**qdrant_cfg)
        col.fit(movies_df, interactions_df)
        recs = col.recommend(args.user_id, top_k=args.top_k)
        print_recs("Collaborative Filtering", recs, movies_df)

    if args.mode in ("hybrid", "all"):
        hyb = HybridRecommender(alpha=args.alpha, **qdrant_cfg)
        hyb.fit(movies_df, interactions_df)
        recs = hyb.recommend(args.user_id, top_k=args.top_k, genre_filter=args.genre)
        print_recs(f"Hybrid  (α={args.alpha})", recs, movies_df)

    print("\n" + "═" * 52 + "\n")


if __name__ == "__main__":
    main()

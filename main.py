"""
main.py
--------
CLI entry point for VecRecSys (Qdrant + MongoDB backend).

Usage:
    python main.py --user_id 1 --top_k 5
    python main.py --user_id 1 --mode content --genre sci-fi
    python main.py --user_id 1 --mode hybrid --alpha 0.7
    python main.py --user_id 1 --mode all --top_k 5

Requirements:
    - Qdrant running:  docker compose up -d
    - MongoDB running: docker compose up -d  (or local mongod)
    - First run:       pass --seed to populate MongoDB
"""

import argparse
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.db import MongoDataLoader
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender


def print_recs(title: str, recs: list, movies_df) -> None:
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
    parser = argparse.ArgumentParser(
        description="VecRecSys — Qdrant + MongoDB Recommendation System"
    )
    parser.add_argument("--user_id",     type=int,   required=True)
    parser.add_argument("--mode",        type=str,   default="all",
                        choices=["content", "collaborative", "hybrid", "all"])
    parser.add_argument("--top_k",       type=int,   default=5)
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--genre",       type=str,   default=None)
    parser.add_argument("--qdrant_host", type=str,   default="localhost")
    parser.add_argument("--qdrant_port", type=int,   default=6333)
    parser.add_argument("--mongo_uri",   type=str,   default="mongodb://localhost:27017",
                        help="MongoDB connection URI")
    parser.add_argument("--mongo_db",    type=str,   default="recsys",
                        help="MongoDB database name")
    parser.add_argument("--seed",        action="store_true",
                        help="Seed MongoDB with sample data before running")
    parser.add_argument("--seed_force",  action="store_true",
                        help="Drop and re-seed MongoDB collections")
    args = parser.parse_args()

    print("\n" + "═" * 52)
    print("       VecRecSys  —  Qdrant + MongoDB")
    print("═" * 52)
    print(f"  User     : {args.user_id}")
    print(f"  Mode     : {args.mode}")
    print(f"  Top-K    : {args.top_k}")
    print(f"  Qdrant   : {args.qdrant_host}:{args.qdrant_port}")
    print(f"  MongoDB  : {args.mongo_uri} / {args.mongo_db}")
    if args.genre:
        print(f"  Genre    : {args.genre}")
    if args.mode in ("hybrid", "all"):
        print(f"  Alpha    : {args.alpha}")
    print("═" * 52)

    # --- Load data from MongoDB ---
    loader = MongoDataLoader(uri=args.mongo_uri, db_name=args.mongo_db)

    if args.seed or args.seed_force:
        loader.seed(force=args.seed_force)

    movies_df, interactions_df = loader.load()
    loader.close()

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
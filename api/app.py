"""
api/app.py
-----------
FastAPI backend — exposes VecRecSys recommendation endpoints.

Endpoints:
    GET  /api/health
    GET  /api/movies              → list all movies
    GET  /api/users               → list all users
    GET  /api/users/{user_id}/ratings   → user's rated movies
    POST /api/recommend           → get recommendations
    POST /api/evaluate            → run offline evaluation
    GET  /api/benchmark           → performance metrics

Start:
    uvicorn api.app:app --reload --port 8000
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time

from utils.db import MongoDataLoader
from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.evaluator import evaluate

# ---------------------------------------------------------------------------
app = FastAPI(
    title="VecRecSys API",
    description="Vector-powered movie recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------
loader: MongoDataLoader = None
movies_df = None
interactions_df = None
cb_rec: ContentBasedRecommender = None
col_rec: CollaborativeRecommender = None
hyb_rec: HybridRecommender = None


@app.on_event("startup")
def startup():
    global loader, movies_df, interactions_df, cb_rec, col_rec, hyb_rec

    loader = MongoDataLoader()
    loader.seed()
    movies_df, interactions_df = loader.load()

    cb_rec = ContentBasedRecommender(recreate=True)
    cb_rec.fit(movies_df, interactions_df)

    col_rec = CollaborativeRecommender(recreate=True)
    col_rec.fit(movies_df, interactions_df)

    hyb_rec = HybridRecommender(alpha=0.5, recreate=False)
    hyb_rec.fit(movies_df, interactions_df)

    print("[API] All recommenders ready.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    user_id: int
    mode: str = Field("hybrid", pattern="^(content|collaborative|hybrid)$")
    top_k: int = Field(5, ge=1, le=20)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    genre: Optional[str] = None


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    genre: str
    year: int
    description: str
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    mode: str
    recommendations: list[RecommendedMovie]
    latency_ms: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": cb_rec is not None}


@app.get("/api/movies")
def get_movies():
    return movies_df.drop(columns=["embedding"], errors="ignore").to_dict(orient="records")


@app.get("/api/users")
def get_users():
    user_ids = sorted(interactions_df["user_id"].unique().tolist())
    result = []
    for uid in user_ids:
        rated = interactions_df[interactions_df["user_id"] == uid]
        result.append({
            "user_id": uid,
            "n_ratings": len(rated),
            "avg_rating": round(rated["rating"].mean(), 2),
            "liked_genres": (
                movies_df[movies_df["movie_id"].isin(
                    rated[rated["rating"] >= 4]["movie_id"]
                )]["genre"].value_counts().to_dict()
            ),
        })
    return result


@app.get("/api/users/{user_id}/ratings")
def get_user_ratings(user_id: int):
    rated = interactions_df[interactions_df["user_id"] == user_id]
    if rated.empty:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    merged = rated.merge(
        movies_df.drop(columns=["embedding"], errors="ignore"),
        on="movie_id",
    )
    return merged.sort_values("rating", ascending=False).to_dict(orient="records")


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.perf_counter()

    try:
        if req.mode == "content":
            recs = cb_rec.recommend(req.user_id, top_k=req.top_k, genre_filter=req.genre)
        elif req.mode == "collaborative":
            recs = col_rec.recommend(req.user_id, top_k=req.top_k)
        else:
            hyb_rec.alpha = req.alpha
            recs = hyb_rec.recommend(req.user_id, top_k=req.top_k, genre_filter=req.genre)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000

    result = []
    for movie_id, score in recs:
        row = movies_df[movies_df["movie_id"] == movie_id]
        if row.empty:
            continue
        r = row.iloc[0]
        result.append(RecommendedMovie(
            movie_id=int(movie_id),
            title=r["title"],
            genre=r.get("genre", "unknown"),
            year=int(r.get("year", 0)),
            description=r.get("description", ""),
            score=round(float(score), 4),
        ))

    return RecommendResponse(
        user_id=req.user_id,
        mode=req.mode,
        recommendations=result,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/api/evaluate")
def run_evaluation(top_k: int = 5):
    results = {}
    for name, rec in [("content", cb_rec), ("collaborative", col_rec), ("hybrid", hyb_rec)]:
        results[name] = evaluate(rec, interactions_df, movies_df, k=top_k)
    return {"top_k": top_k, "results": results}


@app.get("/api/genres")
def get_genres():
    return sorted(movies_df["genre"].unique().tolist())
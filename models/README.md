# 🎬 Movie Recommendation System

> A modular, Qdrant-backed recommendation engine combining **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid blend** — all powered by SentenceTransformer embeddings.

---

## 🚀 Overview

This project implements three complementary recommendation strategies that share the same embedding backbone (`all-MiniLM-L6-v2`) and vector store (Qdrant). Each strategy addresses a different aspect of the recommendation problem; the Hybrid model combines all three signals into a single ranked output.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query (user_id)                  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│  Content-Based  │    │    Collaborative      │
│  Recommender    │    │    Recommender        │
│                 │    │                       │
│  movie desc     │    │  user profile =       │
│  → embedding    │    │  mean(liked movies)   │
│  → Qdrant ANN   │    │  → Qdrant ANN         │
│  collection:    │    │  collection:          │
│  recsys_movies  │    │  recsys_users         │
└────────┬────────┘    └──────────┬────────────┘
         │  content_score         │  collab_score
         └───────────┬────────────┘
                     ▼
         ┌───────────────────────┐
         │   Hybrid Recommender  │
         │                       │
         │  score = α·collab     │
         │        + (1-α)·content│
         └───────────────────────┘
```

---

## 🧠 Embedding Backbone

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Embedding size | 384 dimensions |
| Type | Transformer-based sentence embedding |

**Strengths:** Fast inference ⚡ · Lightweight · Strong semantic understanding

The `EmbeddingGenerator` module provides three encoding modes used across the system:

| Method | Output Shape | When it's used |
|---|---|---|
| `encode()` | `(384,)` | Single text at inference time |
| `encode_batch()` | `(N, 384)` | Offline indexing of the full movie catalog |
| `encode_with_metadata()` | `(384 + G,)` | Hybrid vector (text + one-hot genre) |

---

## 🔍 Recommendation Strategies

### 🔹 Content-Based Filtering (`models/content_based.py`)

**Core idea:** recommend movies that are semantically similar to what a user has already liked.

**How it works:**
1. `fit()` — encode all movie descriptions with `encode_batch()` → store vectors in the Qdrant collection `recsys_movies` (genre stored as payload for server-side filtering).
2. At query time, compute the **user profile** as the mean embedding of all movies the user rated ≥ `min_rating`.
3. Run an **ANN search** in Qdrant to find the nearest movie vectors to that profile.

**Key advantage:** optional `genre_filter` is applied server-side in Qdrant — no client-side overhead.

```python
rec = ContentBasedRecommender(min_rating=4.0)
rec.fit(movies_df, interactions_df)

# Top-5 Action movies for user 42
results = rec.recommend(user_id=42, top_k=5, genre_filter="Action")
```

---

### 🔹 Collaborative Filtering (`models/collaborative.py`)

**Core idea:** find users with similar taste profiles, then aggregate the movies they liked.

**How it works:**
1. `fit()` — for each user, compute their profile as the mean embedding of their liked movies → store in Qdrant collection `recsys_users`.
2. At query time, retrieve the `n_neighbours` most similar users via ANN search (excluding the query user itself).
3. Aggregate neighbour ratings using a **weighted scoring formula**:

```
score(movie) = Σ  similarity(user, neighbour) × (rating / 5.0)
              neighbours
```

**Key advantage:** captures taste patterns that don't appear in movie descriptions (e.g., niche preferences, cross-genre affinity).

```python
rec = CollaborativeRecommender(min_rating=4.0, n_neighbours=3)
rec.fit(movies_df, interactions_df)

results = rec.recommend(user_id=42, top_k=5, exclude_seen=True)
```

---

### 🔹 Hybrid Recommender (`models/hybrid.py`)

**Core idea:** blend both scores to mitigate the weaknesses of each individual approach.

**Score formula:**

```
hybrid_score = α × collab_score + (1 − α) × content_score
```

| `alpha` | Behaviour |
|---|---|
| `1.0` | Pure collaborative |
| `0.0` | Pure content-based |
| `0.5` | Equal blend (default) |

**How it works:**
1. Both sub-recommenders are fit on the same data (sharing the same Qdrant instance, different collections).
2. At query time, each returns a candidate list → scores are **min-max normalized** to `[0, 1]` before blending.
3. If the user is absent from the collaborative index (cold-start), the system gracefully falls back to content-based only.

```python
rec = HybridRecommender(alpha=0.5, min_rating=4.0, n_neighbours=3)
rec.fit(movies_df, interactions_df)

results = rec.recommend(user_id=42, top_k=5, genre_filter="Drama")
```

---

## 📦 Installation

```bash
pip install sentence-transformers numpy pandas qdrant-client
```

Qdrant must be running locally (default: `localhost:6333`). Quick start with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `min_rating` | `4.0` | Minimum rating threshold to consider a movie "liked" |
| `n_neighbours` | `3` | Number of similar users (collaborative only) |
| `alpha` | `0.5` | Blend weight — collab vs content (hybrid only) |
| `qdrant_host` | `localhost` | Qdrant host |
| `qdrant_port` | `6333` | Qdrant REST port |
| `recreate` | `True` | Recreate Qdrant collection on each `fit()` call |

---

## 📌 Design Decisions

| Decision | Rationale |
|---|---|
| Qdrant for ANN search | Scalable vector similarity at query time without loading all vectors in memory |
| Mean pooling for user profiles | Simple, effective aggregation; robust to varying number of liked movies |
| Min-max normalization before blending | Prevents one score distribution from dominating the other in hybrid mode |
| Cold-start fallback in hybrid | A user absent from the collaborative index still gets content-based results |
| Server-side genre filter | Pushes filtering into Qdrant's payload index — no wasted embedding computation |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
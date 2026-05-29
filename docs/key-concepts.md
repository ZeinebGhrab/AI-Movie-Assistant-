# VecRecSys — Key Concepts & Research Notes

> This document maps the theoretical foundations from the *Vector Databases Basics* course to the concrete components used in the **VecRecSys** project. Each section explains what a concept is, why it matters, and exactly where it appears in the codebase.

---

## Table of Contents

1. [Vector Embeddings](#1-vector-embeddings)
2. [Sentence Transformers](#2-sentence-transformers)
3. [Cosine Similarity](#3-cosine-similarity)
4. [Qdrant — Vector Database](#4-qdrant--vector-database)
5. [HNSW Index](#5-hnsw-index)
6. [Approximate Nearest Neighbor (ANN) Search](#6-approximate-nearest-neighbor-ann-search)
7. [Metadata & Payload Filtering](#7-metadata--payload-filtering)
8. [User Profile as a Mean Embedding](#8-user-profile-as-a-mean-embedding)
9. [Content-Based Filtering](#9-content-based-filtering)
10. [Collaborative Filtering](#10-collaborative-filtering)
11. [Hybrid Filtering & Score Blending](#11-hybrid-filtering--score-blending)
12. [Evaluation Metrics — Precision, Recall, NDCG](#12-evaluation-metrics--precision-recall-ndcg)

---

## 1. Vector Embeddings

### What they are

A vector embedding is a numerical representation of a piece of data (text, image, audio) as a fixed-length array of floating-point numbers. The core property is that **semantically similar items map to nearby points in the embedding space** — the closer two vectors are, the more similar the underlying data.

> *"Things that are similar will have vectors that are close together in the vector space."* — Course, Lesson 1

Each dimension of the vector captures a latent feature of the data. A 384-dimensional embedding means the model has learned 384 independent axes of meaning.

### Why we use them in VecRecSys

Traditional keyword-based matching cannot capture that *"spacecraft"* and *"astronaut"* are semantically related even when they don't share characters. Embeddings solve this: a movie description about space exploration and one about astronauts stranded in orbit will land near each other in vector space, enabling the recommender to surface one when a user likes the other.

### Where it appears in the code

| File | Role |
|------|------|
| `embeddings/embedding_generator.py` | `EmbeddingGenerator` class — wraps `SentenceTransformer` and exposes `encode()` / `encode_batch()` |
| `models/content_based.py` — `fit()` | Each movie description → 384-dim vector via `encode_batch()` |
| `models/collaborative.py` — `fit()` | Same encoding; vectors are then averaged into user profiles |

---

## 2. Sentence Transformers

### What they are

Sentence Transformers are transformer-based models (derived from BERT/RoBERTa architectures) that have been fine-tuned with a contrastive objective to produce **high-quality sentence-level embeddings**. Unlike raw BERT, which produces one embedding per token, a Sentence Transformer collapses the sequence into a single fixed-size vector that represents the whole sentence.

> *"Sentence Transformers: Models that are specifically trained to produce high-quality sentence embeddings. They are based on transformer architectures like BERT and RoBERTa."* — Course, Lesson 1

### Model used — `all-MiniLM-L6-v2`

The project uses `all-MiniLM-L6-v2`, a compact but high-quality model from the `sentence-transformers` library:

| Property | Value |
|----------|-------|
| Output dimension | **384** |
| Architecture | MiniLM (distilled BERT) |
| Training objective | Semantic textual similarity on 1B+ sentence pairs |
| Speed | Fast — good for local inference |

The 384-dim output is the value stored in Qdrant and the size of every collection (`dim=384` in `VectorStore.__init__`).

### Where it appears in the code

```python
# embeddings/embedding_generator.py
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()  # → 384
```

To swap to a higher-accuracy model, only this one line needs to change (then recreate Qdrant collections with `recreate=True`).

---

## 3. Cosine Similarity

### What it is

Cosine similarity measures the **angle** between two vectors, not their magnitude. It is defined as:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Score = **1.0** → vectors point in the same direction (identical semantic content)
- Score = **0.0** → orthogonal vectors (no shared semantic content)
- Score = **−1.0** → opposite directions (antonyms)

> *"Cosine similarity is often preferred for text embeddings, as it is less sensitive to the magnitude of the vectors."* — Course, Lesson 2

### Why cosine for this project

Movie descriptions vary in length. A short description and a long description of the same concept would have vectors of very different magnitudes. Cosine similarity normalizes for magnitude, so the *direction* of meaning matters, not the *size* of the text.

### Where it appears in the code

```python
# utils/vector_store.py — VectorStore.__init__()
from qdrant_client.models import Distance, VectorParams

self.client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)
```

`Distance.COSINE` tells Qdrant to use cosine similarity for all searches in that collection. The scores returned by `client.search()` are cosine similarity scores in `[−1, 1]`, clipped to `[0, 1]` in practice for normalized embeddings.

---

## 4. Qdrant — Vector Database

### What it is

Qdrant is an open-source, Rust-based **vector database** designed specifically for storing vector embeddings and performing fast similarity search. Unlike general-purpose databases (PostgreSQL, MongoDB), it is optimized for the high-dimensional nearest-neighbor problem.

> *"Vector databases are specialized databases designed to efficiently store, manage, and query vector embeddings."* — Course, Vector DB Fundamentals Lesson 1

### Key features used in VecRecSys

| Feature | How VecRecSys uses it |
|---------|----------------------|
| **Persistent storage** | Vectors survive process restarts via a Docker named volume (`qdrant_data`) |
| **HNSW indexing** | Qdrant uses HNSW by default — searches are O(log n) not O(n) |
| **Cosine distance** | Configured per collection via `Distance.COSINE` |
| **Payload filtering** | Genre stored in payload, filtered server-side with `FieldCondition` |
| **REST API** | Python client talks to Qdrant over HTTP on port 6333 |
| **Collections** | Two collections: `recsys_movies` and `recsys_users` |

### Collections architecture

```
Qdrant instance (localhost:6333)
├── recsys_movies   — 15 points, dim=384, payload: {genre, title, item_id}
└── recsys_users    — 5 points,  dim=384, payload: {user_id, item_id}
```

Each point = one vector + one payload dict + one integer ID.

### Where it appears in the code

```python
# utils/vector_store.py
from qdrant_client import QdrantClient
self.client = QdrantClient(host=host, port=port)
```

The `VectorStore` class is the thin wrapper around the Qdrant client. All three recommenders (`ContentBasedRecommender`, `CollaborativeRecommender`, `HybridRecommender`) instantiate it with different `collection_name` values.

### Running Qdrant

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.9.2
    ports:
      - "6333:6333"   # REST
      - "6334:6334"   # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
```

`docker compose up -d` starts Qdrant in the background. The named volume means embeddings do not need to be recomputed on every run (set `recreate=False` in `VectorStore` to reuse).

---

## 5. HNSW Index

### What it is

**Hierarchical Navigable Small World (HNSW)** is the graph-based indexing algorithm used by Qdrant. It builds a multi-layer graph where:

- The **top layers** are sparse, long-range connections for fast coarse navigation.
- The **bottom layers** are dense, short-range connections for precise local search.

Search starts at the top, greedily descends toward the query vector, then refines at the bottom layer.

> *"HNSW builds a multi-layered graph structure that enables efficient approximate nearest neighbor search. It offers a good balance between speed and accuracy."* — Course, Vector DB Fundamentals Lesson 4

### Key parameters

| Parameter | Meaning | Default in Qdrant |
|-----------|---------|-------------------|
| `M` | Neighbors per node per layer | 16 |
| `efConstruction` | Search effort during build | 100 |
| `efSearch` | Search effort during query | 128 |

Higher values = better accuracy, more memory, slower indexing. For 15 movies and 5 users (project scale), defaults are more than sufficient.

### Why it matters for VecRecSys

Without HNSW, finding the nearest movie to a user profile vector would require computing cosine similarity against every vector in the collection — O(n). With HNSW, Qdrant navigates the graph in O(log n). This is what makes the approach **scalable to millions of movies** without changing any application code.

---

## 6. Approximate Nearest Neighbor (ANN) Search

### What it is

ANN search finds vectors that are *approximately* the closest to a query vector, without comparing against every entry. It trades a small amount of accuracy (recall) for a large gain in speed.

> *"ANN search algorithms aim to find data points that are approximately the closest neighbors to a query vector, without exhaustively comparing the query to every vector in the database."* — Course, Vector DB Fundamentals Lesson 2

### Accuracy vs. Speed trade-off

| Scenario | Priority | Acceptable trade-off |
|----------|----------|----------------------|
| Real-time recommendations | Low latency | Some imprecision in results |
| Medical image retrieval | High recall | Slower queries |
| VecRecSys (movie recs) | Balanced | Good recall, instant response |

### Where it appears in the code

Qdrant's `client.search()` call performs ANN search internally via HNSW — there is no explicit ANN call in the project code. The `top_k` parameter controls how many nearest neighbors are returned:

```python
# utils/vector_store.py — search()
results = self.client.search(
    collection_name=self.collection_name,
    query_vector=query_vector.astype(np.float32).tolist(),
    limit=top_k,
    query_filter=qdrant_filter,
)
```

---

## 7. Metadata & Payload Filtering

### What it is

Vector databases allow each stored point to carry a **payload** — a JSON-like dict of metadata. Filters on this metadata are applied **server-side, inside the database**, before returning results. This avoids fetching irrelevant vectors to the client.

> *"In many applications, it's necessary to filter the search results based on metadata. For example, you might want to find similar products within a specific price range or category."* — Course, Vector DB Fundamentals Lesson 1

### How VecRecSys uses it

Each movie point in `recsys_movies` carries:
```json
{"item_id": 5, "genre": "sci-fi", "title": "Arrival"}
```

When the `--genre` CLI flag is passed, a `FieldCondition` is built and passed to Qdrant's search:

```python
# utils/vector_store.py — search()
from qdrant_client.models import Filter, FieldCondition, MatchValue

conditions = [FieldCondition(key="genre", match=MatchValue(value="sci-fi"))]
qdrant_filter = Filter(must=conditions)

results = self.client.search(..., query_filter=qdrant_filter)
```

Qdrant evaluates the filter before scoring — only sci-fi movies enter the similarity computation. This is more efficient than fetching all movies and filtering in Python.

### CLI usage

```bash
# Returns only sci-fi recommendations for user 1
python main.py --user_id 1 --mode content --genre sci-fi
```

---

## 8. User Profile as a Mean Embedding

### What it is

Since there is no single "user vector" in the training data, a user profile is **derived** by averaging the embeddings of all movies that user has positively rated (rating ≥ `min_rating`, default 4.0).

> *"Calculate user embeddings for Alice and Bob by averaging the embeddings of the movies they have liked."* — Course, Recommendation Systems Lesson

This is the **mean pooling** strategy: the centroid of a user's liked movies in embedding space becomes their taste vector.

### Why it works

If a user liked *Interstellar*, *Arrival*, and *Inception* (all semantically similar sci-fi/mind-bending films), their mean vector will point toward the region of embedding space where those concepts cluster. New movies near that region will be semantically aligned with their demonstrated preferences.

### Where it appears in the code

```python
# models/content_based.py — _user_profile()
liked_embeddings = self.movies_df[
    self.movies_df["movie_id"].isin(liked)
]["embedding"].tolist()

return np.mean(liked_embeddings, axis=0)   # shape: (384,)
```

```python
# models/collaborative.py — fit()
user_vec = np.mean(liked_vecs, axis=0)
self._user_embeddings[user_id] = user_vec
```

Both recommenders build user profiles the same way. The difference is what they do with that vector next (query the movie collection vs. the user collection).

---

## 9. Content-Based Filtering

### What it is

Content-based filtering recommends items similar to what a user has already liked, **based on item attributes alone** — here, the movie description text. It does not use data from other users.

> *"Content-based filtering recommends items that are semantically similar to what a user has already liked, based solely on item attributes."* — Course, Recommendation Systems

### Pipeline in VecRecSys

```
1. encode_batch(all movie descriptions)  →  384-dim vectors
2. upsert into recsys_movies collection
3. for target user: compute mean vector of liked movies
4. query Qdrant with user vector  →  top-k nearest movies
5. exclude movies user has already seen
```

### Strengths and limitations

| Strengths | Limitations |
|-----------|-------------|
| Works for brand-new movies (cold start for items) | Cannot discover outside the user's established taste |
| Self-contained per user — no other users' data needed | Two movies with different wording but same theme may not score as similar as expected |
| Genre filter integrates naturally via payload | No serendipity — always recommends the same region of embedding space |

### Where it appears in the code

`models/content_based.py` — `ContentBasedRecommender` class, methods `fit()` and `recommend()`.

---

## 10. Collaborative Filtering

### What it is

Collaborative filtering recommends items based on **similar users' preferences**, without examining item content. "Users who liked what you liked also liked this."

> *"Collaborative filtering recommends items based on the preferences of similar users, without looking at item content at all."* — Course, Recommendation Systems

### Pipeline in VecRecSys

```
1. encode all movie descriptions (same as content-based)
2. for each user: compute mean vector of liked movies  →  user profile
3. upsert all user profiles into recsys_users collection
4. for target user: query recsys_users  →  K nearest-neighbor users
5. aggregate their liked movies with weighted score:
       score(movie) = Σ  neighbour_similarity × (rating / 5)
6. rank and return top-k unseen movies
```

### Weighted aggregation formula

```python
# models/collaborative.py — recommend()
for neighbour_id, similarity in neighbours:
    for movie_id, rating in nb_liked:
        item_scores[movie_id] += similarity * (rating / 5.0)
```

A movie rated 5/5 by a very similar neighbour (similarity ≈ 1.0) scores ~1.0. A movie rated 4/5 by a moderately similar neighbour (similarity ≈ 0.6) scores ~0.48. This naturally down-weights both weak signals and mediocre ratings.

### Strengths and limitations

| Strengths | Limitations |
|-----------|-------------|
| Can surface unexpected discoveries | Requires interaction history for both target user and neighbours |
| Captures cross-genre trends from similar users | New users with no history cannot be served (cold start for users) |
| Does not depend on item metadata quality | Sparse interaction data weakens similarity signals |

### Where it appears in the code

`models/collaborative.py` — `CollaborativeRecommender` class, methods `fit()` and `recommend()`.

---

## 11. Hybrid Filtering & Score Blending

### What it is

The hybrid approach combines content-based and collaborative scores into a single ranked list using a configurable **α (alpha)** parameter:

```
hybrid_score = α × collaborative_score + (1 − α) × content_score
```

> *"Combine content-based and collaborative filtering by weighting the similarity scores from both methods."* — Course, Recommendation Systems Exercises

### Min-max normalization before blending

Raw scores from both recommenders live on different scales — cosine similarities from content-based vs. weighted rating aggregates from collaborative. Before blending, each score list is independently min-max normalized to `[0, 1]`:

```python
# models/hybrid.py — recommend()
def _norm(d: dict) -> dict:
    mx = max(d.values()) + 1e-10
    return {k: v / mx for k, v in d.items()}

content_scores = _norm(content_scores)
collab_scores  = _norm(collab_scores)
```

Without this step, the collaborative branch (which produces larger raw scores for popular movies) would dominate regardless of α.

### Alpha guide

| α | Behaviour | Best for |
|---|-----------|----------|
| `0.0` | Pure content-based | New users, niche tastes |
| `0.3` | Leans content-based | Users with limited history |
| `0.5` | Equal blend (default) | General use |
| `0.7` | Leans collaborative | Well-established users |
| `1.0` | Pure collaborative | Dense interaction data |

### Cold start fallback

If a user has no profile in the collaborative index (e.g. first interaction), the system logs a warning and falls back to pure content-based:

```python
try:
    collab_results = self.collab_rec.recommend(user_id, ...)
except ValueError:
    print(f"[Hybrid] User {user_id} not in collaborative index → content-only fallback.")
    collab_scores = {}
```

### Where it appears in the code

`models/hybrid.py` — `HybridRecommender` class.

---

## 12. Evaluation Metrics — Precision, Recall, NDCG

### Overview

Offline evaluation uses a **leave-one-out** strategy: the last positively rated movie per user is held out as ground truth; the system recommends top-k from the rest and measures how often the held-out item appears.

> *"Evaluate recommendations using metrics such as precision, recall, and NDCG."* — Course, Recommendation Systems Lesson 4

### Metrics

#### Precision@K

> Of the K items recommended, what fraction were actually relevant?

```python
hits = sum(1 for item in recommended[:k] if item in relevant)
precision = hits / k
```

A perfect score (1.0) means every recommendation was relevant. In practice, with one held-out item and K=5, the maximum achievable Precision@5 is 0.2.

#### Recall@K

> Of all items the user actually liked, what fraction appeared in the top K?

```python
hits = sum(1 for item in recommended[:k] if item in relevant)
recall = hits / len(relevant)
```

With one held-out item, Recall@K is either 0.0 or 1.0 — did the system recover the held-out item or not?

#### NDCG@K (Normalized Discounted Cumulative Gain)

> Measures ranking quality — a relevant item at rank 1 scores higher than the same item at rank 5.

```
DCG  = Σ  1 / log2(rank + 1)   for relevant items in top-k
NDCG = DCG / IDCG              where IDCG = best possible DCG
```

NDCG is the most informative metric for ordered recommendation lists because it penalizes relevant items pushed down the ranking.

### Where it appears in the code

`utils/evaluator.py` — functions `precision_at_k()`, `recall_at_k()`, `ndcg_at_k()`, and the orchestrator `evaluate()`.

```bash
# Run offline evaluation across all users
python utils/evaluator.py
```

---

## Summary Table — Theory → Code Mapping

| Concept (from course) | Where used in VecRecSys | File |
|-----------------------|------------------------|------|
| Vector embeddings | Movie and user vectors (384-dim) | `embeddings/embedding_generator.py` |
| Sentence Transformers | `all-MiniLM-L6-v2` model | `embeddings/embedding_generator.py` |
| Cosine similarity | `Distance.COSINE` in Qdrant collection config | `utils/vector_store.py` |
| Vector database | Qdrant — persistent, Docker-backed | `utils/vector_store.py`, `docker-compose.yml` |
| HNSW index | Default Qdrant index (automatic) | Qdrant internals |
| ANN search | `client.search()` with `limit=top_k` | `utils/vector_store.py` |
| Metadata / payload filtering | Genre filter via `FieldCondition` | `utils/vector_store.py`, `models/content_based.py` |
| Mean pooling for user profiles | `np.mean(liked_vecs, axis=0)` | `models/content_based.py`, `models/collaborative.py` |
| Content-based filtering | `ContentBasedRecommender` | `models/content_based.py` |
| Collaborative filtering | `CollaborativeRecommender` | `models/collaborative.py` |
| Hybrid blending (α-weighting) | `HybridRecommender` | `models/hybrid.py` |
| Precision@K / Recall@K / NDCG@K | `evaluate()` function | `utils/evaluator.py` |
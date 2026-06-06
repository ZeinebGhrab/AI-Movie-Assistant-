# 🎬 Movie Recommendation System

> A modular recommendation engine combining **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid blend** — powered by SentenceTransformer embeddings and a **pluggable vector store layer** (Qdrant or ChromaDB).

---

## 🚀 Overview

This project implements three complementary recommendation strategies that share the same embedding backbone (`all-MiniLM-L6-v2`) and a swappable vector store interface. Each strategy addresses a different aspect of the recommendation problem; the Hybrid model combines all three signals into a single ranked output.

The vector store layer is fully abstracted: swap between **Qdrant** (Docker-based, production-ready) and **ChromaDB** (embedded, zero-infrastructure) by changing a single constructor argument — no model code changes required.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query (user_id)                 │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│  Content-Based  │    │    Collaborative     │
│  Recommender    │    │    Recommender       │
│                 │    │                      │
│  movie desc     │    │  user profile =      │
│  → embedding    │    │  mean(liked movies)  │
│  → ANN search   │    │  → ANN search        │
│  collection:    │    │  collection:         │
│  recsys_movies  │    │  recsys_users        │
└────────┬────────┘    └──────────┬───────────┘
         │  content_score         │  collab_score
         └───────────┬────────────┘
                     ▼
         ┌───────────────────────┐
         │   Hybrid Recommender  │
         │                       │
         │  score = α·collab     │
         │        + (1-α)·content│
         └───────────────────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│   QdrantStore   │    │    ChromaStore       │
│  (Docker/cloud) │    │ (embedded, no server)│
└─────────────────┘    └──────────────────────┘
         └───────────┬────────────┘
                     ▼
            BaseVectorStore (ABC)
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

## 🗄️ Vector Store Layer

The `vectordb/` package provides a **backend-agnostic interface** so that all recommender models remain decoupled from the underlying storage engine.

### `BaseVectorStore` — the contract (`vectordb/base.py`)

Every backend must implement five methods:

| Method | Signature | Description |
|---|---|---|
| `add()` | `(item_id, vector, metadata)` | Insert or upsert a single vector |
| `add_batch()` | `(item_ids, vectors, metadatas)` | Bulk insert / upsert |
| `search()` | `(query_vector, top_k, filters)` | ANN search → `[(id, score)]` |
| `count()` | `() → int` | Number of stored vectors |
| `reset()` | `()` | Drop and recreate the collection |

`__len__` and `__repr__` are provided as free helpers on the base class.

All IDs are treated as strings at the interface level — each backend handles its own internal ID conversion (e.g., Qdrant requires unsigned integers).

---

### `QdrantStore` — production backend (`vectordb/qdrant_store.py`)

Connects to a running Qdrant instance via REST. Designed for **production workloads** where persistence, scalability, and server-side filtering matter.

**Key behaviours:**
- `recreate=True` drops and recreates the collection on init (clean experiments); `recreate=False` reuses an existing collection.
- Point IDs are hashed to 63-bit unsigned integers; the original string ID is preserved in the payload for transparent retrieval.
- Batches are chunked at 100 points per upsert call (Qdrant's recommended limit for stability).
- `filters` follows the Qdrant filter DSL:
  ```python
  filters={"must": [{"key": "genre", "match": {"value": "sci-fi"}}]}
  ```

**Supported distance metrics:** `"Cosine"` · `"Euclid"` · `"Dot"`

**Setup:**
```bash
pip install qdrant-client

docker run -d --name qdrant \
    -p 6333:6333 \
    -v $(pwd)/qdrant_data:/qdrant/storage \
    qdrant/qdrant
```

```python
from vectordb.qdrant_store import QdrantStore

store = QdrantStore(collection="movies", dim=384, recreate=True)
store.add_batch(item_ids, vectors, metadatas)
results = store.search(query_vec, top_k=5)
```

---

### `ChromaStore` — embedded backend (`vectordb/chroma_store.py`)

Runs **entirely in-process** — no server, no Docker. Data is persisted to a local SQLite + Parquet layout under `persist_dir`.

**Key behaviours:**
- `persist_dir=None` creates an ephemeral in-memory instance (useful for tests and CI).
- Uses `upsert` internally — safe to re-run `fit()` without duplicate entries.
- Distance → similarity conversion is handled transparently:
  - `cosine` → `similarity = 1 − distance`
  - `l2` → `similarity = 1 / (1 + distance)`
  - `ip` → raw inner product score
- `filters` follows the ChromaDB `where` clause syntax:
  ```python
  filters={"genre": "sci-fi"}
  ```

**Supported distance metrics:** `"cosine"` · `"l2"` · `"ip"`

**Setup:**
```bash
pip install chromadb
```

```python
from vectordb.chroma_store import ChromaStore

# Persistent (recommended)
store = ChromaStore(collection="movies", persist_dir="./chroma_data")

# In-memory (tests)
store = ChromaStore(collection="movies", persist_dir=None)
```

---

### Choosing a backend

| | QdrantStore | ChromaStore |
|---|---|---|
| Infrastructure | Docker / managed cloud | None (in-process) |
| Persistence | Docker volume | Local directory |
| Scalability | High (dedicated server) | Moderate (single process) |
| Server-side filtering | Qdrant filter DSL | ChromaDB `where` clause |
| Best for | Production, large datasets | Development, prototyping, CI |

---

## 🔍 Recommendation Strategies

### 🔹 Content-Based Filtering (`models/content_based.py`)

**Core idea:** recommend movies that are semantically similar to what a user has already liked.

**How it works:**
1. `fit()` — encode all movie descriptions with `encode_batch()` → store vectors in the collection `recsys_movies` (genre stored in metadata for server-side filtering).
2. At query time, compute the **user profile** as the mean embedding of all movies the user rated ≥ `min_rating`.
3. Run an **ANN search** to find the nearest movie vectors to that profile.

**Key advantage:** optional `genre_filter` is applied server-side — no client-side overhead.

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
1. `fit()` — for each user, compute their profile as the mean embedding of their liked movies → store in collection `recsys_users`.
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
1. Both sub-recommenders are fit on the same data (different collections in the same store).
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
# Core dependencies
pip install sentence-transformers numpy pandas

# Choose your backend (or install both)
pip install qdrant-client   # for QdrantStore
pip install chromadb        # for ChromaStore
```

---

## ⚙️ Configuration Reference

| Parameter | Default | Applies to | Description |
|---|---|---|---|
| `model_name` | `all-MiniLM-L6-v2` | All models | SentenceTransformer model |
| `min_rating` | `4.0` | All models | Minimum rating to consider a movie "liked" |
| `n_neighbours` | `3` | Collaborative, Hybrid | Number of similar users to consult |
| `alpha` | `0.5` | Hybrid | Blend weight — collab vs content |
| `distance` | `"Cosine"` / `"cosine"` | Vector stores | Similarity metric |
| `recreate` | `False` | QdrantStore | Drop and recreate collection on init |
| `persist_dir` | `"./chroma_data"` | ChromaStore | Disk path (`None` = in-memory) |

---

## 📌 Design Decisions

| Decision | Rationale |
|---|---|
| `BaseVectorStore` ABC | Decouples model logic from storage — swap backends without touching recommender code |
| QdrantStore for production | Scalable ANN at query time, server-side payload filtering, Docker persistence |
| ChromaStore for dev/CI | Zero infrastructure: in-process, no daemon, ephemeral mode for tests |
| Mean pooling for user profiles | Simple, robust aggregation regardless of how many movies a user has rated |
| Min-max normalization before blending | Prevents one score distribution from dominating in hybrid mode |
| Cold-start fallback in hybrid | A user absent from the collaborative index still gets content-based results |
| Server-side genre filter | Filtering inside the vector store avoids embedding computation on irrelevant candidates |
| 63-bit hash for Qdrant IDs | Qdrant requires unsigned integer IDs; original string IDs are preserved in payload |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
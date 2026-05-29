# 🎬 VecRecSys — Vector-Powered Recommendation System

> A modular movie recommendation system backed by **Qdrant** as a persistent vector database.
> It implements three complementary approaches: Content-Based Filtering, Collaborative Filtering, and a Hybrid method that blends both.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Qdrant — Vector Database](#qdrant--vector-database)
- [Approaches Implemented](#approaches-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Qdrant Collections](#qdrant-collections)
- [CLI Reference](#cli-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Extending the System](#extending-the-system)
- [License](#license)

---

## Overview

Traditional recommendation systems struggle with three fundamental problems as datasets grow:

- **Scalability** — computing pairwise user-item similarities across millions of records is computationally prohibitive using a plain matrix approach.
- **Cold Start** — when a new user or item appears with no interaction history, purely collaborative systems have nothing to work from.
- **Sparsity** — in most real-world datasets, each user interacts with only a tiny fraction of available items, making similarity signals weak and unreliable.

VecRecSys addresses all three by representing both movies and users as **dense vector embeddings** produced by a SentenceTransformer model, then storing and querying them through **Qdrant**, a production-grade vector database. Similarity search in this embedding space is fast (HNSW index), semantically rich (meaning is encoded in the vectors, not just co-occurrence), and persistent across restarts thanks to a Docker volume.

| Challenge | Classical approach | VecRecSys + Qdrant |
|---|---|---|
| Scalability | User-item matrix — O(n²) | ANN search — O(log n) |
| Cold Start | Fails for new users / items | Metadata-driven embeddings work from day one |
| Sparsity | Collaborative filtering breaks down | Dense semantic similarity fills the gap |
| Persistence | In-memory, lost on restart | Docker volume — survives restarts |
| Attribute filtering | Client-side post-processing | Server-side Qdrant payload filter |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          VecRecSys                              │
│                                                                 │
│  ┌──────────┐   ┌──────────────────┐   ┌─────────────────────┐  │ 
│  │   Data   │──▶│   Embedding      │──▶│       Qdrant       │  │
│  │  Layer   │   │   Generator      │   │  ┌───────────────┐  │  │
│  │ movies   │   │ SentenceTransfor-│   │  │recsys_movies  │  │  │
│  │ interact │   │ mer MiniLM-L6-v2 │   │  │recsys_users   │  │  │
│  └──────────┘   └──────────────────┘   │  └───────────────┘  │  │
│                                        │  Cosine · Persistent│  │
│                                        └─────────────────────┘  │
│                                                   │             │
│  ┌────────────────────────────────────────────────▼──────────┐  │
│  │                  Recommendation Engine                    │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐   │  │
│  │  │ Collaborative │  │ Content-Based │  │    Hybrid    │   │  │
│  │  │  Filtering    │  │  Filtering    │  │  α-blending  │   │  │
│  │  └───────────────┘  └───────────────┘  └──────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The pipeline has three stages:

1. **Data Layer** — raw CSV files for movies (title, genre, year, description) and user-movie interactions (ratings).
2. **Embedding Generator** — `SentenceTransformer('all-MiniLM-L6-v2')` encodes each movie description into a 384-dimensional float vector. User profiles are derived by averaging the vectors of movies they rated positively.
3. **Qdrant** — two persistent collections (`recsys_movies`, `recsys_users`) store all vectors. All similarity searches, including optional server-side genre filters, are executed inside Qdrant using its HNSW index.

---

## Qdrant — Vector Database

Qdrant is an open-source, Rust-based vector database designed specifically for high-performance similarity search. It is the persistence and search backbone of VecRecSys.

### Why Qdrant?

| Feature | What it means for this project |
|---|---|
| **Persistent storage** | Vectors are stored on a Docker volume and survive process restarts. Embeddings do not need to be recomputed every run. |
| **HNSW indexing** | Hierarchical Navigable Small World graph — delivers approximate nearest-neighbour search in O(log n), making queries fast even with millions of vectors. |
| **Payload filtering** | Every point can carry a JSON payload (e.g. `{"genre": "sci-fi", "year": 2016}`). Filters are applied *inside* Qdrant before returning results, eliminating unnecessary client-side post-processing. |
| **Cosine distance** | Collections are configured with `Distance.COSINE`, which is the natural metric for normalised sentence embeddings. |
| **gRPC + REST** | Two communication interfaces — REST for simplicity, gRPC for lower latency in high-throughput production scenarios. |
| **Open-source** | Apache 2.0 licence. Can run fully locally or migrate to Qdrant Cloud without code changes. |

### Collections used

| Collection | Content | Payload fields | Dimension |
|---|---|---|---|
| `recsys_movies` | One vector per movie — encodes the description | `genre`, `title` | 384 |
| `recsys_users` | One vector per user — encodes their taste profile | `user_id` | 384 |

Each point in a collection has a numeric ID (mapped internally from the original movie or user ID), a vector, and a payload dictionary. The payload fields are indexed and used by the server-side filters described in the CLI reference.

---

## Approaches Implemented

### 1. 🎯 Content-Based Filtering (`models/content_based.py`)

Content-based filtering recommends items that are **semantically similar** to what a user has already liked, based solely on item attributes — in this case, the movie description text.

**How it works, step by step:**

1. Every movie description is encoded into a 384-dim vector by `SentenceTransformer`. Vectors are upserted into the `recsys_movies` Qdrant collection, each with a payload containing its genre and title.
2. When a recommendation request arrives for a user, the system retrieves the embeddings of all movies that user rated ≥ 4/5 and computes their **mean vector**. This mean vector is the user's taste profile.
3. That profile vector is sent to Qdrant as a query. Qdrant returns the `top_k` movies with the highest cosine similarity to the profile — these become the recommendations.
4. An optional `genre_filter` argument causes Qdrant to apply a **server-side payload filter** before scoring, so only movies of the requested genre are considered.

**Strengths:** Works for brand-new movies (cold start for items) as long as a description exists. Does not require any other user's data — entirely self-contained per user.

**Limitations:** Cannot discover items outside the user's established taste profile. Two movies with different wording but the same theme may not score as similar as expected.

---

### 2. 🤝 Collaborative Filtering (`models/collaborative.py`)

Collaborative filtering recommends items based on the preferences of **similar users**, without looking at item content at all — "users who liked what you liked also liked this."

**How it works, step by step:**

1. All movie descriptions are encoded (same model). For each user, the embeddings of their positively rated movies are averaged into a single user profile vector.
2. All user profile vectors are upserted into the `recsys_users` Qdrant collection.
3. For a target user, the system queries `recsys_users` with the user's own profile vector to find the **K nearest neighbour users** — users whose taste profiles are closest in embedding space.
4. For each neighbour, the system retrieves their liked movies and computes a weighted score: `score = Σ (neighbour_similarity × rating / 5)`. Movies already seen by the target user are excluded.
5. Results are ranked by aggregated score and returned as recommendations.

**Strengths:** Can surface unexpected discoveries the user would never have searched for. Captures social trends and genre crossovers.

**Limitations:** Requires sufficient interaction history for both the target user and their neighbours. New users with no history cannot be served — the hybrid approach handles this case with a graceful fallback.

---

### 3. 🔀 Hybrid Approach (`models/hybrid.py`)

The hybrid recommender combines both signals into a single ranked list using a configurable **α parameter**:

```
hybrid_score = α × collaborative_score + (1 − α) × content_score
```

**How it works, step by step:**

1. Both sub-recommenders (`ContentBasedRecommender` and `CollaborativeRecommender`) are fitted independently. They each maintain their own Qdrant collection.
2. For a given user, both produce a candidate list with raw similarity scores.
3. Each score list is **min-max normalised** to [0, 1] independently before blending. This prevents one method from dominating simply because its raw scores are on a larger scale.
4. The union of both candidate sets is scored using the formula above. Items appearing in only one list receive a score of 0 for the missing component.
5. If a user has no profile in the collaborative index (new user), the system logs a warning and falls back to pure content-based scoring (`α` is effectively set to 0 for that user).
6. An optional `genre_filter` is forwarded to the content-based branch, where Qdrant applies it server-side before the similarity search.

**Parameter guide:**

| α value | Behaviour |
|---|---|
| `0.0` | Pure content-based — ignores other users entirely |
| `0.5` | Equal blend — default, balanced behaviour |
| `1.0` | Pure collaborative — ignores item content entirely |
| `0.3` | Leans content-based — useful for niche or new users |
| `0.7` | Leans collaborative — useful for well-established users |

---

## Project Structure

```
recsys/
│
├── docker-compose.yml          ← Launches Qdrant with a persistent Docker volume
├── main.py                     ← CLI entry point — runs one or all approaches
├── requirements.txt            ← Python dependencies
├── README.md
│
├── data/
│   ├── movies.csv              ← Movie catalogue: movie_id, title, genre, year, description
│   └── interactions.csv        ← User ratings: user_id, movie_id, rating (1–5)
│
├── embeddings/
│   └── embedding_generator.py  ← SentenceTransformer wrapper with batch encoding support
│
├── models/
│   ├── content_based.py        ← Content-Based Recommender (uses recsys_movies)
│   ├── collaborative.py        ← Collaborative Recommender (uses recsys_users)
│   └── hybrid.py               ← Hybrid Recommender (α-blending of both)
│
└── utils/
    ├── vector_store.py         ← Qdrant client wrapper: upsert, search, payload filter
    └── evaluator.py            ← Offline metrics: Precision@K, Recall@K, NDCG@K
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose (to run Qdrant)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourname/vecrec-sys.git
cd vecrec-sys

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start Qdrant (runs in the background, data persisted in a named Docker volume)
docker compose up -d

# 5. Verify Qdrant is ready
curl http://localhost:6333/healthz
# Expected response: {"title":"qdrant - vector search engine"}
```

> **Note:** The first run will download the `qdrant/qdrant:v1.9.2` Docker image (~100 MB) and the `all-MiniLM-L6-v2` SentenceTransformer model (~90 MB). Subsequent runs use the local cache.

---

## Quick Start

```bash
# Run all three approaches for user 1, return top 5 each
python main.py --user_id 1 --top_k 5

# Content-based only, restricted to sci-fi movies (Qdrant payload filter)
python main.py --user_id 1 --mode content --genre sci-fi

# Collaborative filtering only
python main.py --user_id 1 --mode collaborative --top_k 5

# Hybrid with α=0.7 — leans towards collaborative signal
python main.py --user_id 1 --mode hybrid --alpha 0.7 --top_k 5

# Connect to a remote Qdrant instance
python main.py --user_id 1 --qdrant_host 192.168.1.10 --qdrant_port 6333
```

**Example output:**
```
════════════════════════════════════════════════════════
       VecRecSys  —  Qdrant Backend
════════════════════════════════════════════════════════
  User     : 1
  Mode     : all
  Top-K    : 5
  Qdrant   : localhost:6333
════════════════════════════════════════════════════════

────────────────────────────────────────────────────────
  Content-Based Filtering
────────────────────────────────────────────────────────
   1. Arrival                          (2016)  [sci-fi]  score=0.8912
   2. Ex Machina                       (2014)  [sci-fi]  score=0.8754
   3. The Martian                      (2015)  [sci-fi]  score=0.8601

────────────────────────────────────────────────────────
  Collaborative Filtering
────────────────────────────────────────────────────────
   1. Gravity                          (2013)  [sci-fi]  score=0.7230
   2. Arrival                          (2016)  [sci-fi]  score=0.6890

────────────────────────────────────────────────────────
  Hybrid  (α=0.5)
────────────────────────────────────────────────────────
   1. Arrival                          (2016)  [sci-fi]  score=0.7851
   2. Ex Machina                       (2014)  [sci-fi]  score=0.7102
   3. Gravity                          (2013)  [sci-fi]  score=0.6844
```

---

## Qdrant Collections

### Inspect via the REST API

```bash
# List all collections
curl http://localhost:6333/collections

# Get info about the movies collection
curl http://localhost:6333/collections/recsys_movies

# Manual vector search with a genre payload filter
curl -X POST http://localhost:6333/collections/recsys_movies/points/search \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, ...],
    "limit": 5,
    "filter": {
      "must": [{"key": "genre", "match": {"value": "sci-fi"}}]
    }
  }'
```

### Qdrant Web Dashboard

Once Docker is running, the built-in dashboard is available at:

```
http://localhost:6333/dashboard
```

The dashboard lets you browse collections, inspect individual points and their payloads, and run test queries interactively — useful for debugging embeddings without writing code.

---

## CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--user_id` | int | **required** | Target user ID to generate recommendations for |
| `--mode` | str | `all` | Which approach to run: `content`, `collaborative`, `hybrid`, or `all` |
| `--top_k` | int | `5` | Number of recommendations to return per approach |
| `--alpha` | float | `0.5` | Hybrid blending weight — 0.0 = pure content, 1.0 = pure collaborative |
| `--genre` | str | `None` | Server-side Qdrant payload filter — restricts candidates to a specific genre |
| `--qdrant_host` | str | `localhost` | Hostname or IP address of the Qdrant instance |
| `--qdrant_port` | int | `6333` | REST API port of the Qdrant instance |

---

## Evaluation Metrics

Run offline evaluation using a leave-one-out split across all users:

```bash
python utils/evaluator.py
```

| Metric | What it measures |
|---|---|
| **Precision@K** | Of the K items recommended, what fraction were actually relevant? A score of 1.0 means every recommendation was on target. |
| **Recall@K** | Of all items the user actually liked, what fraction appeared in the top K? A score of 1.0 means nothing relevant was missed. |
| **NDCG@K** | Normalized Discounted Cumulative Gain — measures ranking quality. A relevant item ranked 1st scores higher than the same item ranked 5th. This is the most informative metric for ordered recommendation lists. |

---

## Extending the System

### Swap the embedding model

Only one line needs to change in `EmbeddingGenerator`:

```python
model_name = "all-mpnet-base-v2"                        # Higher accuracy, slower (~768 dim)
model_name = "multilingual-e5-large"                    # Multilingual support (EN, FR, AR, ...)
model_name = "paraphrase-multilingual-MiniLM-L12-v2"    # Fast multilingual, good for mixed datasets
```

> Changing the model changes the vector dimension. Recreate the Qdrant collections (`recreate=True`) after any model swap.

### Switch to gRPC for lower latency

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
```

gRPC is more efficient than REST for high-throughput scenarios (batch inserts, many parallel queries).

### Add a year range filter

Extend `VectorStore.search()` with a `RangeCondition` on the `year` payload field:

```python
from qdrant_client.models import Range, FieldCondition

year_condition = FieldCondition(key="year", range=Range(gte=2010, lte=2020))
# Pass it into the Filter(must=[...]) alongside the genre condition
```

### Migrate to Qdrant Cloud

No code changes beyond the client initialisation:

```python
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)
```

Everything else — collections, payloads, search logic — remains identical.

---

## License

MIT — free to use, modify, and distribute.
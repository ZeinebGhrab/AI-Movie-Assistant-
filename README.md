# 🎬 VecRecSys — Vector-Powered Recommendation System

> A full-stack movie recommendation system backed by **Qdrant** (vector database) and **MongoDB** (data layer).
> It implements three complementary approaches — Content-Based Filtering, Collaborative Filtering, and a Hybrid α-blending method — exposed through a **FastAPI** backend and a **React** dashboard.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Stack](#stack)
- [Qdrant — Vector Database](#qdrant--vector-database)
- [Approaches Implemented](#approaches-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [CLI](#cli)
  - [API Server](#api-server)
  - [Frontend](#frontend)
- [API Reference](#api-reference)
- [Qdrant Collections](#qdrant-collections)
- [CLI Reference](#cli-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Performance Tuning (Lesson 6)](#performance-tuning-lesson-6)
- [Extending the System](#extending-the-system)
- [License](#license)

---

## Overview

Traditional recommendation systems struggle with three fundamental problems as datasets grow:

- **Scalability** — computing pairwise user-item similarities across millions of records is computationally prohibitive.
- **Cold Start** — purely collaborative systems fail for new users or items with no interaction history.
- **Sparsity** — most users rate only a tiny fraction of available items, making similarity signals weak.

VecRecSys addresses all three by representing both movies and users as **dense vector embeddings** produced by a SentenceTransformer model, then storing and querying them through **Qdrant**, a production-grade vector database. The data layer is backed by **MongoDB**, replacing static CSV files with a persistent, query-friendly document store.

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
┌──────────────────────────────────────────────────────────────────────┐
│                            VecRecSys                                 │
│                                                                      │
│  ┌──────────┐   ┌──────────────────┐   ┌──────────────────────────┐  │
│  │  MongoDB │──▶│   Embedding      │──▶│         Qdrant           │  │
│  │  movies  │   │   Generator      │   │  ┌──────────────────────┐ │  │
│  │  interact│   │ all-MiniLM-L6-v2 │   │  │  recsys_movies       │ │  │
│  └──────────┘   │   384 dim        │   │  │  recsys_users        │ │  │
│                 └──────────────────┘   │  └──────────────────────┘ │  │
│                                        │  Cosine · HNSW · Persist  │  │
│                                        └──────────────────────────┘  │
│                                                    │                  │
│  ┌─────────────────────────────────────────────────▼──────────────┐  │
│  │                    Recommendation Engine                        │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐   │  │
│  │  │ Collaborative │  │ Content-Based │  │  Hybrid (α-blend)│   │  │
│  │  └───────────────┘  └───────────────┘  └──────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                  │
│            ┌───────────────────────┴──────────────────┐              │
│            │           FastAPI REST API                │              │
│            │  /api/recommend · /api/evaluate · ...     │              │
│            └───────────────────────┬──────────────────┘              │
│                                    │                                  │
│                    ┌───────────────▼──────────────┐                  │
│                    │   React Dashboard (Vite)      │                  │
│                    │   Recommend · Evaluate tabs   │                  │
│                    └──────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dim) |
| Vector DB | Qdrant v1.9.2 — HNSW index, cosine similarity, Docker volume |
| Data layer | MongoDB 7.0 — movies & interactions collections |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 18 + Vite |
| CLI | Python `argparse` |
| Evaluation | Precision@K · Recall@K · NDCG@K |
| Perf. tuning | `PerformanceEvaluator`, `HNSWTuner`, `TopKTuner`, `FilterOverheadProfiler` |

---

## Qdrant — Vector Database

Qdrant is an open-source, Rust-based vector database optimised for high-performance similarity search. It is the persistence and search backbone of VecRecSys.

### Key features used

| Feature | How VecRecSys uses it |
|---|---|
| **Persistent storage** | Vectors stored on a Docker named volume (`qdrant_data`) — survive restarts |
| **HNSW indexing** | Default Qdrant index — ANN search in O(log n) |
| **Cosine distance** | Configured per collection via `Distance.COSINE` |
| **Payload filtering** | Genre stored in payload, filtered server-side with `FieldCondition` |
| **REST + gRPC** | Python client uses REST (port 6333); gRPC available on 6334 |
| **Collections** | `recsys_movies` and `recsys_users` |

### Collections

| Collection | Content | Payload fields | Dimension |
|---|---|---|---|
| `recsys_movies` | One vector per movie (description embedding) | `genre`, `title`, `item_id` | 384 |
| `recsys_users` | One vector per user (mean of liked movies) | `user_id`, `item_id` | 384 |

---

## Approaches Implemented

### 1. 🎯 Content-Based Filtering (`models/content_based.py`)

Recommends movies semantically similar to what a user has already liked, based solely on description text.

**Pipeline:**
1. Encode all movie descriptions → 384-dim vectors → upsert into `recsys_movies`.
2. Build user profile: average the vectors of movies rated ≥ 4/5.
3. Query Qdrant with the user vector → top-k nearest movies by cosine similarity.
4. Optional `genre_filter` triggers a **server-side Qdrant payload filter**.

**Strengths:** Works for brand-new movies (no interaction history needed). Entirely self-contained per user.

**Limitations:** Cannot surface items outside the user's established taste. Serendipity is low.

---

### 2. 🤝 Collaborative Filtering (`models/collaborative.py`)

Recommends items based on the preferences of similar users — no item content required.

**Pipeline:**
1. Encode movies; average liked-movie vectors per user → user profile vectors.
2. Upsert user profiles into `recsys_users`.
3. For the target user, query `recsys_users` → K nearest-neighbour users.
4. Aggregate their liked movies with a weighted score: `score = Σ (similarity × rating / 5)`.
5. Exclude movies already seen by the target user.

**Strengths:** Can surface unexpected discoveries. Captures cross-genre social trends.

**Limitations:** New users with no history cannot be served — the hybrid approach handles this with a content-based fallback.

---

### 3. 🔀 Hybrid Approach (`models/hybrid.py`)

Blends both signals into a single ranked list using a configurable **α** parameter:

```
hybrid_score = α × collaborative_score + (1 − α) × content_score
```

Both score lists are **min-max normalised** to `[0, 1]` before blending to prevent scale dominance. If a user has no collaborative profile, the system gracefully falls back to pure content-based scoring.

**Alpha guide:**

| α | Behaviour | Best for |
|---|---|---|
| `0.0` | Pure content-based | New / niche users |
| `0.5` | Equal blend (default) | General use |
| `1.0` | Pure collaborative | Users with dense history |

---

## Project Structure

```
recsys/
│
├── docker-compose.yml          ← Qdrant + MongoDB with persistent Docker volumes
├── main.py                     ← CLI entry point
├── requirements.txt
├── README.md
│
├── api/
│   └── app.py                  ← FastAPI backend (7 endpoints)
│
├── embeddings/
│   └── embedding_generator.py  ← SentenceTransformer wrapper
│
├── models/
│   ├── content_based.py
│   ├── collaborative.py
│   └── hybrid.py
│
├── utils/
│   ├── db.py                   ← MongoDB data layer (MongoDataLoader)
│   ├── vector_store.py         ← Qdrant client wrapper
│   ├── evaluator.py            ← Precision@K · Recall@K · NDCG@K
│   ├── performance_evaluator.py← Lesson 6 — latency · QPS · F1
│   └── tuner.py                ← HNSWTuner · TopKTuner · FilterOverheadProfiler
│
├── vectordb/
│   ├── base.py                 ← Abstract BaseVectorStore interface
│   ├── qdrant_store.py         ← Qdrant backend (pluggable)
│   └── chroma_store.py         ← ChromaDB backend (pluggable)
│
├── docs/
│   ├── Course-Vector-Databases-Basics.md
│   └── key-concepts.md         ← Theory → code mapping (35 lessons)
│
└── frontend/
    ├── src/
    │   ├── App.jsx             ← Main React component
    │   └── main.jsx
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Installation

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Node.js 18+ (for the React frontend)

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

# 4. Start Qdrant + MongoDB
docker compose up -d

# 5. Verify services are ready
curl http://localhost:6333/healthz   # → {"title":"qdrant - vector search engine"}
mongosh --eval "db.adminCommand('ping')"
```

> **Note:** The first run downloads the `qdrant/qdrant:v1.9.2` image (~100 MB), the `mongo:7.0` image, and the `all-MiniLM-L6-v2` SentenceTransformer model (~90 MB). Subsequent runs use the local cache.

---

## Quick Start

### CLI

```bash
# Seed MongoDB with sample data (first run only)
python main.py --user_id 1 --seed

# Run all three approaches for user 1, top-5 each
python main.py --user_id 1 --top_k 5

# Content-based only, filtered to sci-fi (Qdrant server-side filter)
python main.py --user_id 1 --mode content --genre sci-fi

# Collaborative filtering only
python main.py --user_id 1 --mode collaborative

# Hybrid with α=0.7 — leans towards collaborative
python main.py --user_id 1 --mode hybrid --alpha 0.7

# Connect to remote Qdrant + MongoDB
python main.py --user_id 1 --qdrant_host 192.168.1.10 --mongo_uri mongodb://192.168.1.10:27017
```

### API Server

```bash
uvicorn api.app:app --reload --port 8000
```

The API seeds MongoDB and fits all three recommenders at startup. Navigate to `http://localhost:8000/docs` for the interactive Swagger UI.

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The Vite dev server proxies all `/api` calls to `http://localhost:8000`, so both services must be running simultaneously.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/movies` | List all movies |
| `GET` | `/api/users` | List all users with stats |
| `GET` | `/api/users/{user_id}/ratings` | User's rated movies |
| `GET` | `/api/genres` | List available genres |
| `POST` | `/api/recommend` | Get recommendations |
| `GET` | `/api/evaluate` | Run offline evaluation |

### POST `/api/recommend`

```json
{
  "user_id": 1,
  "mode": "hybrid",
  "top_k": 5,
  "alpha": 0.5,
  "genre": "sci-fi"
}
```

**Response:**
```json
{
  "user_id": 1,
  "mode": "hybrid",
  "recommendations": [
    {
      "movie_id": 5,
      "title": "Arrival",
      "genre": "sci-fi",
      "year": 2016,
      "description": "...",
      "score": 0.8912
    }
  ],
  "latency_ms": 4.23
}
```

---

## Qdrant Collections

### Inspect via the REST API

```bash
# List all collections
curl http://localhost:6333/collections

# Get info about the movies collection
curl http://localhost:6333/collections/recsys_movies

# Manual vector search with genre payload filter
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

```
http://localhost:6333/dashboard
```

Browse collections, inspect point payloads, and run test queries interactively.

---

## CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--user_id` | int | **required** | Target user ID |
| `--mode` | str | `all` | `content` · `collaborative` · `hybrid` · `all` |
| `--top_k` | int | `5` | Number of recommendations per approach |
| `--alpha` | float | `0.5` | Hybrid blending weight (0 = content, 1 = collaborative) |
| `--genre` | str | `None` | Server-side Qdrant genre filter |
| `--qdrant_host` | str | `localhost` | Qdrant hostname |
| `--qdrant_port` | int | `6333` | Qdrant REST port |
| `--mongo_uri` | str | `mongodb://localhost:27017` | MongoDB connection URI |
| `--mongo_db` | str | `recsys` | MongoDB database name |
| `--seed` | flag | — | Seed MongoDB with sample data if empty |
| `--seed_force` | flag | — | Drop and re-seed MongoDB collections |

---

## Evaluation Metrics

Run offline evaluation using a leave-one-out split across all users:

```bash
python utils/evaluator.py
```

| Metric | What it measures |
|---|---|
| **Precision@K** | Of the K items recommended, what fraction were actually relevant? |
| **Recall@K** | Of all items the user liked, what fraction appeared in the top K? |
| **NDCG@K** | Ranking quality — a relevant item at rank 1 scores higher than at rank 5. |

Or via the API:

```bash
curl http://localhost:8000/api/evaluate?top_k=5
```

---

## Performance Tuning (Lesson 6)

VecRecSys ships two dedicated tools for benchmarking and tuning vector search performance.

### Performance Evaluator

Measures query latency (mean, P50, P95, P99), QPS, Precision@K, Recall@K, F1, and NDCG@K across multiple top-k values:

```bash
python utils/performance_evaluator.py --seed --top_k_values 1 5 10
```

### Tuner

Three tuning workflows in one script:

```bash
python utils/tuner.py --seed --genre_filter sci-fi
```

| Tool | What it does |
|---|---|
| `HNSWTuner` | Sweeps Qdrant's `efSearch` parameter — higher ef = better recall, higher latency |
| `TopKTuner` | Sweeps K to visualise the classic Precision–Recall tradeoff curve |
| `FilterOverheadProfiler` | Quantifies the latency cost of server-side payload filtering |

**Example HNSW sweep output:**

| ef | Lat mean | Recall | Prec | NDCG |
|---|---|---|---|---|
| 32 | 1.2 ms | 0.7200 | 0.1440 | 0.7200 |
| 128 | 2.1 ms | 0.8400 | 0.1680 | 0.8400 |
| 512 | 4.8 ms | 0.8600 | 0.1720 | 0.8600 |

---

## Extending the System

### Swap the embedding model

One line change in `EmbeddingGenerator`:

```python
model_name = "all-mpnet-base-v2"                         # Higher accuracy (~768 dim)
model_name = "multilingual-e5-large"                     # EN / FR / AR support
model_name = "paraphrase-multilingual-MiniLM-L12-v2"     # Fast multilingual
```

> Changing the model changes the vector dimension. Recreate Qdrant collections (`recreate=True`) after any model swap.

### Switch to gRPC for lower latency

```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
```

### Add a year range filter

```python
from qdrant_client.models import Range, FieldCondition

year_condition = FieldCondition(key="year", range=Range(gte=2010, lte=2020))
```

### Use the pluggable vector store backends

The `vectordb/` package exposes a `BaseVectorStore` interface. Both `QdrantStore` and `ChromaStore` are drop-in replacements:

```python
from vectordb.chroma_store import ChromaStore

store = ChromaStore(collection="movies", persist_dir="./chroma_data")
```

### Migrate to Qdrant Cloud

No code changes beyond the client initialisation:

```python
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)
```

### Add a new movie or rating at runtime

```python
from utils.db import MongoDataLoader

loader = MongoDataLoader()
loader.add_movie({"movie_id": 16, "title": "Dune", "genre": "sci-fi", "year": 2021, "description": "..."})
loader.add_interaction(user_id=1, movie_id=16, rating=5)
```

---

## License

MIT License © Zeineb Ghrab — free to use, modify, and distribute.

---

## Author

**Zeineb Ghrab** <br/>
*Data & Decisional Systems Engineering Student — ENET'Com*

---

*Last Updated: 06/06/2026* <br/>
*Status: Active ✓* <br/>
*Maturity Level: Prototype / Research Implementation (Not Production Ready)*
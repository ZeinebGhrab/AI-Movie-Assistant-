# 🎬 VecRecSys — Vector-Powered Recommendation System

> Système de recommandation modulaire utilisant **Qdrant** comme vector database persistante.
> Implémente trois approches : Collaborative Filtering, Content-Based Filtering, et Hybrid.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Qdrant — Vector Database](#qdrant--vector-database)
- [Approaches Implemented](#approaches-implemented)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Collections Qdrant](#collections-qdrant)
- [CLI Reference](#cli-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Extending the System](#extending-the-system)

---

## Overview

| Challenge            | Approche classique              | VecRecSys + Qdrant                    |
|----------------------|---------------------------------|---------------------------------------|
| Scalabilité          | Matrice user-item O(n²)         | ANN search O(log n)                   |
| Cold Start           | Échec sur nouveaux items/users  | Embeddings basés sur les métadonnées  |
| Sparsité             | Filtrage collaboratif fragile   | Similarité sémantique dense           |
| Persistance          | Mémoire volatile                | Volume Docker persistant              |
| Filtrage par attribut| Post-processing client-side     | Payload filter Qdrant server-side     |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          VecRecSys                              │
│                                                                 │
│  ┌──────────┐   ┌──────────────────┐   ┌─────────────────────┐ │
│  │   Data   │──▶│   Embedding      │──▶│       Qdrant        │ │
│  │  Layer   │   │   Generator      │   │  ┌───────────────┐  │ │
│  │ movies   │   │ SentenceTransfor-│   │  │recsys_movies  │  │ │
│  │ interact │   │ mer MiniLM-L6-v2 │   │  │recsys_users   │  │ │
│  └──────────┘   └──────────────────┘   │  └───────────────┘  │ │
│                                        │  Cosine · Persistent │ │
│                                        └─────────────────────┘  │
│                                                   │             │
│  ┌────────────────────────────────────────────────▼──────────┐  │
│  │                  Recommendation Engine                    │  │
│  │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  │  │
│  │  │ Collaborative │  │ Content-Based │  │    Hybrid    │  │  │
│  │  │  Filtering    │  │  Filtering    │  │  α-blending  │  │  │
│  │  └───────────────┘  └───────────────┘  └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Qdrant — Vector Database

### Pourquoi Qdrant ?

| Fonctionnalité         | Détail                                          |
|------------------------|-------------------------------------------------|
| **Stockage persistant**| Volume Docker — données survivent aux redémarrages |
| **HNSW indexing**      | Approximate Nearest Neighbour ultra-rapide      |
| **Payload filtering**  | Filtrer par genre, année, etc. côté serveur     |
| **gRPC + REST**        | Deux interfaces disponibles                     |
| **Open-source**        | Apache 2.0, Rust-based, haute performance       |

### Collections utilisées

| Collection       | Contenu                       | Dimension |
|------------------|-------------------------------|-----------|
| `recsys_movies`  | Embeddings des films          | 384       |
| `recsys_users`   | Embeddings des profils users  | 384       |

---

## Approaches Implemented

### 1. 🎯 Content-Based Filtering (`models/content_based.py`)
- Encodage des descriptions films via `SentenceTransformer`
- Stockage dans `recsys_movies` avec payload `{genre, title}`
- Profil user = moyenne des vecteurs des films aimés
- Recherche Qdrant + filtre payload optionnel (genre)

### 2. 🤝 Collaborative Filtering (`models/collaborative.py`)
- Encodage de tous les films → calcul du profil user
- Profils users stockés dans `recsys_users`
- Recherche des K voisins les plus proches dans Qdrant
- Agrégation pondérée : `score = Σ similarity × (rating / 5)`

### 3. 🔀 Hybrid Approach (`models/hybrid.py`)
- Fusionne les scores des deux approches :
  `hybrid = α × collab + (1−α) × content`
- Normalisation min-max avant fusion
- Filtre genre Qdrant appliqué sur la branche content
- Fallback automatique si user absent du collaboratif

---

## Project Structure

```
recsys/
│
├── docker-compose.yml          ← Lance Qdrant (volume persistant)
├── main.py                     ← CLI — toutes les approches
├── requirements.txt
├── README.md
│
├── data/
│   ├── movies.csv              ← Catalogue films (id, title, genre, year, description)
│   └── interactions.csv        ← Ratings (user_id, movie_id, rating)
│
├── embeddings/
│   └── embedding_generator.py  ← SentenceTransformer wrapper
│
├── models/
│   ├── content_based.py        ← Content-Based (Qdrant recsys_movies)
│   ├── collaborative.py        ← Collaborative (Qdrant recsys_users)
│   └── hybrid.py               ← Hybrid (α-blending)
│
└── utils/
    ├── vector_store.py         ← Client Qdrant (upsert, search, payload filter)
    └── evaluator.py            ← Precision@K, Recall@K, NDCG@K
```

---

## Installation

### Prérequis
- Python 3.9+
- Docker + Docker Compose

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/ZeinebGhrab/VecRecSys.git
cd vecrec-sys

# 2. Environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Dépendances Python
pip install -r requirements.txt

# 4. Lancer Qdrant (volume persistant sur Docker)
docker compose up -d

# Vérifier que Qdrant est prêt
curl http://localhost:6333/healthz
# → {"title":"qdrant - vector search engine"}
```

---

## Quick Start

```bash
# Toutes les approches pour user 1
python main.py --user_id 1 --top_k 5

# Content-based uniquement, filtré sur le genre sci-fi
python main.py --user_id 1 --mode content --genre sci-fi

# Hybrid avec alpha = 0.7 (favorise le collaboratif)
python main.py --user_id 1 --mode hybrid --alpha 0.7 --top_k 5

# Qdrant sur hôte distant
python main.py --user_id 1 --qdrant_host 192.168.1.10 --qdrant_port 6333
```

**Exemple de sortie :**
```
════════════════════════════════════════════════════
       VecRecSys  —  Qdrant Backend
════════════════════════════════════════════════════
  User     : 1
  Mode     : all
  Top-K    : 5
  Qdrant   : localhost:6333
════════════════════════════════════════════════════

────────────────────────────────────────────────────
  Content-Based Filtering
────────────────────────────────────────────────────
   1. Arrival                          (2016)  [sci-fi]  score=0.8912
   2. Ex Machina                       (2014)  [sci-fi]  score=0.8754
   3. The Martian                      (2015)  [sci-fi]  score=0.8601

────────────────────────────────────────────────────
  Collaborative Filtering
────────────────────────────────────────────────────
   1. Gravity                          (2013)  [sci-fi]  score=0.7230
   2. Arrival                          (2016)  [sci-fi]  score=0.6890

────────────────────────────────────────────────────
  Hybrid  (α=0.5)
────────────────────────────────────────────────────
   1. Arrival                          (2016)  [sci-fi]  score=0.7851
   2. Ex Machina                       (2014)  [sci-fi]  score=0.7102
   3. Gravity                          (2013)  [sci-fi]  score=0.6844
```

---

## Collections Qdrant

### Inspecter via l'API REST

```bash
# Lister toutes les collections
curl http://localhost:6333/collections

# Infos sur la collection films
curl http://localhost:6333/collections/recsys_movies

# Chercher manuellement (exemple)
curl -X POST http://localhost:6333/collections/recsys_movies/points/search \
  -H 'Content-Type: application/json' \
  -d '{
    "vector": [0.1, 0.2, ...],
    "limit": 5,
    "filter": {"must": [{"key": "genre", "match": {"value": "sci-fi"}}]}
  }'
```

### Dashboard Qdrant Web UI

Accessible sur `http://localhost:6333/dashboard` une fois Docker lancé.

---

## CLI Reference

| Argument         | Type    | Défaut      | Description                                      |
|------------------|---------|-------------|--------------------------------------------------|
| `--user_id`      | int     | **requis**  | ID de l'utilisateur cible                        |
| `--mode`         | str     | `all`       | `content` / `collaborative` / `hybrid` / `all`  |
| `--top_k`        | int     | `5`         | Nombre de recommandations                        |
| `--alpha`        | float   | `0.5`       | Poids collaboratif dans le hybrid                |
| `--genre`        | str     | `None`      | Filtre genre (payload Qdrant)                    |
| `--qdrant_host`  | str     | `localhost` | Hôte Qdrant                                      |
| `--qdrant_port`  | int     | `6333`      | Port REST Qdrant                                 |

---

## Evaluation Metrics

```bash
python utils/evaluator.py
```

| Métrique       | Description                                            |
|----------------|--------------------------------------------------------|
| **Precision@K** | Fraction des top-K qui sont pertinents               |
| **Recall@K**    | Fraction des items pertinents trouvés dans le top-K  |
| **NDCG@K**      | Qualité du classement — pénalise les items mal placés|

---

## Extending the System

### Remplacer le modèle d'embedding
```python
# Dans EmbeddingGenerator — swap one line
model_name = "all-mpnet-base-v2"          # plus précis, plus lent
model_name = "multilingual-e5-large"      # multilingue
model_name = "paraphrase-multilingual-MiniLM-L12-v2"  # FR/AR/EN
```

### Activer gRPC (plus performant en production)
```python
from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", grpc_port=6334, prefer_grpc=True)
```

### Ajouter un filtre année
```python
recs = cb.recommend(user_id=1, genre_filter="sci-fi")
# Étendre VectorStore.search() avec un RangeCondition sur payload "year"
```

### Passer à Qdrant Cloud
```python
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key",
)
```

---

## License

MIT — libre d'utilisation, modification et distribution.

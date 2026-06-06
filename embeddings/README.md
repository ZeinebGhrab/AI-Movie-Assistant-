# 📌 Embedding Generator

> A lightweight, powerful wrapper around **SentenceTransformers** to generate high-quality text embeddings for NLP and AI applications.

---

## 🚀 Overview

This project provides a clean interface for generating semantic text embeddings using state-of-the-art transformer models. It supports single text encoding, batch processing, and hybrid embeddings that combine semantic vectors with structured metadata.

**Supported modes:**
- Single text embedding
- Batch embedding
- Hybrid embeddings with metadata (e.g., genre)

---

## 🧠 Model

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Embedding size | 384 dimensions |
| Type | Transformer-based sentence embedding |

**Strengths:** Fast inference ⚡ · Lightweight · Strong semantic understanding

---

## ⚙️ Features

Three encoding modes are provided because different use cases have fundamentally different requirements in terms of input volume, available data, and downstream tasks.

### 🔹 Single Encoding — *when you have one text at a time*

Used for **real-time or on-demand inference**, such as encoding a user's search query the moment they type it. Processing a single input minimizes latency and keeps the API response fast.

```
Input : one string  →  Output : vector (384,)
```

### 🔹 Batch Encoding — *when you have many texts to process*

Used for **offline indexing or dataset preprocessing**, such as embedding an entire movie catalog before building a search index. Batching leverages GPU/CPU parallelism for significantly faster throughput compared to encoding one text at a time.

```
Input : list of N strings  →  Output : matrix (N, 384)
```

### 🔹 Hybrid Embedding — *when text alone is not enough*

Used when **structured metadata carries meaningful signal** that the text itself doesn't express. For example, the genre of a movie is a categorical business feature — the transformer embedding captures the plot description, but appending a one-hot genre vector allows downstream models to reason about category boundaries explicitly.

```
Input : string + metadata  →  Output : vector (384 + G,)
                                        └─ semantic ─┘ └─ structured ─┘
```

> **Rule of thumb:** start with `encode()` for prototyping, switch to `encode_batch()` for scale, and reach for `encode_with_metadata()` when your metadata consistently improves retrieval or recommendation quality.

---

## 📦 Installation

```bash
pip install sentence-transformers numpy
```

---

## 🧾 Usage

### Initialize the model

```python
from embedding_generator import EmbeddingGenerator

emb_gen = EmbeddingGenerator()
```

### Encode a single text

```python
vector = emb_gen.encode("This movie is amazing")
print(vector.shape)  # (384,)
```

### Encode a batch

```python
texts = [
    "A funny movie",
    "A dramatic story",
    "An action film"
]
vectors = emb_gen.encode_batch(texts)
print(vectors.shape)  # (N, 384)
```

### Hybrid encoding (text + metadata)

```python
genres = ["Action", "Comedy", "Drama"]

vec = emb_gen.encode_with_metadata(
    text="A thrilling battle scene",
    genre="Action",
    genres_list=genres
)
print(vec.shape)  # (384 + len(genres),)
```

---

## 📊 Output Summary

| Method | Output Shape | Description |
|---|---|---|
| `encode()` | `(384,)` | Single text embedding |
| `encode_batch()` | `(N, 384)` | Batch embeddings |
| `encode_with_metadata()` | `(384 + G,)` | Hybrid embedding (text + genre) |

---

## 🎯 Use Cases

- 🎬 Movie recommendation systems
- 🔎 Semantic search engines
- 🤖 RAG (Retrieval-Augmented Generation) pipelines
- 📊 Clustering and classification
- 🧠 Hybrid AI recommendation systems

---

## 🧩 Project Structure

```
embedding-generator/
│
├── embedding_generator.py   # Core module
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## 📌 Key Idea

This module bridges the gap between unstructured text and structured business features:

| Layer | What it captures |
|---|---|
| 🧠 Deep learning embeddings | Semantic meaning of text |
| 🏷️ Structured metadata | Business features (e.g., genre) |
| ✅ Combined representation | Richer vectors for recommendation & retrieval |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
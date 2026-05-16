# 🎬 CineMatch — Hybrid AI Movie Recommender

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A production-grade movie recommendation system that blends **TF-IDF content-based filtering** with **SVD collaborative filtering** into a hybrid engine — served via a Netflix-inspired Streamlit UI with live TMDB movie posters.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Hybrid engine** | Content-based (TF-IDF + cosine) + Collaborative (SVD matrix factorisation) |
| **Smart caching** | Models precomputed once, loaded from disk on every subsequent run |
| **TMDB integration** | Live movie posters, ratings, release years, overviews |
| **Netflix-style UI** | Dark theme, grid layout, hover animations, match % badges |
| **Autocomplete** | Fuzzy partial-match suggestions as you type |
| **Tunable weights** | Sidebar slider to control content vs collaborative balance |
| **Modular codebase** | Clean separation: models/, utils/, config.py |

---

## 🏗 Project Structure

```
movie-recommender/
├── app.py                  ← Streamlit UI (thin layer, no logic)
├── config.py               ← All constants and paths
├── build_models.py         ← One-time model precomputation script
├── requirements.txt
├── README.md
│
├── data/                   ← Place MovieLens CSVs here
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
│
├── models/
│   ├── content_model.py    ← TF-IDF vectorisation + cosine similarity
│   ├── collab_model.py     ← SVD collaborative filtering
│   └── hybrid.py           ← Weighted score blending
│
├── utils/
│   ├── data_loader.py      ← CSV ingestion + genre/tag enrichment
│   ├── tmdb_api.py         ← TMDB REST API client (LRU-cached)
│   └── cache.py            ← Pickle save/load helpers
│
└── artifacts/              ← Auto-created; stores .pkl model files
```

---

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/yourusername/cinematch.git
cd cinematch
pip install -r requirements.txt
```

### 2. Download the dataset
Get the MovieLens dataset from [grouplens.org](https://grouplens.org/datasets/movielens/latest/)
and place `movies.csv`, `ratings.csv`, and `tags.csv` inside the `data/` folder.

### 3. Set your TMDB API key (optional but recommended)
```bash
# Get a free key at https://www.themoviedb.org/settings/api
export TMDB_API_KEY="your_api_key_here"
```

### 4. Precompute models (run once)
```bash
python build_models.py
```
This builds the TF-IDF similarity matrix and SVD model, then caches them to `artifacts/`.

### 5. Launch the app
```bash
streamlit run app.py
```

---

## 🧠 How It Works

### Content-Based Filtering
Each movie's **genres + user tags** are combined into a "soup" string. TF-IDF vectorization
converts this into a high-dimensional sparse vector, and cosine similarity measures pairwise
movie closeness. Movies with similar genre profiles score high.

### Collaborative Filtering (SVD)
The user×movie rating matrix is decomposed using **Singular Value Decomposition**:

```
R ≈ U × Σ × Vᵀ
```

The latent factor vectors capture hidden taste patterns — users who liked sci-fi thrillers
will have similar factor profiles. Item-item similarity on the `Vᵀ` matrix gives fast,
accurate collaborative scores.

### Hybrid Blending
```python
final_score = content_weight × normalize(content_score)
            + collab_weight  × normalize(collab_score)
```
Both scores are min-max normalised before blending. Default: 40% content, 60% collaborative.

---

## ☁️ Deploy to Streamlit Cloud

1. Push your repo to GitHub (do **not** commit `data/` or `artifacts/` — they're in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set `TMDB_API_KEY` in **Secrets** (Settings → Secrets)
4. Add a `setup.sh` if you need to precompute models on first deploy

---

## 📋 Tech Stack

- **Streamlit** — UI framework
- **scikit-learn** — TF-IDF, cosine similarity
- **SciPy** — Sparse SVD (`svds`)
- **pandas / numpy** — Data processing
- **TMDB API** — Poster images and metadata
- **MovieLens** — Training data

---

## 📄 License
MIT — use it, extend it, put it on your resume.

---

*Built as a portfolio project demonstrating production-grade ML system design.*

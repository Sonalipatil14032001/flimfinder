# ◈ FilmFinder
### Two-Stage Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=flat-square&logo=pytorch)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![TMDB](https://img.shields.io/badge/Dataset-TMDB-01B4E4?style=flat-square)

> A production-style recommendation pipeline combining **LightGCN graph neural network** for candidate retrieval and **XGBoost learning-to-rank** for re-ranking — trained on Bollywood, South Indian, and Korean cinema.

---

## 📸 Demo

| Parasite (2019) | Pushpa: The Rise |
|---|---|
| ![Parasite Results](assets/parasite.png) | ![Pushpa Results](assets/pushpa.png) |

---

## 🧠 How It Works

FilmFinder uses a **two-stage pipeline** — the same architecture used by YouTube, Pinterest, and Netflix at scale:

```
User Query (Movie Title)
         │
         ▼
┌─────────────────────────┐
│   Stage 1: LightGCN     │  ← Graph Neural Network
│   Retrieval (top-50)    │    Cosine similarity in
│                         │    embedding space
└──────────┬──────────────┘
           │ 50 candidates
           ▼
┌─────────────────────────┐
│   Stage 2: XGBoost      │  ← Learning-to-Rank
│   Re-ranking (top-15)   │    rank:ndcg objective
│                         │    8 engineered features
└──────────┬──────────────┘
           │
           ▼
   15 Ranked Recommendations
   (with GCN score + XGB score)
```

---

## 🔬 Stage 1 — LightGCN Graph Retrieval

LightGCN (He et al., 2020) simplifies Graph Convolutional Networks by removing feature transformation and non-linear activation — keeping only neighbourhood aggregation.

**Graph structure:**
- **Nodes:** Users + Movies (bipartite graph)
- **Edges:** User-movie rating interactions
- **Node features:** Genre one-hot + Language one-hot + Popularity + Vote average (26-dim)

**Propagation rule:**
```
E^(k+1) = D^(-1/2) · A · D^(-1/2) · E^(k)
```

**Final embedding** = mean pooling across all K layers (including layer 0):
```
E_final = (1/K+1) Σ E^(k)
```

**Training:** BPR (Bayesian Personalised Ranking) loss — maximises score(user, positive_item) - score(user, negative_item)

**Results:**
| Metric | Score |
|---|---|
| Recall@15 | 0.1233 |
| Precision@15 | 0.0313 |
| NDCG@15 | 0.0935 |
| BPR Loss (100 epochs) | 0.5096 |

---

## 🌲 Stage 2 — XGBoost Re-ranker

XGBoost re-ranks the 50 LightGCN candidates using engineered features and a `rank:ndcg` learning-to-rank objective.

**Features engineered per (query, candidate) pair:**

| Feature | Description |
|---|---|
| `gcn_score` | Cosine similarity from LightGCN embeddings |
| `genre_overlap` | Jaccard similarity between movie genres |
| `cand_popularity` | log(number of ratings) |
| `cand_avg_rating` | Mean rating score |
| `cand_rating_std` | Rating variance (penalises divisive films) |
| `cand_recency` | How recently the movie was rated |
| `popularity_diff` | Absolute popularity difference from query |
| `rating_diff` | Absolute rating difference from query |

**Feature Importances:**
```
genre_overlap      0.515  ████████████████████
gcn_score          0.480  ███████████████████
cand_popularity    0.002  ▏
cand_avg_rating    0.001  ▏
...
```

---

## 🎬 Dataset

Custom dataset built using the **TMDB API** — covering 6 languages:

| Language | Movies |
|---|---|
| Hindi (Bollywood) | 300 |
| Korean | 300 |
| Tamil | 109 |
| Telugu | 42 |
| Malayalam | 41 |
| Kannada | 8 |
| **Total** | **800 movies** |

Features per movie: title, genres, language, popularity, vote_average, vote_count, poster_path, overview, tmdb_id

Ratings: **74,939 simulated ratings** across **2,000 users** — weighted by popularity and language preference.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Graph Model | PyTorch (LightGCN from scratch, MPS/GPU) |
| Ranker | XGBoost (rank:ndcg) |
| Data | TMDB API + custom collector |
| Backend | Flask REST API |
| Frontend | React + Vite |
| Posters | TMDB Image CDN |

---

## 📁 Project Structure

```
filmfinder/
├── backend/
│   ├── app.py                          # Flask API (3 endpoints)
│   ├── requirements.txt
│   ├── data/
│   │   ├── download.py                 # MovieLens downloader
│   │   └── tmdb_collector.py           # Custom Indian/Korean dataset builder
│   ├── stage1_retrieval/
│   │   └── lightgcn.py                 # LightGCN model + BPR training + retrieval
│   └── stage2_ranking/
│       └── xgboost_ranker.py           # Feature engineering + XGBoost ranker
└── frontend/
    ├── src/
    │   ├── App.jsx                     # Main React component
    │   ├── index.css                   # Global dark theme styles
    │   └── main.jsx                    # Entry point
    ├── index.html
    └── package.json
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Node.js 18+
- TMDB API key (free at [themoviedb.org](https://www.themoviedb.org/settings/api))

### 1. Clone the repo
```bash
git clone https://github.com/your-username/filmfinder.git
cd filmfinder
```

### 2. Backend setup
```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
# Create backend/.env
TMDB_API_KEY=your_tmdb_read_access_token_here
```

### 4. Build the dataset
```bash
python data/tmdb_collector.py
```

### 5. Train Stage 1 — LightGCN
```bash
python stage1_retrieval/lightgcn.py --epochs 100
```

### 6. Train Stage 2 — XGBoost Ranker
```bash
python -c "from stage2_ranking.xgboost_ranker import train_ranker; train_ranker()"
```

### 7. Start Flask API
```bash
python app.py
# Runs on http://localhost:8080
```

### 8. Start React frontend
```bash
cd ../frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

---

## 🔌 API Endpoints

### `POST /recommend`
```json
Request:  { "title": "Parasite", "n": 15 }
Response: {
  "query": "Parasite (2019)",
  "pipeline": "LightGCN → XGBoost",
  "candidates_pool": 50,
  "recommendations": [
    { "rank": 1, "title": "The Handmaiden (2016)", "gcn_score": 0.994, "xgb_score": 3.159 },
    ...
  ]
}
```

### `GET /search?q=para`
```json
{ "results": ["Parasite (2019)", "Paranormal Activity (2007)", ...] }
```

### `POST /movie-details`
```json
Request:  { "titles": ["Parasite (2019)"] }
Response: {
  "Parasite (2019)": {
    "poster": "https://image.tmdb.org/t/p/w300/...",
    "overview": "All unemployed, Ki-taek's family...",
    "vote_avg": 8.49,
    "language": "Korean"
  }
}
```

---

## 📊 Model Training Arguments

```bash
python stage1_retrieval/lightgcn.py \
  --epochs 100 \
  --embedding_dim 64 \
  --num_layers 3 \
  --lr 0.001
```

---

## 🔮 Roadmap

- [ ] Deploy Flask on Render
- [ ] Deploy frontend on Vercel
- [ ] Add user accounts + personal rating history
- [ ] Train on 25M MovieLens for better embeddings
- [ ] Add vector database (FAISS) for faster retrieval
- [ ] Hybrid filtering: content + collaborative

---

## 👩‍💻 Author

**Sonali Patil** — MS Computer Science, UAB | Data Analyst & ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/sonali-patil-a119611a1/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/Sonalipatil14032001/)

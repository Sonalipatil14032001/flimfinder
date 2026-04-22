import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"


def build_item_features(data_dir):
    """
    Build a feature table for every movie.
    These features are what XGBoost uses to re-rank LightGCN candidates.

    Features we engineer:
        log_popularity  — log(number of ratings) — popular movies rank higher
        avg_rating      — mean rating across all users
        rating_std      — std of ratings — controversial movies penalised
        recency_score   — how recently the movie has been rated (0 to 1)
        genre_*         — one-hot encoded genre flags (18 genres)
    """
    # Use Indian dataset if available, otherwise fall back to MovieLens
    if (data_dir / "indian-movies").exists():
        folder  = data_dir / "indian-movies"
        print("[→] Using Indian + Korean Movies dataset")
    else:
        folder  = data_dir / "ml-latest-small"
        print("[→] Using MovieLens Small dataset")

    ratings = pd.read_csv(folder / "ratings.csv")
    movies  = pd.read_csv(folder / "movies.csv")
    
    # Aggregate rating statistics per movie
    stats = ratings.groupby("movieId").agg(
        popularity  = ("rating", "count"),
        avg_rating  = ("rating", "mean"),
        rating_std  = ("rating", "std"),
        latest_ts   = ("timestamp", "max"),
    ).reset_index()
    stats["rating_std"]    = stats["rating_std"].fillna(0)
    stats["log_popularity"] = np.log1p(stats["popularity"])

    # Recency score: 0 = oldest, 1 = most recently rated
    max_ts = ratings["timestamp"].max()
    min_ts = ratings["timestamp"].min()
    stats["recency_score"] = (stats["latest_ts"] - min_ts) / (max_ts - min_ts + 1)

    # Genre one-hot encoding
    all_genres = set()
    for g in movies["genres"].str.split("|"):
        all_genres.update(g)
    all_genres.discard("(no genres listed)")

    for genre in sorted(all_genres):
        col = f"genre_{genre.replace('-', '_')}"
        movies[col] = movies["genres"].str.contains(genre, regex=False).astype(int)

    movies = movies.merge(stats, on="movieId", how="left").fillna(0)
    print(f"[✓] Item features: {len(movies)} movies × {len(movies.columns)} columns")
    return movies


def genre_jaccard(genres_a, genres_b):
    """Jaccard similarity between two genre strings — measures genre overlap."""
    set_a = set(genres_a.split("|")) - {"(no genres listed)"}
    set_b = set(genres_b.split("|")) - {"(no genres listed)"}
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def build_pair_features(query_row, candidate_row, gcn_score):
    """
    Build a feature vector for a (query movie, candidate movie) pair.
    This is what XGBoost sees for each candidate during ranking.
    """
    return {
        "gcn_score":       gcn_score,
        "cand_popularity": candidate_row["log_popularity"],
        "cand_avg_rating": candidate_row["avg_rating"],
        "cand_rating_std": candidate_row["rating_std"],
        "cand_recency":    candidate_row["recency_score"],
        "genre_overlap":   genre_jaccard(
            query_row["genres"],
            candidate_row["genres"]
        ),
        "popularity_diff": abs(query_row["log_popularity"] - candidate_row["log_popularity"]),
        "rating_diff":     abs(query_row["avg_rating"]     - candidate_row["avg_rating"]),
    }


def build_training_pairs(item_features):
    """
    Build (query, candidate, label) training pairs for learning-to-rank.

    Strategy:
        - Sample 300 movies as queries
        - Positives: movies with genre_jaccard > 0.3 (genuinely similar)
        - Negatives: movies with genre_jaccard <= 0.1 (clearly dissimilar)
        - Label = 1 for positives, 0 for negatives
    """
    print("[→] Building training pairs...")
    rows, groups = [], []

    query_sample = item_features.sample(n=300, random_state=42)

    for group_id, (_, query_row) in enumerate(query_sample.iterrows()):
        candidates = item_features[item_features["movieId"] != query_row["movieId"]].copy()
        candidates["jaccard"] = candidates["genres"].apply(
            lambda g: genre_jaccard(query_row["genres"], g)
        )

        positives = candidates[candidates["jaccard"] > 0.3].sample(
            n=min(10, len(candidates[candidates["jaccard"] > 0.3])),
            random_state=42
        )
        negatives = candidates[candidates["jaccard"] <= 0.1].sample(
            n=min(10, len(candidates[candidates["jaccard"] <= 0.1])),
            random_state=42
        )

        for _, cand in positives.iterrows():
            feat = build_pair_features(query_row, cand, gcn_score=0.8)
            feat["label"] = 1
            rows.append(feat)
            groups.append(group_id)

        for _, cand in negatives.iterrows():
            feat = build_pair_features(query_row, cand, gcn_score=0.2)
            feat["label"] = 0
            rows.append(feat)
            groups.append(group_id)

    df = pd.DataFrame(rows)
    print(f"[✓] {len(df)} training pairs across {len(query_sample)} queries")
    return df, groups


def train_ranker():
    """Train XGBoost ranker and save artifacts."""
    print("[→] Building item features...")
    item_features = build_item_features(DATA_DIR)

    train_df, groups = build_training_pairs(item_features)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X = train_df[feature_cols].values
    y = train_df["label"].values

    # Group sizes needed by XGBoost ranker
    # Each group = one query movie with its candidates
    _, counts = np.unique(groups, return_counts=True)

    print("[→] Training XGBoost ranker with rank:ndcg objective...")
    model = xgb.XGBRanker(
        objective        = "rank:ndcg",
        n_estimators     = 200,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        tree_method      = "hist",
        random_state     = 42,
        verbosity        = 0,
    )

    model.fit(X, y, group=counts)

    # Print feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    print("\n[✓] Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"     {feat:<25} {imp:.4f}")

    # Save artifacts
    artifacts = {
        "model":         model,
        "feature_cols":  feature_cols,
        "item_features": item_features,
    }
    out_path = MODEL_DIR / "xgboost_ranker.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n[✓] XGBoost ranker saved to {out_path}")
    return artifacts

def rerank(candidates, query_title):
    """
    Takes LightGCN candidates and re-ranks them using XGBoost.

    Args:
        candidates:  list of dicts from get_candidates()
        query_title: the matched movie title (to get its features)

    Returns:
        Re-ranked list with xgb_score and final_rank added
    """
    ranker_path = MODEL_DIR / "xgboost_ranker.pkl"
    with open(ranker_path, "rb") as f:
        artifacts = pickle.load(f)

    model         = artifacts["model"]
    feature_cols  = artifacts["feature_cols"]
    item_features = artifacts["item_features"]

    # Get query movie's feature row
    query_rows = item_features[item_features["title"] == query_title]
    if query_rows.empty:
        # Fallback: return LightGCN order unchanged
        for i, c in enumerate(candidates, 1):
            c["xgb_score"]  = None
            c["final_rank"] = i
        return candidates

    query_row = query_rows.iloc[0]

    # Build feature vector for each candidate
    rows = []
    for cand in candidates:
        cand_rows = item_features[item_features["title"] == cand["title"]]
        if cand_rows.empty:
            feat = {col: 0.0 for col in feature_cols}
            feat["gcn_score"] = cand["gcn_score"]
        else:
            feat = build_pair_features(query_row, cand_rows.iloc[0], cand["gcn_score"])
        rows.append(feat)

    X      = pd.DataFrame(rows)[feature_cols].fillna(0).values
    scores = model.predict(X)

    # Attach scores and sort
    for i, cand in enumerate(candidates):
        cand["xgb_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: x["xgb_score"], reverse=True)
    for rank, cand in enumerate(reranked, 1):
        cand["final_rank"] = rank

    return reranked
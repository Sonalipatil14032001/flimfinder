import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class LightGCN(nn.Module):
    """
    LightGCN: Simplified Graph Convolutional Network for Recommendation.
    
    Key idea: Remove feature transformation and non-linear activation from GCN.
    Only keep neighbourhood aggregation across K layers.
    Final embedding = mean of all layer outputs (including layer 0).
    
    Paper: He et al. 2020 - LightGCN: Simplifying and Powering GCN for Recommendation
    """

    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_layers=3, item_feature_dim=0):
        super().__init__()
        self.num_users        = num_users
        self.num_items        = num_items
        self.embedding_dim    = embedding_dim
        self.num_layers       = num_layers
        self.item_feature_dim = item_feature_dim

        # Base learnable embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Feature projection: genre + language + stats → embedding_dim
        if item_feature_dim > 0:
            self.feature_proj = nn.Linear(item_feature_dim, embedding_dim)
            nn.init.xavier_uniform_(self.feature_proj.weight)
        else:
            self.feature_proj = None

    def forward(self, edge_index, item_features=None):
        """
        Propagate embeddings through graph.
        If item_features provided, add projected features to item embeddings.
        """
        # Start with base item embeddings
        item_emb = self.item_embedding.weight

        # Add projected node features to item embeddings
        if self.feature_proj is not None and item_features is not None:
            item_emb = item_emb + self.feature_proj(item_features)

        all_emb = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        layer_outputs = [all_emb]

        num_nodes = self.num_users + self.num_items
        row, col  = edge_index

        deg = torch.zeros(num_nodes, device=DEVICE)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=DEVICE))
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=DEVICE))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        for _ in range(self.num_layers):
            norm_emb = all_emb * deg_inv_sqrt.unsqueeze(1)
            agg = torch.zeros_like(all_emb)
            agg.scatter_add_(0, col.unsqueeze(1).expand(-1, self.embedding_dim),
                             norm_emb[row])
            agg.scatter_add_(0, row.unsqueeze(1).expand(-1, self.embedding_dim),
                             norm_emb[col])
            all_emb = agg * deg_inv_sqrt.unsqueeze(1)
            layer_outputs.append(all_emb)

        final = torch.stack(layer_outputs, dim=1).mean(dim=1)
        return final[:self.num_users], final[self.num_users:]

    def bpr_loss(self, users, pos_items, neg_items, edge_index, item_features=None):
        user_emb, item_emb = self.forward(edge_index, item_features)

        u   = user_emb[users]
        pos = item_emb[pos_items]
        neg = item_emb[neg_items]

        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

        reg = (self.user_embedding(users).norm(2).pow(2) +
               self.item_embedding(pos_items).norm(2).pow(2) +
               self.item_embedding(neg_items).norm(2).pow(2)) / len(users)

        return loss + 1e-4 * reg
    
def load_and_prepare(data_dir, dataset="indian"):
    """
    Load dataset and prepare for training.
    Supports 'indian' (TMDB) and 'small' (MovieLens) datasets.
    """
    if dataset == "indian":
        folder = data_dir / "indian-movies"
        print("[→] Using Indian + Korean Movies dataset (TMDB)")
    else:
        folder = data_dir / "ml-latest-small"
        print("[→] Using MovieLens Small dataset")

    ratings = pd.read_csv(folder / "ratings.csv")
    movies  = pd.read_csv(folder / "movies.csv")

    # Filter: keep users and movies with at least 3 ratings
    ratings = ratings[ratings.groupby('userId')['userId'].transform('count') >= 3]
    ratings = ratings[ratings.groupby('movieId')['movieId'].transform('count') >= 3]

    # Re-index to consecutive 0-based IDs
    user_ids = {uid: i for i, uid in enumerate(ratings['userId'].unique())}
    item_ids = {mid: i for i, mid in enumerate(ratings['movieId'].unique())}

    ratings['user_idx'] = ratings['userId'].map(user_ids)
    ratings['item_idx'] = ratings['movieId'].map(item_ids)

    # Build item index → title mapping
    id_to_title = dict(zip(movies['movieId'], movies['title']))
    item_idx_to_title = {
        item_ids[mid]: id_to_title[mid]
        for mid in item_ids
        if mid in id_to_title
    }

    # Also store language info if available
    item_idx_to_lang = {}
    if 'language' in movies.columns:
        id_to_lang = dict(zip(movies['movieId'], movies['language']))
        item_idx_to_lang = {
            item_ids[mid]: id_to_lang[mid]
            for mid in item_ids
            if mid in id_to_lang
        }

    train_df, test_df = train_test_split(ratings, test_size=0.1, random_state=42)

    print(f"[✓] Users: {len(user_ids)} | Items: {len(item_ids)}")
    print(f"[✓] Train: {len(train_df)} | Test: {len(test_df)}")

    return train_df, test_df, user_ids, item_ids, item_idx_to_title, item_idx_to_lang

def build_item_feature_matrix(movies_df, item_ids):
    """
    Build item node feature matrix from genre + language + stats.
    Shape: [num_items, feature_dim]

    Features:
        - Genre one-hot (18 genres)
        - Language one-hot (6 languages)
        - Normalized popularity
        - Normalized vote average
    """
    GENRES = [
        "Action", "Adventure", "Animation", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History",
        "Horror", "Music", "Mystery", "Romance", "SciFi",
        "Thriller", "War", "Western"
    ]
    LANGUAGES = [
        "Hindi (Bollywood)", "Tamil", "Telugu",
        "Malayalam", "Kannada", "Korean"
    ]

    num_items   = len(item_ids)
    feature_dim = len(GENRES) + len(LANGUAGES) + 2
    features    = np.zeros((num_items, feature_dim), dtype=np.float32)

    # Normalize popularity and vote_avg
    max_pop  = movies_df["popularity"].max() + 1e-8
    max_vote = movies_df["vote_avg"].max()   + 1e-8

    for _, row in movies_df.iterrows():
        mid = row["movieId"]
        if mid not in item_ids:
            continue
        idx = item_ids[mid]

        # Genre one-hot
        movie_genres = row.get("genres", "").split("|")
        for g in movie_genres:
            if g in GENRES:
                features[idx, GENRES.index(g)] = 1.0

        # Language one-hot
        lang = row.get("language", "")
        if lang in LANGUAGES:
            features[idx, len(GENRES) + LANGUAGES.index(lang)] = 1.0

        # Normalized stats
        features[idx, -2] = row.get("popularity", 0) / max_pop
        features[idx, -1] = row.get("vote_avg",   0) / max_vote

    print(f"[✓] Item feature matrix: {features.shape}")
    return features, feature_dim

def build_edge_index(df, num_users):
    """
    Build bipartite graph edge index.
    Item nodes are offset by num_users so all nodes have unique IDs.
    
    Example: user_idx=0, item_idx=0 → edge (0, num_users+0)
    """
    users = torch.tensor(df['user_idx'].values, dtype=torch.long)
    items = torch.tensor(df['item_idx'].values + num_users, dtype=torch.long)

    # Stack into [2, num_edges] and make undirected (add reverse edges)
    edge_index = torch.stack([users, items], dim=0)
    reverse    = torch.stack([items, users], dim=0)
    edge_index = torch.cat([edge_index, reverse], dim=1)

    return edge_index.to(DEVICE)


def sample_negatives(df, num_items, n_samples):
    """
    Sample negative items (movies the user has NOT rated) for BPR training.
    For each positive (user, item) pair, we sample one random negative item.
    """
    # Build set of known positives for fast lookup
    user_item_set = set(zip(df['user_idx'], df['item_idx']))

    users, pos_items, neg_items = [], [], []
    sample = df.sample(n=n_samples, replace=True, random_state=42)

    for _, row in sample.iterrows():
        u = int(row['user_idx'])
        p = int(row['item_idx'])

        # Keep sampling until we find an item the user hasn't rated
        n = np.random.randint(0, num_items)
        while (u, n) in user_item_set:
            n = np.random.randint(0, num_items)

        users.append(u)
        pos_items.append(p)
        neg_items.append(n)

    return (
        torch.tensor(users,     dtype=torch.long).to(DEVICE),
        torch.tensor(pos_items, dtype=torch.long).to(DEVICE),
        torch.tensor(neg_items, dtype=torch.long).to(DEVICE),
    )

def evaluate(model, edge_index, test_df, num_users, num_items, K=15, item_features=None):
    """
    Evaluate LightGCN using ranking metrics on the test set.

    Metrics:
        Recall@K    — of all movies user rated, how many did we retrieve in top-K?
        Precision@K — of K movies recommended, how many did user actually rate?
        NDCG@K      — rewards correct items ranked higher
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index, item_features)
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()

    # Build ground truth: user → set of rated items in test set
    ground_truth = {}
    for _, row in test_df.iterrows():
        u = int(row['user_idx'])
        i = int(row['item_idx'])
        ground_truth.setdefault(u, set()).add(i)

    recalls, precisions, ndcgs = [], [], []

    for user, true_items in ground_truth.items():
        if user >= num_users:
            continue

        # Score all items for this user
        u_vec  = user_emb[user]
        scores = item_emb @ u_vec
        top_k  = np.argsort(scores)[::-1][:K]

        hits = [1 if item in true_items else 0 for item in top_k]

        # Recall@K
        recall = sum(hits) / min(len(true_items), K)
        recalls.append(recall)

        # Precision@K
        precision = sum(hits) / K
        precisions.append(precision)

        # NDCG@K
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), K)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)

    results = {
        f"Recall@{K}":    round(np.mean(recalls),    4),
        f"Precision@{K}": round(np.mean(precisions), 4),
        f"NDCG@{K}":      round(np.mean(ndcgs),      4),
    }

    print(f"\n[✓] Evaluation Metrics (K={K}):")
    for metric, value in results.items():
        print(f"     {metric:<15} {value}")

    return results

def train(epochs=50, embedding_dim=64, num_layers=3, lr=1e-3, batch_size=2048):
    """
    Full training loop for LightGCN.
    Saves model artifacts to backend/model/ after training.
    """
    print("[→] Loading data...")
    train_df, test_df, user_ids, item_ids, item_idx_to_title, item_idx_to_lang = load_and_prepare(DATA_DIR, dataset="indian")

    num_users = len(user_ids)
    num_items = len(item_ids)

    edge_index = build_edge_index(train_df, num_users)

    # Load movies for node features
    if (DATA_DIR / "indian-movies").exists():
        movies_df = pd.read_csv(DATA_DIR / "indian-movies" / "movies.csv")
    else:
        movies_df = pd.read_csv(DATA_DIR / "ml-latest-small" / "movies.csv")

    item_features_np, feature_dim = build_item_feature_matrix(movies_df, item_ids)
    item_features = torch.tensor(item_features_np, dtype=torch.float32).to(DEVICE)

    model     = LightGCN(num_users, num_items, embedding_dim,
                         num_layers, item_feature_dim=feature_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"[→] Training LightGCN for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        users, pos_items, neg_items = sample_negatives(train_df, num_items, batch_size)

        optimizer.zero_grad()
        loss = model.bpr_loss(users, pos_items, neg_items, edge_index, item_features)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} — BPR Loss: {loss.item():.4f}")

    # Extract final embeddings for inference
    print("[→] Extracting final embeddings...")
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index, item_features)
        user_emb = user_emb.cpu().numpy()
        item_emb = item_emb.cpu().numpy()

    # Save everything needed for inference
    artifacts = {
        "user_embeddings":   user_emb,
        "item_embeddings":   item_emb,
        "user_ids":          user_ids,
        "item_ids":          item_ids,
        "item_idx_to_title": item_idx_to_title,
        "item_idx_to_lang":  item_idx_to_lang,
        "num_users":         num_users,
        "num_items":         num_items,
        "embedding_dim":     embedding_dim,
        "num_layers":        num_layers,
    }

    out_path = MODEL_DIR / "lightgcn_artifacts.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)

    # Evaluate on test set
    print("[→] Evaluating on test set...")
    metrics = evaluate(model, edge_index, test_df, num_users, num_items, K=15, item_features=item_features)
    artifacts["eval_metrics"] = metrics

    out_path = MODEL_DIR / "lightgcn_artifacts.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\n[✓] Training complete! Artifacts saved to {out_path}")
    return artifacts


def get_candidates(movie_title, top_k=50):
    """
    Given a movie title, retrieve top-K similar movies
    using cosine similarity between LightGCN item embeddings.
    """
    from difflib import get_close_matches

    artifacts_path = MODEL_DIR / "lightgcn_artifacts.pkl"
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    item_emb          = artifacts["item_embeddings"]
    item_idx_to_title = artifacts["item_idx_to_title"]

    # Build reverse lookup: title → index
    title_to_idx = {v: k for k, v in item_idx_to_title.items()}

    # Fuzzy match the query title
    matches = get_close_matches(
        movie_title.lower(),
        [t.lower() for t in title_to_idx.keys()],
        n=1, cutoff=0.4
    )
    if not matches:
        return None, None

    lower_map     = {t.lower(): t for t in title_to_idx}
    matched_title = lower_map[matches[0]]
    query_idx     = title_to_idx[matched_title]
    query_vec     = item_emb[query_idx]

    # Cosine similarity between query and all items
    norms = np.linalg.norm(item_emb, axis=1) + 1e-8
    sims  = (item_emb @ query_vec) / (norms * (np.linalg.norm(query_vec) + 1e-8))

    # Get top-K (excluding the query movie itself)
    top_indices = np.argsort(sims)[::-1][1:top_k + 1]

    candidates = [
        {
            "item_idx":  int(idx),
            "title":     item_idx_to_title.get(int(idx), f"Movie {idx}"),
            "gcn_score": float(sims[idx]),
        }
        for idx in top_indices
    ]

    return candidates, matched_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--embedding_dim", type=int,   default=64)
    parser.add_argument("--num_layers",    type=int,   default=3)
    parser.add_argument("--lr",            type=float, default=1e-3)
    args = parser.parse_args()

    train(
        epochs        = args.epochs,
        embedding_dim = args.embedding_dim,
        num_layers    = args.num_layers,
        lr            = args.lr,
    )
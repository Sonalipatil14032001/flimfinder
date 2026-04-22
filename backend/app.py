import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from stage1_retrieval.lightgcn import get_candidates
from stage2_ranking.xgboost_ranker import rerank
import pickle
from pathlib import Path

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).parent / "model"


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "service": "FilmFinder v2 — LightGCN + XGBoost Pipeline",
        "stages":  ["LightGCN Retrieval (top-50)", "XGBoost Re-ranking (top-15)"],
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True)
    if not body or "title" not in body:
        return jsonify({"error": "Request must include 'title'."}), 400

    title = body["title"].strip()
    n     = min(int(body.get("n", 15)), 50)

    if not title:
        return jsonify({"error": "Title cannot be empty."}), 400

    # Stage 1: LightGCN retrieval
    candidates, matched_title = get_candidates(title, top_k=50)
    if not candidates:
        return jsonify({"error": f"Movie '{title}' not found."}), 404

    # Stage 2: XGBoost re-ranking
    reranked = rerank(candidates, matched_title)

    return jsonify({
        "query":           matched_title,
        "pipeline":        "LightGCN → XGBoost",
        "candidates_pool": len(candidates),
        "recommendations": [
            {
                "rank":      r["final_rank"],
                "title":     r["title"],
                "gcn_score": round(r["gcn_score"], 4),
                "xgb_score": round(r["xgb_score"], 4) if r.get("xgb_score") else None,
            }
            for r in reranked[:n]
        ],
    }), 200


@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify({"results": []})

    artifacts_path = MODEL_DIR / "lightgcn_artifacts.pkl"
    if not artifacts_path.exists():
        return jsonify({"results": []})

    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    all_titles = list(artifacts["item_idx_to_title"].values())
    matches    = [t for t in all_titles if q in t.lower()][:10]
    return jsonify({"results": matches})

@app.route("/movie-details", methods=["POST"])
def movie_details():
    """
    Returns poster URL and details for a list of movie titles.
    POST body: { "titles": ["RRR (2022)", "Parasite (2019)"] }
    """
    TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"

    body   = request.get_json(silent=True)
    titles = body.get("titles", [])

    movies_df = pd.read_csv(
        MODEL_DIR.parent / "data" / "indian-movies" / "movies.csv"
    )

    result = {}
    for title in titles:
        row = movies_df[movies_df["title"] == title]
        if row.empty:
            result[title] = {"poster": None, "overview": "", "vote_avg": None}
        else:
            r = row.iloc[0]
            poster = f"{TMDB_IMAGE_BASE}{r['poster_path']}" if r.get("poster_path") else None
            result[title] = {
                "poster":   poster,
                "overview": r.get("overview", ""),
                "vote_avg": r.get("vote_avg", None),
                "language": r.get("language", ""),
            }

    return jsonify(result), 200

if __name__ == "__main__":
    app.run(debug=True, port=8080)
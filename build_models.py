"""
build_models.py — Precompute and cache both models

Run this once before launching the app:
    python build_models.py

This saves the similarity matrix and SVD model to the artifacts/ folder
so the Streamlit app loads instantly instead of recomputing on every start.
"""
import time
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.content_model import build_content_model
from models.collab_model  import build_collab_model


def main():
    print("\n" + "="*55)
    print("  CineMatch — Model Builder")
    print("="*55)

    # ── Content model ─────────────────────────────────────
    print("\n[1/2] Building content-based model…")
    t0 = time.time()
    movies, sim = build_content_model(force_rebuild=True)
    print(f"      ✓ Done in {time.time()-t0:.1f}s  |  {len(movies):,} movies  |  matrix {sim.shape}")

    # ── Collaborative model ───────────────────────────────
    print("\n[2/2] Building collaborative filtering model (SVD)…")
    t0 = time.time()
    svd = build_collab_model(force_rebuild=True)
    n_movies = len(svd['movie_ids'])
    print(f"      ✓ Done in {time.time()-t0:.1f}s  |  {n_movies:,} movies with ratings")

    print("\n" + "="*55)
    print("  All models built and cached in artifacts/")
    print("  Run the app: streamlit run app.py")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()

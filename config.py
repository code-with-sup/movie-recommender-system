"""
config.py — Central configuration for CineMatch
All constants, paths, and environment settings live here.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MOVIES_PATH  = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
TAGS_PATH    = os.path.join(DATA_DIR, "tags.csv")

SIMILARITY_PATH = os.path.join(ARTIFACT_DIR, "similarity_matrix.pkl")
SVD_PATH        = os.path.join(ARTIFACT_DIR, "svd_model.pkl")

# ── Model Settings ─────────────────────────────────────────────────────────────
NUM_RECOMMENDATIONS      = 8
CONTENT_WEIGHT           = 0.4
COLLAB_WEIGHT            = 0.6
MAX_VECTORIZER_FEATURES  = 5000
SVD_LATENT_FACTORS       = 50
MIN_USER_RATINGS         = 20

# ── TMDB API ───────────────────────────────────────────────────────────────────
# Set your API key as an environment variable: export TMDB_API_KEY="your_key"
# Get a free key at: https://www.themoviedb.org/settings/api
TMDB_API_KEY    = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL   = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_TIMEOUT    = 5  # seconds

# ── UI Settings ────────────────────────────────────────────────────────────────
APP_TITLE       = "CineMatch"
APP_ICON        = "🎬"
APP_DESCRIPTION = "Hybrid AI Movie Recommender — Content + Collaborative Filtering"
GRID_COLS       = 4

"""
app.py — CineMatch: Hybrid AI Movie Recommender
Streamlit entry point. This file handles only UI — all logic lives in models/.
"""
import streamlit as st
import logging
import sys
import os

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────────
from config import (
    NUM_RECOMMENDATIONS, CONTENT_WEIGHT, COLLAB_WEIGHT,
    TMDB_API_KEY, APP_TITLE
)
from models.content_model import build_content_model
from models.collab_model  import build_collab_model
from models.hybrid        import recommend, get_movie_details
from utils.tmdb_api       import search_movie, fetch_batch

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #0d0d14;
    --surface:   #16161f;
    --surface2:  #1e1e2a;
    --accent:    #e94560;
    --accent2:   #f5a623;
    --text:      #e8e8f0;
    --muted:     #8888a8;
    --radius:    12px;
    --card-w:    200px;
}

/* ── Global ── */
.stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
.block-container { padding: 2rem 3rem; max-width: 1400px; }

/* ── Header ── */
.cinema-header {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #ffffff10;
    margin-bottom: 2.5rem;
}
.cinema-header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, #e94560 0%, #f5a623 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1;
}
.cinema-header p {
    color: var(--muted);
    font-size: 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Search bar ── */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1.5px solid #ffffff18 !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-size: 1.1rem !important;
    padding: 0.85rem 1.2rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(233,69,96,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e94560, #c73652);
    color: #fff;
    border: none;
    border-radius: var(--radius);
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.75rem 2rem;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(233,69,96,0.4);
}

/* ── Now Searching card ── */
.query-card {
    background: var(--surface);
    border: 1px solid #ffffff12;
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.query-label { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
.query-title { font-size: 1.4rem; font-weight: 600; margin-top: 0.15rem; }
.query-genres { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }

/* ── Section heading ── */
.section-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.1em;
    color: var(--text);
    margin: 2rem 0 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #ffffff14;
    margin-left: 0.5rem;
}

/* ── Movie cards ── */
.movie-card {
    background: var(--surface);
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid #ffffff0a;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
}
.movie-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    border-color: var(--accent);
    z-index: 10;
}
.movie-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    display: block;
}
.movie-card-body {
    padding: 0.75rem;
}
.movie-rank {
    position: absolute;
    top: 8px; left: 8px;
    background: var(--accent);
    color: #fff;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    padding: 2px 8px;
    border-radius: 6px;
}
.movie-score-badge {
    position: absolute;
    top: 8px; right: 8px;
    background: rgba(0,0,0,0.8);
    color: var(--accent2);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid rgba(245,166,35,0.4);
}
.movie-title-text {
    font-weight: 600;
    font-size: 0.88rem;
    line-height: 1.3;
    margin-bottom: 0.3rem;
    color: var(--text);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.movie-meta {
    font-size: 0.75rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.3rem;
}
.genre-tag {
    background: var(--surface2);
    color: var(--muted);
    font-size: 0.68rem;
    padding: 2px 7px;
    border-radius: 20px;
    border: 1px solid #ffffff10;
}
.star-rating {
    color: var(--accent2);
    font-size: 0.78rem;
}
.overview-text {
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.5;
    margin-top: 0.4rem;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid #ffffff0a;
}
.sidebar-section {
    background: var(--surface2);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.9rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* ── Metrics ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid #ffffff0a;
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    flex: 1;
    text-align: center;
}
.metric-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--accent);
    letter-spacing: 0.05em;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.15rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 4px;
    gap: 4px;
    border: 1px solid #ffffff0a;
}
.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* ── Spinner / loading ── */
.stSpinner { color: var(--accent) !important; }

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { color: var(--accent); }

/* ── Info / warning banners ── */
.stAlert {
    background: var(--surface2) !important;
    border-radius: var(--radius) !important;
    border: 1px solid #ffffff12 !important;
}

/* ── Footer ── */
.cinema-footer {
    text-align: center;
    padding: 3rem 0 1rem;
    color: var(--muted);
    font-size: 0.8rem;
    border-top: 1px solid #ffffff08;
    margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached model loading ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load both models once per Streamlit session — cached in memory."""
    with st.spinner("🎬 Loading CineMatch AI engine…"):
        movies, similarity = build_content_model()
        svd_data           = build_collab_model()
    return movies, similarity, svd_data


# ── Helpers ───────────────────────────────────────────────────────────────────
def render_movie_card(rec: dict, tmdb: dict) -> str:
    """Return HTML for a single movie card."""
    poster     = tmdb.get('poster_url', '')
    rating     = tmdb.get('vote_average', 0)
    year       = tmdb.get('release_date', '')
    overview   = tmdb.get('overview', '')
    genres_raw = rec.get('genres', '')
    genre_tags = ' '.join(
        f'<span class="genre-tag">{g.strip()}</span>'
        for g in genres_raw.split()[:2] if g.strip()
    )
    stars = '★' * min(int(rating / 2), 5) if rating else ''
    score_pct = int(rec['score'] * 100)

    return f"""
    <div class="movie-card">
        <div class="movie-rank">#{rec['rank']}</div>
        <div class="movie-score-badge">Match {score_pct}%</div>
        <img class="movie-poster" src="{poster}" alt="{rec['title']}"
             onerror="this.src='https://via.placeholder.com/300x450/1a1a2e/e94560?text=No+Image'"/>
        <div class="movie-card-body">
            <div class="movie-title-text">{rec['title']}</div>
            <div class="movie-meta">
                <div>{genre_tags}</div>
                <span class="star-rating">{stars}</span>
            </div>
            {f'<div class="overview-text">{overview}</div>' if overview else ''}
            <div class="movie-meta" style="margin-top:0.5rem;">
                <span style="color:#555;">{year}</span>
                <span style="color:#555;">⭐ {rating}</span>
            </div>
        </div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(movies):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem 0 1rem;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;
                        background:linear-gradient(135deg,#e94560,#f5a623);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                CINEMATCH
            </div>
            <div style="color:#8888a8;font-size:0.7rem;letter-spacing:0.15em;
                        text-transform:uppercase;margin-top:-4px;">
                AI Recommender
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">⚙ Model Settings</div>', unsafe_allow_html=True)

        content_w = st.slider(
            "Content weight", 0.0, 1.0, CONTENT_WEIGHT, 0.05,
            help="How much to rely on genre/tag similarity"
        )
        collab_w = round(1.0 - content_w, 2)
        st.caption(f"Collaborative weight: **{collab_w}**")

        n_recs = st.slider(
            "Recommendations", 4, 12, NUM_RECOMMENDATIONS,
            help="Number of movies to show"
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-title">🎯 Quick Search</div>', unsafe_allow_html=True)
        popular = [
            "Toy Story", "The Matrix", "Inception",
            "Pulp Fiction", "The Dark Knight", "Forrest Gump",
            "Interstellar", "Goodfellas", "Fight Club",
        ]
        for title in popular:
            if st.button(title, key=f"quick_{title}"):
                st.session_state['query'] = title

        st.markdown("---")

        if not TMDB_API_KEY:
            st.warning(
                "🔑 **TMDB key not set**\n\n"
                "Posters won't load. Add your key to `.env` or run:\n"
                "```\nexport TMDB_API_KEY=your_key\n```\n"
                "[Get a free key →](https://www.themoviedb.org/settings/api)"
            )

        st.markdown("---")
        st.markdown("""
        <div style="color:#555;font-size:0.72rem;line-height:1.8;">
            <b style="color:#8888a8;">Built with</b><br>
            TF-IDF · Cosine Similarity<br>
            SVD Collaborative Filtering<br>
            TMDB API · MovieLens Dataset<br>
            Streamlit · scikit-learn · SciPy
        </div>
        """, unsafe_allow_html=True)

    return content_w, collab_w, n_recs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Load models ──────────────────────────────────────────────────────────
    try:
        movies, similarity, svd_data = load_models()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    content_w, collab_w, n_recs = render_sidebar(movies)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cinema-header">
        <h1>CINEMATCH</h1>
        <p>Hybrid AI · Content-Based + Collaborative Filtering</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ──────────────────────────────────────────────────────────
    total_movies  = f"{len(movies):,}"
    total_ratings = "25M+"   # MovieLens full dataset
    model_type    = "Hybrid SVD"
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-val">{total_movies}</div>
            <div class="metric-label">Movies</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{total_ratings}</div>
            <div class="metric-label">Ratings</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{model_type}</div>
            <div class="metric-label">Engine</div>
        </div>
        <div class="metric-card">
            <div class="metric-val">{int(content_w*100)}:{int(collab_w*100)}</div>
            <div class="metric-label">Content:Collab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search box ───────────────────────────────────────────────────────────
    tabs = st.tabs(["🔍 Discover", "📖 How it works", "📊 About the data"])

    with tabs[0]:
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            default_query = st.session_state.get('query', '')
            query = st.text_input(
                "Search a movie",
                value=default_query,
                placeholder="e.g. The Dark Knight, Toy Story, Inception…",
                label_visibility="collapsed",
            )
        with col_btn:
            search_clicked = st.button("Search 🎬", use_container_width=True)

        # ── Autocomplete suggestions ──────────────────────────────────────────
        if query and len(query) >= 2:
            suggestions = movies[
                movies['title_clean'].str.contains(query.lower(), na=False)
            ]['title'].head(5).tolist()
            if suggestions and query not in [s.lower() for s in suggestions]:
                with st.expander(f"💡 Did you mean… ({len(suggestions)} suggestions)", expanded=False):
                    cols = st.columns(len(suggestions))
                    for i, sug in enumerate(suggestions):
                        with cols[i]:
                            if st.button(sug, key=f"sug_{i}"):
                                st.session_state['query'] = sug
                                st.rerun()

        # ── Run recommendations ───────────────────────────────────────────────
        if query and (search_clicked or query):
            results = recommend(
                query, movies, similarity, svd_data,
                n=n_recs, content_w=content_w, collab_w=collab_w
            )

            if not results:
                st.warning(f"No results for **'{query}'**. Try a different title.")
            else:
                # Query movie details
                qd = get_movie_details(query, movies)
                q_tmdb = search_movie(qd['title']) if qd else {}

                if qd:
                    st.markdown(f"""
                    <div class="query-card">
                        <div style="font-size:2rem;">🎬</div>
                        <div>
                            <div class="query-label">Because you searched</div>
                            <div class="query-title">{qd['title']}</div>
                            <div class="query-genres">{qd['genres']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(
                    f'<div class="section-heading">🍿 Top {len(results)} Recommendations</div>',
                    unsafe_allow_html=True
                )

                # Fetch all TMDB data in batch
                with st.spinner("Fetching posters…"):
                    tmdb_data = fetch_batch([r['title'] for r in results])

                # Netflix-style grid — 4 columns
                COLS = 4
                for row_start in range(0, len(results), COLS):
                    row_results = results[row_start:row_start + COLS]
                    cols        = st.columns(COLS)
                    for col, rec, tmdb in zip(cols, row_results, tmdb_data[row_start:row_start+COLS]):
                        with col:
                            st.markdown(render_movie_card(rec, tmdb), unsafe_allow_html=True)

                st.markdown("---")
                with st.expander("📋 Raw recommendation data (for debugging)"):
                    import pandas as pd
                    df = pd.DataFrame(results)[['rank','title','genres','score','movieId']]
                    st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("""
        ## How CineMatch works

        CineMatch uses a **hybrid recommendation engine** that blends two complementary techniques:

        ### 1. Content-Based Filtering (TF-IDF + Cosine Similarity)
        - Each movie's genres and user tags are combined into a "soup" string
        - TF-IDF vectorization converts this text into a high-dimensional vector
        - Cosine similarity measures how "close" any two movie vectors are
        - Result: movies with similar genres/themes score high

        ### 2. Collaborative Filtering (SVD Matrix Factorization)
        - Uses the full ratings matrix (users × movies)
        - SVD decomposes it into latent factor vectors capturing hidden taste dimensions
        - If users who liked Movie A also liked Movie B, the model learns this pattern
        - Result: movies liked by similar-taste users score high

        ### 3. Hybrid Blending
        ```
        final_score = content_weight × content_score
                    + collab_weight  × collab_score
        ```
        Both scores are min-max normalised to [0,1] before blending.
        Use the sidebar slider to tune the balance for your use case.

        ### Why hybrid?
        - **Cold start problem**: a new movie with no ratings can still be recommended via content
        - **Genre blindness**: collaborative filtering finds non-obvious connections across genres
        - **Better accuracy**: blended approaches consistently outperform either alone
        """)

    with tabs[2]:
        st.markdown(f"""
        ## Dataset

        This app uses the **MovieLens dataset** by GroupLens Research.

        | File | Description |
        |------|-------------|
        | `movies.csv` | Movie titles and genres |
        | `ratings.csv` | User star ratings (0.5–5.0) |
        | `tags.csv` | Free-text user tags per movie |

        **Loaded:** `{len(movies):,}` movies

        **Download:** [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/latest/)

        ## TMDB API
        Movie posters and metadata are fetched from [The Movie Database (TMDB)](https://www.themoviedb.org/).
        Set your free API key as `TMDB_API_KEY` environment variable.

        ## Tech Stack
        | Component | Technology |
        |-----------|-----------|
        | UI | Streamlit |
        | Content model | scikit-learn TF-IDF + cosine similarity |
        | Collaborative | SciPy SVD (`svds`) |
        | Caching | pickle (disk) + `@st.cache_resource` (memory) |
        | Poster API | TMDB REST API |
        | Data | MovieLens (GroupLens) |
        """)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cinema-footer">
        CineMatch · Built with ❤ using Python, Streamlit & scikit-learn ·
        Data: MovieLens · Posters: TMDB
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

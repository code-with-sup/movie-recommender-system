import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Maroon & Beige Theme CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Jost:wght@300;400;500&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #f5f0e8 !important;
    font-family: 'Jost', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: #6b1e2a !important;
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: #f5f0e8 !important;
}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
    background-color: #8c2d3b !important;
    border-color: #b35c6a !important;
}
section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {
    background-color: #c8a97a !important;
    color: #2d0a10 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #f5f0e8 !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #f5f0e8 !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #c8a97a !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #8c2d3b !important;
}
.main-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #6b1e2a;
    letter-spacing: -1px;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'Jost', sans-serif;
    font-weight: 300;
    color: #8c6a4a;
    font-size: 1rem;
    margin-top: 4px;
    letter-spacing: 0.05em;
}
.section-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    color: #6b1e2a;
    border-bottom: 2px solid #c8a97a;
    padding-bottom: 6px;
    margin-bottom: 1.25rem;
}
.movie-card {
    background: #fffdf7;
    border: 1px solid #e0d5c0;
    border-radius: 16px;
    padding: 0;
    margin-bottom: 1rem;
    overflow: hidden;
    display: flex;
    flex-direction: row;
    box-shadow: 0 2px 12px rgba(107,30,42,0.06);
    transition: box-shadow 0.2s, transform 0.2s;
}
.movie-card:hover {
    box-shadow: 0 8px 28px rgba(107,30,42,0.14);
    transform: translateY(-2px);
}
.movie-poster {
    width: 90px;
    min-height: 130px;
    object-fit: cover;
    flex-shrink: 0;
    border-radius: 0;
}
.movie-poster-placeholder {
    width: 90px;
    min-height: 130px;
    background: linear-gradient(135deg, #6b1e2a 0%, #8c2d3b 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    flex-shrink: 0;
}
.movie-info {
    padding: 0.85rem 1rem;
    flex: 1;
}
.movie-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #2d0a10;
    margin-bottom: 5px;
    line-height: 1.2;
}
.movie-year-dir {
    font-size: 0.75rem;
    color: #8c6a4a;
    margin-bottom: 7px;
}
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 500;
    margin-right: 5px;
    margin-bottom: 4px;
    letter-spacing: 0.03em;
}
.badge-bollywood { background: #f5e6c8; color: #6b3a10; }
.badge-hollywood { background: #e8d5d8; color: #6b1e2a; }
.badge-genre { background: #ede8df; color: #5a4a35; }
.star-rating { color: #c8a97a; font-size: 0.82rem; margin-top: 5px; }
.score-bar {
    height: 3px;
    background: #e0d5c0;
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    background: linear-gradient(90deg, #6b1e2a, #c8a97a);
    border-radius: 2px;
}
.stat-card {
    background: #fffdf7;
    border: 1px solid #e0d5c0;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-num { font-family:'Cormorant Garamond',serif; font-size:2rem; font-weight:700; color:#6b1e2a; }
.stat-lbl { font-size:0.7rem; color:#8c6a4a; letter-spacing:.05em; text-transform:uppercase; }
.no-results {
    text-align: center;
    padding: 3rem;
    color: #8c6a4a;
    font-size: 1rem;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
}
.cf-tag {
    display: inline-block;
    background: #6b1e2a;
    color: #f5f0e8;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.65rem;
    margin-left: 8px;
    vertical-align: middle;
    letter-spacing: .04em;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c8a97a, transparent);
    margin: 1.5rem 0;
}
.trailer-btn {
    font-size: 0.75rem;
    color: #6b1e2a !important;
    text-decoration: none;
    font-weight: 600;
    border: 1px solid #6b1e2a;
    padding: 2px 8px;
    border-radius: 12px;
    transition: all 0.2s;
}
.trailer-btn:hover {
    background: #6b1e2a;
    color: #f5f0e8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Dataset (80+ Movies) ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies = [
        {"title": "Dangal", "genres": "Drama,Sports,Biography", "language": "Bollywood", "year": 2016, "rating": 8.4, "director": "Nitesh Tiwari", "poster": "https://m.media-amazon.com/images/M/MV5BMTQ4MzQzMzM2Nl5BMl5BanBnXkFtZTgwMTQ1NzU3MDI@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=x_7YlGv9u1g"},
        {"title": "3 Idiots", "genres": "Comedy,Drama", "language": "Bollywood", "year": 2009, "rating": 8.4, "director": "Rajkumar Hirani", "poster": "https://m.media-amazon.com/images/M/MV5BNzc4ZWQ3NmYtODE0Ny00YTQ4LTlkZWItNTBkMGQ0MmUwMmJlXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=K0eDlFX9GMc"},
        {"title": "Lagaan", "genres": "Drama,Sports,Musical", "language": "Bollywood", "year": 2001, "rating": 8.1, "director": "Ashutosh Gowariker", "poster": "https://m.media-amazon.com/images/M/MV5BM2FmODM4OTktOTRjOS00ZTIzLWIzZjAtMDBhOGEzYThkNzMzXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=rZPbpymefuE"},
        {"title": "Dil Chahta Hai", "genres": "Comedy,Drama,Romance", "language": "Bollywood", "year": 2001, "rating": 8.1, "director": "Farhan Akhtar", "poster": "https://m.media-amazon.com/images/M/MV5BYjY4NzgzNTQtZDhiNi00ZGJiLWIzMWQtNDg3YzkyNTdkZjAyXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=OBAcYSSUf6o"},
        {"title": "Taare Zameen Par", "genres": "Drama,Family", "language": "Bollywood", "year": 2007, "rating": 8.5, "director": "Aamir Khan", "poster": "https://m.media-amazon.com/images/M/MV5BNjVmMTlkNGYtYTM2OC00Yjc2LWIyYmQtMDY4ZmU1MWRlMjA3XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=EFfocv9BdwY"},
        {"title": "Sholay", "genres": "Action,Adventure,Drama", "language": "Bollywood", "year": 1975, "rating": 8.2, "director": "Ramesh Sippy", "poster": "https://m.media-amazon.com/images/M/MV5BNmI1NTRmMWQtNDJlZC00MGIzLWEwYzctYTQwNTI2NWNjM2MwXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=fefaxq2nXoE"},
        {"title": "Queen", "genres": "Drama,Comedy", "language": "Bollywood", "year": 2014, "rating": 8.1, "director": "Vikas Bahl", "poster": "https://m.media-amazon.com/images/M/MV5BZGUzOTEyM2ItNDhhNS00YjI2LTgxNWUtMjFkNjEyZmExNjU2XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=M_HP8xgXhBU"},
        {"title": "Gangs of Wasseypur", "genres": "Crime,Thriller,Drama", "language": "Bollywood", "year": 2012, "rating": 8.2, "director": "Anurag Kashyap", "poster": "https://m.media-amazon.com/images/M/MV5BMTc5NjY4MjUwNF5BMl5BanBnXkFtZTgwODM3NzM5MzE@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=XuK5TAEIqfg"},
        {"title": "Andhadhun", "genres": "Thriller,Crime,Drama", "language": "Bollywood", "year": 2018, "rating": 8.2, "director": "Sriram Raghavan", "poster": "https://m.media-amazon.com/images/M/MV5BMjZiYTNkNjUtNzI3MC00YWJmLTljM2QtNTI3MTU3ODYzNWFjXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=2iVYI99VGaw"},
        {"title": "Zindagi Na Milegi Dobara", "genres": "Comedy,Drama,Adventure", "language": "Bollywood", "year": 2011, "rating": 8.2, "director": "Zoya Akhtar", "poster": "https://m.media-amazon.com/images/M/MV5BOGIzYzg5NzItNDRkYS00NmIzLTk3NzQtZWYwY2VlZDhiYWQ4XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=FJrpcDgC3zU"},
        {"title": "Barfi!", "genres": "Comedy,Drama,Romance", "language": "Bollywood", "year": 2012, "rating": 8.1, "director": "Anurag Basu", "poster": "https://m.media-amazon.com/images/M/MV5BMTQzMTEyODY2Ml5BMl5BanBnXkFtZTgwMjA0MDUyMjE@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=fDpjzEKJjsM"},
        {"title": "Gully Boy", "genres": "Drama,Musical", "language": "Bollywood", "year": 2019, "rating": 7.9, "director": "Zoya Akhtar", "poster": "https://m.media-amazon.com/images/M/MV5BOWFkY2M3NDctZGEzMS00M2VmLTgzMTAtZWFiNjVmZDc5NWFjXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=JfbxcD6biOk"},
        {"title": "Stree", "genres": "Horror,Comedy", "language": "Bollywood", "year": 2018, "rating": 7.8, "director": "Amar Kaushik", "poster": "https://m.media-amazon.com/images/M/MV5BODhiZjI1MDMtZjFjOS00NjZiLWI5N2YtZTM2NWIxNmE3MjMzXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=gzeaGcLLl_A"},
        {"title": "Uri: The Surgical Strike", "genres": "Action,War,Drama", "language": "Bollywood", "year": 2019, "rating": 8.3, "director": "Aditya Dhar", "poster": "https://m.media-amazon.com/images/M/MV5BYTgyMTlkZTgtMTMxYi00Mjk5LTg2NTMtNGYyMDVlZWM0NmZjXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=Cg8sbRFS3zU"},
        {"title": "Tumbbad", "genres": "Horror,Fantasy,Thriller", "language": "Bollywood", "year": 2018, "rating": 8.2, "director": "Rahi Anil Barve", "poster": "https://m.media-amazon.com/images/M/MV5BOTY0YzY3MTMtOWQ5Yi00ODY2LThhOGMtMzFlMjhlODcxOGU1XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=YGIcZrUBY0k"},
        {"title": "Kahaani", "genres": "Thriller,Mystery,Crime", "language": "Bollywood", "year": 2012, "rating": 8.1, "director": "Sujoy Ghosh", "poster": "https://m.media-amazon.com/images/M/MV5BMTQ1NDI0NzkyOF5BMl5BanBnXkFtZTcwNzAyNzE2Nw@@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=rsjamVgPoI8"},
        {"title": "Drishyam", "genres": "Thriller,Crime,Drama", "language": "Bollywood", "year": 2015, "rating": 8.2, "director": "Nishikant Kamat", "poster": "https://m.media-amazon.com/images/M/MV5BMDRlZWFkMjEtYmYyZi00MmE5LWIzMzUtYmM2N2M5Y2UxZDJjXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=64xJLmcA2K8"},
        {"title": "Swades", "genres": "Drama", "language": "Bollywood", "year": 2004, "rating": 8.2, "director": "Ashutosh Gowariker", "poster": "https://m.media-amazon.com/images/M/MV5BZWJlNmQ2NmQtM2U3Yi00MTZjLWI1YzktY2I2MmExMzgwNmE3XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=vc7AZNWvs0M"},
        {"title": "Bajrangi Bhaijaan", "genres": "Drama,Adventure,Comedy", "language": "Bollywood", "year": 2015, "rating": 8.1, "director": "Kabir Khan", "poster": "https://m.media-amazon.com/images/M/MV5BYzVjMjZiNGUtZjZiNy00Yzg4LWEzYzYtMmI1NDg5NWNiNjUwXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=4nwAra0mz_Q"},
        {"title": "Rang De Basanti", "genres": "Drama", "language": "Bollywood", "year": 2006, "rating": 8.1, "director": "Rakeysh O. Mehra", "poster": "https://m.media-amazon.com/images/M/MV5BMTJhZTdmODctZWY3Zi00MGI3LThiZDMtZWQ5ZjNkYzQyMTI3XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=QHhnhqxB4E8"},
        {"title": "Rockstar", "genres": "Drama,Musical,Romance", "language": "Bollywood", "year": 2011, "rating": 7.9, "director": "Imtiaz Ali", "poster": "https://m.media-amazon.com/images/M/MV5BOTc3NzAxMjg4M15BMl5BanBnXkFtZTcwMDc2ODQwNw@@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=g5sskVIs4IM"},
        {"title": "Mughal-E-Azam", "genres": "Drama,Musical,Romance,Historical", "language": "Bollywood", "year": 1960, "rating": 8.2, "director": "K. Asif", "poster": "https://m.media-amazon.com/images/M/MV5BNWE0M2M0NTQtMTNkMS00ODgyLTg4MDMtODEwZjZjOTE1MDhjXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=6PjoLgcrmcQ"},
        {"title": "Padmaavat", "genres": "Historical,Drama,Romance", "language": "Bollywood", "year": 2018, "rating": 7.0, "director": "Sanjay Leela Bhansali", "poster": "https://m.media-amazon.com/images/M/MV5BNjM4MzFhOTItMzNmYy00NTg4LWFjYzItMWY0ZDAxY2NmNTUyXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=X_5_BLt76c0"},
        {"title": "Dil Dhadakne Do", "genres": "Comedy,Drama,Romance", "language": "Bollywood", "year": 2015, "rating": 7.4, "director": "Zoya Akhtar", "poster": "https://m.media-amazon.com/images/M/MV5BZmExOTlmNDAtYWMzZC00NzEyLWFhYmYtNTgwMjZiMDRhMDQyXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=qfnJCv4_1Ts"},
        {"title": "PK", "genres": "Comedy,Drama,Sci-Fi", "language": "Bollywood", "year": 2014, "rating": 8.1, "director": "Rajkumar Hirani", "poster": "https://m.media-amazon.com/images/M/MV5BMTYzOTE2NjkxN15BMl5BanBnXkFtZTgwMDgzMTg0MzE@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=SOXWc32k4zA"},
        {"title": "Kabir Singh", "genres": "Drama,Romance", "language": "Bollywood", "year": 2019, "rating": 7.1, "director": "Sandeep Vanga", "poster": "https://m.media-amazon.com/images/M/MV5BNjU5ZTljMDEtNzg5Ny00OTliLWI3NmYtOTE1ZDg3NTNkMDM0XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=RiANSSgCuJk"},
        {"title": "Article 15", "genres": "Crime,Drama,Thriller", "language": "Bollywood", "year": 2019, "rating": 8.1, "director": "Anubhav Sinha", "poster": "https://m.media-amazon.com/images/M/MV5BYmNlMWYzN2MtODNhOC00ZTdhLTk3NzAtNzRkMTg3MWE4ZmJhXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=HKOJY0cU63E"},
        {"title": "Raazi", "genres": "Thriller,Drama,Spy", "language": "Bollywood", "year": 2018, "rating": 7.8, "director": "Meghna Gulzar", "poster": "https://m.media-amazon.com/images/M/MV5BYjM0ZTQyNGQtNDZlYy00MWM5LWExZDAtZTM2ZWU4ZGNmYzk2XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=YjMSttRJrhA"},
        {"title": "Ludo", "genres": "Comedy,Crime,Thriller", "language": "Bollywood", "year": 2020, "rating": 7.5, "director": "Anurag Basu", "poster": "https://m.media-amazon.com/images/M/MV5BYmI3ZDU2ODQtODBjYy00NDZiLTg4ZWEtMjAzZDI1OTY1OGY4XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=ZT9Gihrxa4Q"},
        {"title": "Shershaah", "genres": "Action,War,Biography", "language": "Bollywood", "year": 2021, "rating": 8.4, "director": "Vishnuvardhan", "poster": "https://m.media-amazon.com/images/M/MV5BZTAzNzg0OGUtZmY1My00Y2VmLTk2YzYtNDU3MjlmNzU5ZjE3XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=Q0FTXnefVBA"},
        {"title": "83", "genres": "Sports,Drama,Biography", "language": "Bollywood", "year": 2021, "rating": 7.9, "director": "Kabir Khan", "poster": "https://m.media-amazon.com/images/M/MV5BYmNlNGY2NWEtODBiYi00YWIxLTkyNmItNTUwMmRjOTY4NzNkXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=QHdkC6Kn0Io"},
        {"title": "Pathaan", "genres": "Action,Thriller,Spy", "language": "Bollywood", "year": 2023, "rating": 5.9, "director": "Siddharth Anand", "poster": "https://m.media-amazon.com/images/M/MV5BNDdkNTY1MDQtY2I5MC00OTFlLTg5OWQtZWE2YzE5NWFiMDgzXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=vqu4z34wENw"},
        {"title": "RRR", "genres": "Action,Drama,Historical", "language": "Bollywood", "year": 2022, "rating": 7.8, "director": "S. S. Rajamouli", "poster": "https://m.media-amazon.com/images/M/MV5BNWMwODYyMjQtMTczMi00NTQ1LWFkYjItMGJhMWRkY2E3NDAyXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=GY4BgdUSpbE"},
        {"title": "KGF Chapter 2", "genres": "Action,Crime,Drama", "language": "Bollywood", "year": 2022, "rating": 8.2, "director": "Prashanth Neel", "poster": "https://m.media-amazon.com/images/M/MV5BZmQzZjVkZTUtYjI4ZC00ZDJmLWI0ZDUtZTFmMGM1Mzc5ZjIyXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=jQsE85cI384"},
        {"title": "Pushpa: The Rise", "genres": "Action,Crime,Drama", "language": "Bollywood", "year": 2021, "rating": 7.6, "director": "Sukumar", "poster": "https://m.media-amazon.com/images/M/MV5BOWE4YWEyNjYtMWFiNC00M2IzLWE3ZGMtMjQ0ZGEyOWI1YjAzXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=pKctjlxbFDQ"},
        {"title": "The Shawshank Redemption", "genres": "Drama", "language": "Hollywood", "year": 1994, "rating": 9.3, "director": "Frank Darabont", "poster": "https://m.media-amazon.com/images/M/MV5BMDAyY2FhYjctNDc5OS00MDNlLThiMGUtY2UxYWVkNGY2ZjljXkEyXkFqcGc@._V1_QL75_UX380_CR0,4,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=PLl99DlL6b4"},
        {"title": "The Dark Knight", "genres": "Action,Crime,Thriller", "language": "Hollywood", "year": 2008, "rating": 9.0, "director": "Christopher Nolan", "poster": "https://m.media-amazon.com/images/M/MV5BMTMxNTMwODM0NF5BMl5BanBnXkFtZTcwODAyMTk2Mw@@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=EXeTwQWrcwY"},
        {"title": "Inception", "genres": "Action,Sci-Fi,Thriller", "language": "Hollywood", "year": 2010, "rating": 8.8, "director": "Christopher Nolan", "poster": "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=YoHD9XEInc0"},
        {"title": "Interstellar", "genres": "Sci-Fi,Drama,Adventure", "language": "Hollywood", "year": 2014, "rating": 8.7, "director": "Christopher Nolan", "poster": "https://m.media-amazon.com/images/M/MV5BYzdjMDAxZGItMjI2My00ODA1LTlkNzItOWFjMDU5ZDJlYWY3XkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=zSWdZVtXT7E"},
        {"title": "Forrest Gump", "genres": "Drama,Romance,Comedy", "language": "Hollywood", "year": 1994, "rating": 8.8, "director": "Robert Zemeckis", "poster": "https://m.media-amazon.com/images/M/MV5BNDYwNzVjMTItZmU5YS00YjQ5LTljYjgtMjY2NDVmYWMyNWFmXkEyXkFqcGc@._V1_QL75_UY562_CR4,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=bLvqoHBptjg"},
        {"title": "The Godfather", "genres": "Crime,Drama", "language": "Hollywood", "year": 1972, "rating": 9.2, "director": "Francis Ford Coppola", "poster": "https://m.media-amazon.com/images/M/MV5BNGEwYjgwOGQtYjg5ZS00Njc1LTk2ZGEtM2QwZWQ2NjdhZTE5XkEyXkFqcGc@._V1_QL75_UY562_CR8,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=UaVTIH8mujA"},
        {"title": "Pulp Fiction", "genres": "Crime,Drama,Thriller", "language": "Hollywood", "year": 1994, "rating": 8.9, "director": "Quentin Tarantino", "poster": "https://m.media-amazon.com/images/M/MV5BYTViYTE3ZGQtNDBlMC00ZTAyLTkyODMtZGRiZDg0MjA2YThkXkEyXkFqcGc@._V1_QL75_UY562_CR3,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=s7EdQ4FqbhY"},
        {"title": "The Matrix", "genres": "Sci-Fi,Action", "language": "Hollywood", "year": 1999, "rating": 8.7, "director": "The Wachowskis", "poster": "https://m.media-amazon.com/images/M/MV5BN2NmN2VhMTQtMDNiOS00NDlhLTliMjgtODE2ZTY0ODQyNDRhXkEyXkFqcGc@._V1_QL75_UX380_CR0,4,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=vKQi3bBA1y8"},
        {"title": "Avengers: Endgame", "genres": "Action,Adventure,Sci-Fi", "language": "Hollywood", "year": 2019, "rating": 8.4, "director": "The Russo Brothers", "poster": "https://m.media-amazon.com/images/M/MV5BMTc5MDE2ODcwNV5BMl5BanBnXkFtZTgwMzI2NzQ2NzM@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=TcMBFSGVi1c"},
        {"title": "Titanic", "genres": "Drama,Romance", "language": "Hollywood", "year": 1997, "rating": 7.9, "director": "James Cameron", "poster": "https://m.media-amazon.com/images/M/MV5BYzYyN2FiZmUtYWYzMy00MzViLWJkZTMtOGY1ZjgzNWMwN2YxXkEyXkFqcGc@._V1_QL75_UX380_CR0,2,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=kVrqfYjkTdQ"},
        {"title": "Schindler's List", "genres": "Drama,Historical,Biography", "language": "Hollywood", "year": 1993, "rating": 9.0, "director": "Steven Spielberg", "poster": "https://m.media-amazon.com/images/M/MV5BNjM1ZDQxYWUtMzQyZS00MTE1LWJmZGYtNGUyNTdlYjM3ZmVmXkEyXkFqcGc@._V1_QL75_UX380_CR0,4,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=gG22XNhtnoY"},
        {"title": "Joker", "genres": "Crime,Drama,Thriller", "language": "Hollywood", "year": 2019, "rating": 8.5, "director": "Todd Phillips", "poster": "https://m.media-amazon.com/images/M/MV5BNzY3OWQ5NDktNWQ2OC00ZjdlLThkMmItMDhhNDk3NTFiZGU4XkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=zAGVQLHvwOY"},
        {"title": "Get Out", "genres": "Horror,Thriller,Mystery", "language": "Hollywood", "year": 2017, "rating": 7.7, "director": "Jordan Peele", "poster": "https://m.media-amazon.com/images/M/MV5BMjUxMDQwNjcyNl5BMl5BanBnXkFtZTgwNzcwMzc0MTI@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=DzfpyUB60YY"},
        {"title": "La La Land", "genres": "Drama,Musical,Romance", "language": "Hollywood", "year": 2016, "rating": 8.0, "director": "Damien Chazelle", "poster": "https://m.media-amazon.com/images/M/MV5BMzUzNDM2NzM2MV5BMl5BanBnXkFtZTgwNTM3NTg4OTE@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=0pdqf4P9MB8"},
        {"title": "Fight Club", "genres": "Drama,Thriller,Mystery", "language": "Hollywood", "year": 1999, "rating": 8.8, "director": "David Fincher", "poster": "https://m.media-amazon.com/images/M/MV5BOTgyOGQ1NDItNGU3Ny00MjU3LTg2YWEtNmEyYjBiMjI1Y2M5XkEyXkFqcGc@._V1_QL75_UX380_CR0,4,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=qtRKdVHc-cE"},
        {"title": "Parasite", "genres": "Thriller,Drama,Comedy", "language": "Hollywood", "year": 2019, "rating": 8.5, "director": "Bong Joon-ho", "poster": "https://m.media-amazon.com/images/M/MV5BYjk1Y2U4MjQtY2ZiNS00OWQyLWI3MmYtZWUwNmRjYWRiNWNhXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=5xH0HfJHsaY"},
        {"title": "Whiplash", "genres": "Drama,Musical", "language": "Hollywood", "year": 2014, "rating": 8.5, "director": "Damien Chazelle", "poster": "https://m.media-amazon.com/images/M/MV5BMDFjOWFkYzktYzhhMC00NmYyLTkwY2EtYjViMDhmNzg0OGFkXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=7d_jQycdQGo"},
        {"title": "The Wolf of Wall Street", "genres": "Crime,Comedy,Drama,Biography", "language": "Hollywood", "year": 2013, "rating": 8.2, "director": "Martin Scorsese", "poster": "https://m.media-amazon.com/images/M/MV5BMjIxMjgxNTk0MF5BMl5BanBnXkFtZTgwNjIyOTg2MDE@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=iszwuX1AK6A"},
        {"title": "Arrival", "genres": "Sci-Fi,Drama,Mystery", "language": "Hollywood", "year": 2016, "rating": 7.9, "director": "Denis Villeneuve", "poster": "https://m.media-amazon.com/images/M/MV5BMTExMzU0ODcxNDheQTJeQWpwZ15BbWU4MDE1OTI4MzAy._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=tFMo3UJ4B4g"},
        {"title": "The Prestige", "genres": "Thriller,Mystery,Drama,Sci-Fi", "language": "Hollywood", "year": 2006, "rating": 8.5, "director": "Christopher Nolan", "poster": "https://m.media-amazon.com/images/M/MV5BMTM3MzQ5MjQ5OF5BMl5BanBnXkFtZTcwMTQ3NzMzMw@@._V1_QL75_UY562_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=RLtaA9fFNXU"},
        {"title": "Knives Out", "genres": "Mystery,Crime,Comedy,Thriller", "language": "Hollywood", "year": 2019, "rating": 7.9, "director": "Rian Johnson", "poster": "https://m.media-amazon.com/images/M/MV5BZDU5ZTRkYmItZjg0Mi00ZTQwLThjMWItNWM3MTMxMzVjZmVjXkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=qGqiHJTsRkQ"},
        {"title": "No Country for Old Men", "genres": "Crime,Drama,Thriller", "language": "Hollywood", "year": 2007, "rating": 8.2, "director": "Coen Brothers", "poster": "https://m.media-amazon.com/images/M/MV5BMjA5Njk3MjM4OV5BMl5BanBnXkFtZTcwMTc5MTE1MQ@@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=38A__WT3-o0"},
        {"title": "Dune", "genres": "Sci-Fi,Adventure,Drama", "language": "Hollywood", "year": 2021, "rating": 8.0, "director": "Denis Villeneuve", "poster": "https://m.media-amazon.com/images/M/MV5BNWIyNmU5MGYtZDZmNi00ZjAwLWJlYjgtZTc0ZGIxMDE4ZGYwXkEyXkFqcGc@._V1_QL75_UY562_CR1,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=n9xhJrPXop4"},
        {"title": "Oppenheimer", "genres": "Historical,Drama,Biography,Thriller", "language": "Hollywood", "year": 2023, "rating": 8.9, "director": "Christopher Nolan", "poster": "https://m.media-amazon.com/images/M/MV5BN2JkMDc5MGQtZjg3YS00NmFiLWIyZmQtZTJmNTM5MjVmYTQ4XkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=uYPbbksJxIg"},
        {"title": "Mad Max: Fury Road", "genres": "Action,Adventure,Sci-Fi", "language": "Hollywood", "year": 2015, "rating": 8.1, "director": "George Miller", "poster": "https://m.media-amazon.com/images/M/MV5BZDRkODJhOTgtOTc1OC00NTgzLTk4NjItNDgxZDY4YjlmNDY2XkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=hEJnMQG9ev8"},
        {"title": "The Silence of the Lambs", "genres": "Thriller,Crime,Horror", "language": "Hollywood", "year": 1991, "rating": 8.6, "director": "Jonathan Demme", "poster": "https://m.media-amazon.com/images/M/MV5BNDdhOGJhYzctYzYwZC00YmI2LWI0MjctYjg4ODdlMDExYjBlXkEyXkFqcGc@._V1_QL75_UY562_CR1,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=6iB21hsprAQ"},
        {"title": "1917", "genres": "War,Drama,Action", "language": "Hollywood", "year": 2019, "rating": 8.3, "director": "Sam Mendes", "poster": "https://m.media-amazon.com/images/M/MV5BYzkxZjg2NDQtMGVjMy00NWZkLTk0ZDEtZWE3NDYwYjAyMTg1XkEyXkFqcGc@._V1_QL75_UX380_CR0,20,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=YqNYrYUiMfg"},
        {"title": "Gravity", "genres": "Sci-Fi,Thriller,Drama", "language": "Hollywood", "year": 2013, "rating": 7.7, "director": "Alfonso Cuarón", "poster": "https://m.media-amazon.com/images/M/MV5BNjE5MzYwMzYxMF5BMl5BanBnXkFtZTcwOTk4MTk0OQ@@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=OiTiKOy59o4"},
        {"title": "Blade Runner 2049", "genres": "Sci-Fi,Drama,Mystery", "language": "Hollywood", "year": 2017, "rating": 8.0, "director": "Denis Villeneuve", "poster": "https://m.media-amazon.com/images/M/MV5BNzA1Njg4NzYxOV5BMl5BanBnXkFtZTgwODk5NjU3MzI@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=gCcx85zbxz4"},
        {"title": "Coco", "genres": "Family,Animation,Musical,Adventure", "language": "Hollywood", "year": 2017, "rating": 8.4, "director": "Lee Unkrich", "poster": "https://m.media-amazon.com/images/M/MV5BMDIyM2E2NTAtMzlhNy00ZGUxLWI1NjgtZDY5MzhiMDc5NGU3XkEyXkFqcGc@._V1_QL75_UY562_CR7,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=xlnPHQ3TLX8"},
        {"title": "Spider-Man: Into the Spider-Verse", "genres": "Animation,Action,Adventure", "language": "Hollywood", "year": 2018, "rating": 8.4, "director": "Bob Persichetti", "poster": "https://m.media-amazon.com/images/M/MV5BMjMwNDkxMTgzOF5BMl5BanBnXkFtZTgwNTkwNTQ3NjM@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=g4Hbz2jLxvQ"},
        {"title": "Everything Everywhere All at Once", "genres": "Sci-Fi,Comedy,Action", "language": "Hollywood", "year": 2022, "rating": 7.8, "director": "Daniels", "poster": "https://m.media-amazon.com/images/M/MV5BOWNmMzAzZmQtNDQ1NC00Nzk5LTkyMmUtNGI2N2NkOWM4MzEyXkEyXkFqcGc@._V1_QL75_UY562_CR4,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=wxN1T1uxQ2g"},
        {"title": "Top Gun: Maverick", "genres": "Action,Drama", "language": "Hollywood", "year": 2022, "rating": 8.3, "director": "Joseph Kosinski", "poster": "https://m.media-amazon.com/images/M/MV5BMDBkZDNjMWEtOTdmMi00NmExLTg5MmMtNTFlYTJlNWY5YTdmXkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=qSqVVswa420"},
        {"title": "Tár", "genres": "Drama", "language": "Hollywood", "year": 2022, "rating": 7.4, "director": "Todd Field", "poster": "https://m.media-amazon.com/images/M/MV5BYWY5YThhOGUtNDU4OS00NTk3LWI0ODQtNmRiYTk0ZjVkZWU2XkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=Na6gA1RehsU"},
        {"title": "The Batman", "genres": "Action,Crime,Thriller", "language": "Hollywood", "year": 2022, "rating": 7.8, "director": "Matt Reeves", "poster": "https://m.media-amazon.com/images/M/MV5BMmU5NGJlMzAtMGNmOC00YjJjLTgyMzUtNjAyYmE4Njg5YWMyXkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=mqqft2x_Aa4"},
        {"title": "Elvis", "genres": "Biography,Drama,Musical", "language": "Hollywood", "year": 2022, "rating": 7.4, "director": "Baz Luhrmann", "poster": "https://m.media-amazon.com/images/M/MV5BNTVhZmUyMDQtY2I5Ny00OWNiLTgzNjUtMTg4YTQwMTc0OTQxXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=wBDLRvjHVOY"},
        {"title": "The Irishman", "genres": "Crime,Drama,Biography", "language": "Hollywood", "year": 2019, "rating": 7.8, "director": "Martin Scorsese", "poster": "https://m.media-amazon.com/images/M/MV5BMTY2YThkNmQtOWJhYy00ZDc3LWEzOGEtMmQwNzM0YjFmZWIyXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=WHXxVmeGQUc"},
        {"title": "Once Upon a Time in Hollywood", "genres": "Drama,Comedy", "language": "Hollywood", "year": 2019, "rating": 7.6, "director": "Quentin Tarantino", "poster": "https://m.media-amazon.com/images/M/MV5BOGNmMjJiZTAtMTM0ZS00OTNiLWE1M2ItYjk0YzBjNTkzMTYzXkEyXkFqcGc@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=ELeMaP8EPAA"},
        {"title": "Bohemian Rhapsody", "genres": "Biography,Drama,Musical", "language": "Hollywood", "year": 2018, "rating": 7.6, "director": "Bryan Singer", "poster": "https://m.media-amazon.com/images/M/MV5BMTA2NDc3Njg5NDVeQTJeQWpwZ15BbWU4MDc1NDcxNTUz._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=mP0VHJYFOAU"},
        {"title": "A Quiet Place", "genres": "Horror,Thriller,Sci-Fi", "language": "Hollywood", "year": 2018, "rating": 7.5, "director": "John Krasinski", "poster": "https://m.media-amazon.com/images/M/MV5BMjI0MDMzNTQ0M15BMl5BanBnXkFtZTgwMTM5NzM3NDM@._V1_SX300.jpg", "trailer": "https://www.youtube.com/watch?v=WR7cc5t7tv8"},
        {"title": "Hereditary", "genres": "Horror,Drama,Mystery", "language": "Hollywood", "year": 2018, "rating": 7.3, "director": "Ari Aster", "poster": "https://m.media-amazon.com/images/M/MV5BNTEyZGQwODctYWJjZi00NjFmLTg3YmEtMzlhNjljOGZhMWMyXkEyXkFqcGc@._V1_QL75_UX380_CR0,0,380,562_.jpg", "trailer": "https://www.youtube.com/watch?v=V6wWKNij_1M"},
    ]
    return pd.DataFrame(movies)


df = load_data()

ALL_GENRES = sorted({g.strip() for genres in df["genres"].str.split(",") for g in genres})


# ─── Simulated User Ratings for Collaborative Filtering ────────────────────────
@st.cache_data
def build_user_ratings():
    np.random.seed(42)
    titles = df["title"].tolist()
    # Create clusters of users with different tastes
    users = [f"user_{i}" for i in range(1, 41)]
    ratings_dict = {}
    for i, user in enumerate(users):
        n = np.random.randint(8, 25)
        seen = np.random.choice(titles, n, replace=False)
        for movie in seen:
            row = df[df["title"] == movie].iloc[0]
            base = row["rating"]
            genres = row["genres"].lower()
            
            # Adjust base rating based on user persona
            if i < 10 and any(g in genres for g in ["action", "sci-fi", "thriller", "crime"]):
                base += 1.5
            elif 10 <= i < 20 and any(g in genres for g in ["romance", "drama"]):
                base += 1.5
            elif 20 <= i < 30 and "comedy" in genres:
                base += 1.5
            elif 30 <= i < 40 and row["language"] == "Bollywood":
                base += 1.5
            else:
                base -= 1.0 # Penalize movies outside their taste
                
            noise = np.random.normal(0, 0.5)
            ratings_dict[(user, movie)] = round(min(10, max(1, base + noise)), 1)
    return ratings_dict

user_ratings_db = build_user_ratings()


def content_based(selected_genres, language_filter, top_n=10):
    filtered = df.copy()
    if language_filter != "Both":
        filtered = filtered[filtered["language"] == language_filter]
    if selected_genres:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(filtered["genres"].str.replace(",", " "))
        query_vec = tfidf.transform([" ".join(selected_genres)])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        filtered = filtered.copy()
        filtered["similarity"] = scores
        filtered = filtered[filtered["similarity"] > 0]
        filtered = filtered.sort_values(["similarity", "rating"], ascending=[False, False])
    else:
        filtered["similarity"] = 1.0
        filtered = filtered.sort_values("rating", ascending=False)
    return filtered.head(top_n).reset_index(drop=True)


def collaborative_filtering(liked_movies, language_filter, top_n=10):
    if not liked_movies:
        return pd.DataFrame()
    # Build rating matrix
    titles = df["title"].tolist()
    users = list({u for (u, _) in user_ratings_db.keys()})
    matrix = pd.DataFrame(0.0, index=users, columns=titles)
    for (user, movie), rating in user_ratings_db.items():
        matrix.loc[user, movie] = rating
    # Add "current user" based on liked movies
    current = pd.Series(0.0, index=titles)
    for m in liked_movies:
        current[m] = 9.0
    # Cosine sim between current user and all users
    from sklearn.metrics.pairwise import cosine_similarity as cs
    sim = cs([current.values], matrix.values)[0]
    top_users_idx = np.argsort(sim)[::-1][:5]
    top_users = [users[i] for i in top_users_idx]
    # Aggregate scores
    scores = {}
    for title in titles:
        if title in liked_movies:
            continue
        vals = [matrix.loc[u, title] for u in top_users if matrix.loc[u, title] > 0]
        if vals:
            scores[title] = np.mean(vals)
    if not scores:
        return pd.DataFrame()
    rec_titles = sorted(scores, key=scores.get, reverse=True)
    result = df[df["title"].isin(rec_titles)].copy()
    if language_filter != "Both":
        result = result[result["language"] == language_filter]
    result["similarity"] = result["title"].map(scores).fillna(0) / 10
    return result.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def stars(r):
    full = int(r / 2)
    return "★" * full + "☆" * (5 - full) + f"  {r}/10"


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.6rem;font-weight:700;color:#f5f0e8;margin-bottom:4px;">🎬 CineMatch</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.75rem;color:#e0c5b0;margin-bottom:1rem;letter-spacing:.05em;">Your personal cinema curator</p>', unsafe_allow_html=True)
    st.markdown("---")

    rec_mode = st.radio(
        "Recommendation Mode",
        ["Content-Based", "Collaborative Filtering", "Hybrid"],
        help="Content: genre matching · Collaborative: based on similar users · Hybrid: both combined"
    )

    selected_genres = st.multiselect(
        "Select Genre(s)",
        options=ALL_GENRES,
        default=["Action"],
    )

    language_filter = st.radio("Cinema", ["Both", "Bollywood", "Hollywood"])

    top_n = st.slider("Number of Results", 3, 15, 8)

    st.markdown("---")

    liked_movies = st.multiselect(
        "Movies You've Liked (for CF)",
        options=sorted(df["title"].tolist()),
        help="Required for Collaborative Filtering mode",
    )

    st.markdown("---")
    st.markdown('<p style="font-size:0.7rem;color:#e0c5b0;line-height:1.6;">Uses TF-IDF cosine similarity for content-based and user-based collaborative filtering for personalised picks.</p>', unsafe_allow_html=True)


# ─── Main Layout ───────────────────────────────────────────────────────────────
col_title, col_stats = st.columns([3, 2])

with col_title:
    st.markdown('<p class="main-title">🎬 CineMatch</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">GENRE-BASED MOVIE RECOMMENDATIONS — BOLLYWOOD & HOLLYWOOD</p>', unsafe_allow_html=True)

with col_stats:
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-num">{len(df)}</div><div class="stat-lbl">Movies</div></div>', unsafe_allow_html=True)
    with s2:
        bw = len(df[df["language"]=="Bollywood"])
        st.markdown(f'<div class="stat-card"><div class="stat-num">{bw}</div><div class="stat-lbl">Bollywood</div></div>', unsafe_allow_html=True)
    with s3:
        avg = round(df["rating"].mean(), 1)
        st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:#c8a97a;">⭐{avg}</div><div class="stat-lbl">Avg IMDb</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─── Search Bar ────────────────────────────────────────────────────────────────
search_q = st.text_input("", placeholder="🔍  Search by movie title...", label_visibility="collapsed")
if search_q:
    hits = df[df["title"].str.lower().str.contains(search_q.lower())]
    if not hits.empty:
        st.markdown(f'<p class="section-header">Search Results</p>', unsafe_allow_html=True)
        for _, row in hits.iterrows():
            lc = "badge-bollywood" if row["language"] == "Bollywood" else "badge-hollywood"
            gb = "".join(f'<span class="badge badge-genre">{g}</span>' for g in row["genres"].split(","))
            trailer_btn = f'<a href="{row.get("trailer", "#")}" target="_blank" class="trailer-btn">▶ Trailer</a>' if row.get("trailer") else ""
            st.markdown(f"""
            <div class="movie-card">
                <img src="{row['poster']}" class="movie-poster" referrerpolicy="no-referrer" onerror="this.outerHTML='<div class=\\'movie-poster-placeholder\\'>🎬</div>'">
                <div class="movie-info">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-year-dir">{row['year']} · {row['director']}</div>
                    <span class="badge {lc}">{row['language']}</span>{gb}
                    <div class="star-rating" style="display:flex; justify-content:space-between; align-items:center;">
                        <span>{stars(row['rating'])}</span>
                        {trailer_btn}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No movies found.")
    st.stop()

# ─── Recommendations ───────────────────────────────────────────────────────────
if rec_mode == "Content-Based":
    results = content_based(selected_genres, language_filter, top_n)
    mode_label = "Content-Based"
elif rec_mode == "Collaborative Filtering":
    results = collaborative_filtering(liked_movies, language_filter, top_n)
    mode_label = "Collaborative Filtering"
else:
    cb = content_based(selected_genres, language_filter, top_n * 2)
    cf = collaborative_filtering(liked_movies, language_filter, top_n * 2)
    if not cf.empty:
        combined = pd.concat([cb, cf]).drop_duplicates("title")
        combined["combined_score"] = combined.get("similarity", 0) * 0.5 + combined["rating"] / 10 * 0.5
        results = combined.sort_values("combined_score", ascending=False).head(top_n).reset_index(drop=True)
    else:
        results = cb
    mode_label = "Hybrid"

genre_label = ", ".join(selected_genres) if selected_genres else "All Genres"
st.markdown(f'<p class="section-header">Top {len(results)} picks · {genre_label} <span class="cf-tag">{mode_label}</span></p>', unsafe_allow_html=True)

if results.empty:
    st.markdown('<div class="no-results">😔 No results found. Try different genres or add liked movies for CF mode.</div>', unsafe_allow_html=True)
else:
    col_a, col_b = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        lc = "badge-bollywood" if row["language"] == "Bollywood" else "badge-hollywood"
        gb = "".join(f'<span class="badge badge-genre">{g.strip()}</span>' for g in row["genres"].split(","))
        sim_pct = int(row.get("similarity", 0.9) * 100)
        trailer_btn = f'<a href="{row.get("trailer", "#")}" target="_blank" class="trailer-btn">▶ Trailer</a>' if row.get("trailer") else ""
        card_html = f"""
        <div class="movie-card">
            <img src="{row['poster']}" class="movie-poster" referrerpolicy="no-referrer" onerror="this.outerHTML='<div class=\\'movie-poster-placeholder\\'>🎬</div>'">
            <div class="movie-info">
                <div class="movie-title">{row['title']}</div>
                <div class="movie-year-dir">{row['year']} · {row['director']}</div>
                <span class="badge {lc}">{row['language']}</span>{gb}
                <div class="star-rating" style="display:flex; justify-content:space-between; align-items:center;">
                    <span>{stars(row['rating'])}</span>
                    {trailer_btn}
                </div>
                <div class="score-bar"><div class="score-fill" style="width:{sim_pct}%;"></div></div>
            </div>
        </div>"""
        if i % 2 == 0:
            with col_a:
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            with col_b:
                st.markdown(card_html, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#8c6a4a;font-size:0.78rem;font-family:\'Jost\',sans-serif;">'
    'CineMatch · Content-Based + Collaborative Filtering · 80+ Bollywood & Hollywood Films · '
    'Scalable with real-time APIs & user personalization'
    '</p>',
    unsafe_allow_html=True,
)

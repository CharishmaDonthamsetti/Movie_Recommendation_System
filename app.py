import streamlit as st
import pickle
import pandas as pd
import os
import re

MODEL_FILE = "model.pkl"
MOVIES_CSV = "movies.csv"
RATINGS_CSV = "ratings.csv"

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

def get_year_from_title(title):
    m = re.search(r'\((\d{4})\)', str(title))
    return m.group(1) if m else "N/A"

def get_movie_metadata(movieId, movies_df, ratings_df=None):
    row = movies_df[movies_df["movieId"] == movieId]
    if row.empty:
        return {"title": str(movieId), "genres": "N/A", "year": "N/A", "avg_rating": "N/A", "rating_count": 0}
    title = row.iloc[0]["title"]
    genres = row.iloc[0].get("genres", "N/A")
    year = get_year_from_title(title)
    avg_rating = "N/A"
    rating_count = 0
    if ratings_df is not None:
        r = ratings_df[ratings_df["movieId"] == movieId]["rating"]
        if not r.empty:
            avg_rating = round(r.mean(), 2)
            rating_count = int(r.count())
    return {"title": title, "genres": genres, "year": year, "avg_rating": avg_rating, "rating_count": rating_count}

def reverse_lookup_movie_id(idx, movie_to_idx):
    for k, v in movie_to_idx.items():
        if v == idx:
            return k
    return None

def get_recommendations(selected_movie, model, movie_user_matrix_sparse, movies_df, movie_to_idx, top_n=6):
    sel = movies_df[movies_df["title"] == selected_movie]
    if sel.empty:
        return []
    movie_id = sel.iloc[0]["movieId"]
    movie_idx = movie_to_idx.get(movie_id, None)
    if movie_idx is None:
        return []
    distances, indices = model.kneighbors(movie_user_matrix_sparse[movie_idx], n_neighbors=top_n+1)
    recs = []
    flat_idx = indices.flatten()
    flat_dist = distances.flatten()
    for i in range(1, len(flat_idx)):
        idx = int(flat_idx[i])
        sim = float(flat_dist[i])
        similar_movie_id = reverse_lookup_movie_id(idx, movie_to_idx)
        if similar_movie_id is None:
            continue
        recs.append({"movieId": similar_movie_id, "similarity": round((1 - sim) * 100, 2)})
    return recs

def similarity_badge(similarity: float) -> str:

    if similarity >= 75:
        color = "#8e44ad"   # lavender
    elif similarity >= 50:
        color = "#9b59b6"   # lighter lavender
    else:
        color = "#c0392b"   # red
    return f"""
        <span style="
            background:{color};
            color:white;
            padding:6px 14px;
            border-radius:14px;
            font-size:13px;
            font-weight:bold;
            box-shadow:0 3px 8px rgba(0,0,0,0.6);
        ">
            {similarity}%
        </span>
    """


if not os.path.exists(MODEL_FILE):
    st.error(f"Missing '{MODEL_FILE}'. Put model.pkl in this folder.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    try:
        model, movie_user_matrix_sparse, movies, movie_to_idx = pickle.load(f)
    except Exception as e:
        st.error("Failed to load model.pkl: " + str(e))
        st.stop()

ratings_df = None
if os.path.exists(RATINGS_CSV):
    try:
        ratings_df = pd.read_csv(RATINGS_CSV)
    except Exception as e:
        st.warning("Could not read ratings.csv — avg rating/popularity disabled. " + str(e))


st.markdown(
    """
    <style>
    /* Body & Title */
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #121212;
        color: #e0e0e0;
    }
    .stApp > header {
        background: linear-gradient(90deg, #6a0dad, #9b59b6);
        color: white;
        font-size: 32px;
        font-weight: bold;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.6);
    }

    /* Movie card */
    .movie-card {
        padding: 18px;
        margin: 14px 0;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e1e1e, #2c2c2c);
        box-shadow: 0 6px 14px rgba(0,0,0,0.6);
        transition: transform 0.3s, box-shadow 0.3s;
        border-left: 5px solid #8e44ad;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.8);
    }
    .movie-title {
        font-size: 20px;
        font-weight: bold;
        color: #8e44ad;
    }
    .movie-meta {
        color: #b0b0b0;
        font-size: 14px;
    }

    /* Similarity badge */
    span {
        font-size: 13px !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2b1f3f;
        color: white;
        padding: 20px;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white !important;
        font-size: 20px !important;
    }

    /* Buttons */
    [data-testid="stSidebar"] button {
        background: linear-gradient(90deg, #8e44ad, #9b59b6) !important;
        color: #fff !important;
        font-size: 16px !important;
        padding: 10px 25px !important;
        border-radius: 12px !important;
        font-weight: bold;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    [data-testid="stSidebar"] button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.6);
    }

    /* Slider (adjustment bar) */
    .stSlider > div > div > div:nth-child(2) {
        color: white !important;
    }

    /* Input box */
    .stTextInput > div > input {
        background-color: #2c2c2c !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
        border-radius: 8px;
        padding: 6px;
        font-size: 15px;
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #8e44ad !important;
        font-weight: bold;
    }

    /* Expander */
    div[role="button"] > .streamlit-expanderHeader {
        background-color: #1e1e1e !important;
        color: #8e44ad !important;
        border-radius: 8px;
        padding: 10px;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; color:#9b59b6;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.caption("Offline recommendations with a modern dark lavender interface ✨")


st.sidebar.header("🏠 Home")
st.sidebar.subheader("Search Movies")
search_input = st.sidebar.text_input("Search a movie:")
filtered_movies = movies[movies['title'].str.contains(search_input, case=False, na=False)]
selected_movie = st.sidebar.selectbox("Select a movie:", filtered_movies["title"].values if not filtered_movies.empty else ["No movies found"])
st.sidebar.markdown("### Select number of recommendations")
top_n = st.sidebar.slider("", 4, 20, 8)


col1, col2, col3 = st.columns(3)
col1.metric("🎞️ Total movies", len(movies))
col2.metric("👥 Users", movie_user_matrix_sparse.shape[1])
col3.metric("⭐ Ratings stored", movie_user_matrix_sparse.nnz)


if st.sidebar.button("Show Recommendations 🚀"):
    with st.spinner("Finding the best matches..."):
        recs = get_recommendations(selected_movie, model, movie_user_matrix_sparse, movies, movie_to_idx, top_n=top_n)
        if not recs:
            st.warning("No recommendations found. Try another movie.")
        else:
            st.subheader(f"Because you liked **{selected_movie}**, you may also enjoy:")
            for r in recs:
                meta = get_movie_metadata(r["movieId"], movies, ratings_df)
                sim_badge = similarity_badge(r['similarity'])
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-title">🎥 {meta['title']} ({meta['year']})</div>
                        <div class="movie-meta">🎭 {meta['genres']}</div>
                        <div class="movie-meta">⭐ {meta['avg_rating']} from {meta['rating_count']} votes</div>
                        <div class="movie-meta">🔗 Similarity: {sim_badge}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


with st.expander("📊 Insights (offline)"):
    try:
        all_genres = movies["genres"].dropna().str.split("|").explode()
        top_genres = all_genres.value_counts().head(10)
        st.bar_chart(top_genres)
    except Exception as e:
        st.write("Could not compute genres: " + str(e))

    if ratings_df is not None:
        top_popular = ratings_df.groupby("movieId").agg({"rating": ["mean", "count"]})
        top_popular.columns = ["avg_rating", "rating_count"]
        top_popular = top_popular.reset_index().merge(movies, on="movieId", how="left")
        top_popular = top_popular.sort_values(["rating_count", "avg_rating"], ascending=False).head(10)
        st.write("Top popular movies (by rating count):")
        st.table(top_popular[["title", "rating_count", "avg_rating"]])
    else:
        st.write("ratings.csv not present — place it in the folder to enable popularity stats.")


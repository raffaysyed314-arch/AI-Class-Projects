import streamlit as st
import pandas as pd
import joblib

# Load data
movies = pd.read_csv("movie_recommender/data/movies.csv")
ratings = pd.read_csv("movie_recommender/data/ratings.csv")
model = joblib.load("movie_recommender/svd_model.pkl")

st.title("Movie Recommendation Engine")
st.write("Select a user and get personalized movie recommendations.")

user_ids = sorted(ratings["userId"].unique())
user_id = st.selectbox("Choose User ID", user_ids)

if st.button("Show Movies"):
    # Watched movies
    watched_ids = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    watched_titles = movies[movies["movieId"].isin(watched_ids)]["title"].tolist()
    
    st.subheader("Movies Already Watched")
    for title in watched_titles[:10]:  # show first 10 for clarity
        st.write(f"- {title}")
    if len(watched_titles) > 10:
        st.write(f"...and {len(watched_titles)-10} more")

    # Recommendations
    all_ids = movies["movieId"].tolist()
    unseen_ids = [m for m in all_ids if m not in watched_ids]

    preds = [(mid, model.predict(user_id, mid).est) for mid in unseen_ids]
    preds.sort(key=lambda x: x[1], reverse=True)

    st.subheader("Top 5 Recommended Movies")
    for movie_id, score in preds[:5]:
        title = movies[movies["movieId"] == movie_id]["title"].values[0]
        st.write(f"{title} | Predicted rating: {round(score,2)}")

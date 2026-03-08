# Movie Recommendation Engine

This project is a simple Movie Recommendation Engine built with Python and Streamlit.

## Features
- Shows movies already watched by a user
- Provides top 5 recommended movies using SVD collaborative filtering
- Simple and interactive frontend with Streamlit

## How to Run
1. Install Python 3.10+ and required packages:
   pip install scikit-surprise streamlit pandas joblib

2. Run the app:
   streamlit run app.py

3. Select a User ID to see:
   - Movies already watched
   - Top 5 recommended movies

## Notes
- The `svd_model.pkl` file contains the trained model
- `data` folder contains `movies.csv` and `ratings.csv`
- No internet connection required; works offline

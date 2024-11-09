import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Load the dataset
#sample_df = pd.read_csv('C:/Users/PC/Documents/Moringa/phase_5/Spotify-Reccomender-system/Data/dataset.csv')
sample_df = pd.read_csv('Data/dataset.csv')

# Sampling for demonstration
sample_df = sample_df.sample(n=10000, random_state=1)

# Define numerical features for scaling
numerical_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Scale numerical features
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(sample_df[numerical_features])

# Cosine similarity matrix
cosine_sim = cosine_similarity(numerical_scaled)
sim_df = pd.DataFrame(cosine_sim, index=sample_df['track_name'], columns=sample_df['track_name'])

# Load CSS for custom styling
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load HTML template
def load_html(file_name):
    with open(file_name, 'r') as f:
        html_content = f.read()
    return html_content

# Apply CSS styling
load_css("static/style.css")

# Display the HTML content
st.markdown(load_html("templates/index.html"), unsafe_allow_html=True)

# Recommendation function
def recommend(track_name, num_recommendations=5):
    if track_name not in sim_df.columns:
        st.write(f"Track '{track_name}' not found in the dataset.")
        return []
    similar_tracks = sim_df[track_name].sort_values(ascending=False)[1:num_recommendations + 1]
    return similar_tracks.index.tolist()

# Streamlit form for input and recommendations
st.title("Spotify Recommender System")
st.write("Enter up to 5 tracks to get song recommendations.")

with st.form("recommendation_form"):
    track1 = st.text_input("Track 1")
    track2 = st.text_input("Track 2")
    track3 = st.text_input("Track 3")
    track4 = st.text_input("Track 4")
    track5 = st.text_input("Track 5")
    num_recommendations = st.number_input("Number of recommendations per track", min_value=1, max_value=10, value=5)
    submit_button = st.form_submit_button("Get Recommendations")

if submit_button:
    input_tracks = [track1, track2, track3, track4, track5]
    recommendations = {}

    # Generate recommendations for each input track
    for track in input_tracks:
        if track:  # Ensure track input is not empty
            recs = recommend(track, num_recommendations=num_recommendations)
            recommendations[track] = recs if recs else ["No recommendations found"]

    # Display recommendations
    for track, recs in recommendations.items():
        st.write(f"Recommendations for **{track}**:")
        for idx, rec in enumerate(recs):
            rec_genre = sample_df[sample_df['track_name'] == rec]['track_genre'].values[0]
            st.write(f"{idx + 1}. {rec} (Genre: {rec_genre})")

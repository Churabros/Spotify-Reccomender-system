from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your dataset and preprocess as needed
sample_df = pd.read_csv('C:/Users/PC/Documents/Moringa/phase_5/Spotify-Reccomender-system/Data/dataset.csv')  # Adjust the filename

# Sampling for demonstration
sample_df = sample_df.sample(n=10000, random_state=1)  

# Preprocess the data: scale and encode features
numerical_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Example: preprocessing steps (scaling)
scaler = StandardScaler()
numerical_scaled = scaler.fit_transform(sample_df[numerical_features])

# Cosine similarity matrix
cosine_sim = cosine_similarity(numerical_scaled)

# Create a DataFrame for similarity
sim_df = pd.DataFrame(cosine_sim, index=sample_df['track_name'], columns=sample_df['track_name'])

# Recommendation function with debugging statements
def recommend(track_name, num_recommendations=5):
    if track_name not in sim_df.columns:
        print(f"Track '{track_name}' not found in the dataset.")
        return []
    similar_tracks = sim_df[track_name].sort_values(ascending=False)[1:num_recommendations + 1]
    print(f"Recommendations for '{track_name}': {similar_tracks.index.tolist()}")
    return similar_tracks.index.tolist()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Collect input from the form
        input_tracks = [
            request.form.get('track1'),
            request.form.get('track2'),
            request.form.get('track3'),
            request.form.get('track4'),
            request.form.get('track5')
        ]

        # Generate recommendations
        recommendations = {}
        for track in input_tracks:
            recs = recommend(track, num_recommendations=5)
            recommendations[track] = recs if recs else ["No recommendations found"]

        # Debug output to console
        print("Input tracks:", input_tracks)
        print("Generated recommendations:", recommendations)
        
        return render_template('index.html', recommendations=recommendations, input_tracks=input_tracks)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# Spotify Recommender System

## Problem Statement
In a competitive digital music streaming market, delivering personalized, relevant, and engaging music recommendations has become essential for enhancing user satisfaction and fostering brand loyalty. With a vast amount of music available, users rely heavily on effective recommendation systems to discover new songs and artists that align with their tastes. The challenge is to build a recommendation model that accurately captures user preferences by analyzing track features and context, enabling the platform to provide tailored music suggestions. Success in solving this problem will drive higher engagement and retention rates, positioning the platform as a preferred choice for music discovery.

## Business Understanding
In today’s digital landscape, the consumption of music has evolved dramatically, with users relying heavily on personalized recommendations to discover new artists and tracks. The proliferation of streaming services has transformed how listeners engage with music, making the ability to recommend songs effectively a key differentiator for platforms. A well-designed music recommendation model can significantly enhance user experience, fostering brand loyalty and increasing engagement. By analyzing user preferences, listening habits, and contextual data, our model aims to provide tailored suggestions that resonate with individual tastes. This not only satisfies the diverse musical interests of users but also drives higher retention rates, ultimately contributing to the platform's success in a competitive market.

## Objective
The goal of this project is to develop a robust recommendation system using the Spotify dataset. By leveraging data-driven techniques, we aim to generate song recommendations that closely match user preferences based on track features and other contextual information.

## Data Overview
The project uses a dataset containing various attributes related to songs on Spotify, such as:
- **Track Features**: Danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, etc.
- **Track Metadata**: Artists, album names, track names, popularity, and genre.

## Methodology
1. **Data Preprocessing**: Cleaning and transforming the data to prepare it for model training.
2. **Exploratory Data Analysis (EDA)**: Understanding the characteristics of the data and identifying patterns.
3. **Modeling**: Developing recommendation models using techniques like collaborative filtering, content-based filtering, and hybrid approaches.
4. **Evaluation**: Assessing model performance using appropriate metrics and refining the approach for better accuracy.

## Technologies Used
- **Python**: For data processing and modeling.
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib, and keras.
- **Jupyter Notebook**: For interactive code and analysis.

## Results
### Model Comparison

1. **Cosine Similarity Model**
   - **Evaluation Metrics**:
     - **Average Precision**: 0.83 — indicating that the model’s top recommendations align with the genre of the input track quite well.
     - **Average Recall**: 0.06 — showing that it could recommend more relevant tracks if expanded.
     - **Average MRR (Mean Reciprocal Rank)**: 0.90 — high rank consistency for genre relevance.
   - **Performance**: This model shows high precision but has lower recall, meaning it's particularly good at finding genre-specific recommendations but might miss out on broader similarities due to the restricted genre-based approach. The high MRR indicates that relevant recommendations often appear at the top of the list.

2. **Deep Learning Model (Autoencoder)**
   - **Evaluation Metrics**:
     - **Reconstruction Loss**:
       - Final Training Loss: 0.0032
       - Final Validation Loss: 0.0045 — low reconstruction error, suggesting that the autoencoder effectively captures underlying patterns in the track features.
     - **Average Similarity of Recommendations**: The recommendations have a high degree of similarity to input tracks, aligning with their numerical features but showing some variability in genre compared to the cosine similarity model.
   - **Performance**: The autoencoder produces well-embedded tracks and offers diversity in recommendations by leveraging learned features beyond genre. The low reconstruction loss supports that the model has learned distinct musical characteristics effectively.


## Future Work
- Implementing more advanced recommendation algorithms.
- Fine-tuning model parameters to enhance recommendation quality.
- Exploring additional features or contextual data to improve personalization.

## Getting Started
To run the project locally:
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the `Index.ipynb` notebook and follow the steps to preprocess the data, build models, and generate recommendations.

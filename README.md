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

# Final Evaluation


This evaluation compares two recommendation models developed for this project: a deep learning autoencoder-based model and a Transformer-based model. Both models aim to recommend tracks that closely match the genre and characteristics of the input track, enhancing the user experience by providing relevant and interesting song recommendations.

#### 1. Deep Learning Autoencoder Model

The autoencoder model leverages dense layers to generate compact embeddings for each track based on musical features. This model’s effectiveness was evaluated using reconstruction loss and average similarity as key metrics:

- **Final Training Loss**: 0.0009
- **Final Validation Loss**: 0.0012
- **Average Similarity of Recommendations**: 0.9969

These metrics indicate that the model successfully learns to reconstruct track features, with low training and validation losses suggesting a good fit without overfitting. The high average similarity (0.9969) reflects that recommendations are closely aligned with the characteristics of the input track, indicating high relevance for the user.

#### 2. Transformer-Based Model

The Transformer model uses a simplified architecture to process the combined features of each track, generating embeddings that capture genre and characteristic relationships. Evaluated with precision and Mean Reciprocal Rank (MRR) metrics:

- **Average Precision**: 0.76
- **Average MRR**: 0.90

With a precision score of 76%, the Transformer model performs slightly lower in genre alignment compared to other models. However, it maintains a high MRR score, indicating that relevant recommendations frequently appear early in the list. This approach offers users both relevance and diversity, introducing new music within similar genres.

#### Summary

Both models provide effective recommendations, though they emphasize different aspects of relevance and diversity. The **Deep Learning Autoencoder Model** excels in feature reconstruction and similarity, making it highly relevant for closely matching input tracks. Conversely, the **Transformer-Based Model** offers competitive ranking with some diversity in its recommendations. Based on overall performance, the Deep Learning Autoencoder Model may offer the best balance of relevance for users seeking recommendations highly aligned with their initial inputs feature embeddings.


## Future Work
- Implementing more advanced recommendation algorithms.
- Fine-tuning model parameters to enhance recommendation quality.
- Exploring additional features or contextual data to improve personalization.

## Getting Started
To run the project locally:
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the `Index.ipynb` notebook and follow the steps to preprocess the data, build models, and generate recommendations.

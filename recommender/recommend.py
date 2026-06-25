# src/recommender/recommend.py

import numpy as np


def recommend_songs_by_category(predicted_emotion, music_df, n=5):
    """
    Simple version: returns random songs matching the predicted emotion category
    """
    matches = music_df[music_df['emotion'].astype(str).str.lower() == str(predicted_emotion).lower()]
    if len(matches) == 0:
        print(f"No songs found for emotion: {predicted_emotion}")
        return None

    n = min(n, len(matches))
    return matches.sample(n=n)[['track_name', 'track_artist', 'emotion']]


def recommend_songs_by_distance(predicted_valence, predicted_arousal, music_df, n=5):
    """
    Better version: finds songs closest in valence-arousal space
    using Euclidean distance
    """
    music_df = music_df.copy()
    music_df['distance'] = np.sqrt(
        (music_df['valence'] - predicted_valence) ** 2 +
        (music_df['energy'] - predicted_arousal) ** 2
    )
    return music_df.nsmallest(n, 'distance')[['track_name', 'track_artist', 'emotion', 'distance']]


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/processed/cleaned_spotify_data.csv')

    print("Test 1 — Category-based recommendation:")
    print(recommend_songs_by_category("Happy", df, n=5))

    print("\nTest 2 — Distance-based recommendation:")
    print(recommend_songs_by_distance(0.8, 0.7, df, n=5))
    
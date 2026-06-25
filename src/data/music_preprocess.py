# src/preprocessing/preprocess_music.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ─────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────
def load_dataset(path):
    df = pd.read_csv(path)
    print(f"✅ Loaded dataset: {df.shape[0]} songs, {df.shape[1]} columns")
    return df

# ─────────────────────────────────────────
# STEP 2: DROP UNNECESSARY COLUMNS
# ─────────────────────────────────────────
def drop_unnecessary_columns(df):
    # These columns are not useful for recommendation
    cols_to_drop = [
        'track_href',
        'uri',
        'analysis_url',
        'track_album_id',
        'playlist_id',
        'id',
        'type'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"✅ Dropped unnecessary columns. Remaining: {df.shape[1]} columns")
    return df

# ─────────────────────────────────────────
# STEP 3: HANDLE MISSING VALUES
# ─────────────────────────────────────────
def handle_missing_values(df):
    # Fill missing track_album_name with Unknown
    df['track_album_name'] = df['track_album_name'].fillna('Unknown')
    print(f"✅ Missing values handled")
    print(df.isnull().sum().sum(), "missing values remaining")
    return df

# ─────────────────────────────────────────
# STEP 4: ADD EMOTION LABEL
# based on valence and energy
# ─────────────────────────────────────────
def add_emotion_label(df):
    """
    Map songs to emotions using:
    Valence = positivity of song (0 to 1)
    Energy  = intensity of song  (0 to 1)

    Emotion Map:
    High Valence + High Energy = Happy
    Low  Valence + High Energy = Stressed/Angry
    High Valence + Low  Energy = Relaxed/Calm
    Low  Valence + Low  Energy = Sad
    """
    def get_emotion(row):
        if row['valence'] >= 0.5 and row['energy'] >= 0.5:
            return 'happy'
        elif row['valence'] < 0.5 and row['energy'] >= 0.5:
            return 'stressed'
        elif row['valence'] >= 0.5 and row['energy'] < 0.5:
            return 'relaxed'
        else:
            return 'sad'

    df['emotion'] = df.apply(get_emotion, axis=1)
    print(f"✅ Emotion labels added")
    print(df['emotion'].value_counts())
    return df

# ─────────────────────────────────────────
# STEP 5: NORMALIZE AUDIO FEATURES
# ─────────────────────────────────────────
def normalize_features(df):
    # These are the audio features we use
    audio_features = [
        'energy',
        'tempo',
        'danceability',
        'loudness',
        'liveness',
        'valence',
        'speechiness',
        'instrumentalness',
        'acousticness'
    ]

    scaler = MinMaxScaler()
    df[audio_features] = scaler.fit_transform(df[audio_features])
    print(f"✅ Features normalized between 0 and 1")
    return df

# ─────────────────────────────────────────
# STEP 6: REMOVE DUPLICATES
# ─────────────────────────────────────────
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=['track_name', 'track_artist'])
    after = len(df)
    print(f"✅ Removed {before - after} duplicate songs")
    return df

# ─────────────────────────────────────────
# STEP 7: SAVE CLEANED DATASET
# ─────────────────────────────────────────
def save_cleaned_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")

# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Paths
    INPUT_PATH  = "D:/Projects/eeg-project/raw_data/deap/high_popularity_spotify_data.csv"
    OUTPUT_PATH = "D:/Projects/eeg-project/data/processed/cleaned_spotify_data.csv"

    print("\n🎵 Starting Music Dataset Cleaning Pipeline...\n")

    # Run pipeline step by step
    df = load_dataset(INPUT_PATH)
    df = drop_unnecessary_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = add_emotion_label(df)
    df = normalize_features(df)
    save_cleaned_data(df, OUTPUT_PATH)

    print("\n✅ Music Dataset Cleaning Complete!")
    print(f"Final dataset shape: {df.shape}")
    print(f"\nEmotion distribution:")
    print(df['emotion'].value_counts())
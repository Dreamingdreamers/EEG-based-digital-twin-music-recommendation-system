# src/recommender/pipeline_test.py

import joblib
import pandas as pd
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender.emotion_mapper import emotion_to_valence_arousal
from recommender.recommend import recommend_songs_by_distance
from data.preprocess import load_all_subjects_cached

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
MUSIC_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_spotify_data.csv')


def load_model(model_path=MODEL_PATH, le_path=ENCODER_PATH):
    """
    Mirrors load_model() in bagging.py exactly
    """
    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le


def predict_emotion(model, le, features):
    """
    Mirrors predict_emotion() in bagging.py exactly
    """
    features = np.array(features).reshape(1, -1)
    pred = model.predict(features)
    proba = model.predict_proba(features)[0]
    emotion = le.inverse_transform(pred)[0]
    confidence = round(float(proba.max()) * 100, 1)
    return emotion, confidence


def run_full_pipeline(eeg_sample):
    # Load trained model + label encoder
    model, le = load_model()

    # Load music database
    music_df = pd.read_csv(MUSIC_PATH)

    # Predict emotion from EEG sample
    predicted_emotion, confidence = predict_emotion(model, le, eeg_sample)
    print(f"EEG Predicted Emotion: {predicted_emotion} ({confidence}% confidence)")

    # Convert emotion label to approximate valence/arousal
    valence, arousal = emotion_to_valence_arousal(predicted_emotion)
    print(f"Mapped to Valence={valence}, Arousal={arousal}")

    # Get song recommendations using distance matching
    recommendations = recommend_songs_by_distance(valence, arousal, music_df, n=5)

    print("\nRecommended Songs:")
    print(recommendations)

    return predicted_emotion, confidence, recommendations


if __name__ == "__main__":
    print("=" * 55)
    print("  ThetaPlay — Full Pipeline Test")
    print("=" * 55)

    # Use REAL EEG data (same cache bagging.py uses)
    X_raw, y_raw = load_all_subjects_cached()

    # Test on a few real samples
    test_indices = [0, 100, 5000, 10000]

    for idx in test_indices:
        print(f"\n{'─'*45}")
        print(f"Sample index: {idx} | True label: {y_raw[idx]}")
        run_full_pipeline(X_raw[idx])
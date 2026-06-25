# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.database import log_prediction, log_recommendations
from recommender.emotion_mapper import emotion_to_valence_arousal
from recommender.recommend import recommend_songs_by_distance
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ThetaPlay EEG Music Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
MUSIC_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_spotify_data.csv')

# Load once at startup, not per-request
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)
music_df = pd.read_csv(MUSIC_PATH)


class EEGInput(BaseModel):
    features: list[float]  # 355 values expected


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ThetaPlay API is running"}


@app.post("/predict-emotion")
def predict_emotion_endpoint(eeg_input: EEGInput):
    features = np.array(eeg_input.features).reshape(1, -1)
    pred = model.predict(features)
    proba = model.predict_proba(features)[0]
    emotion = le.inverse_transform(pred)[0]
    confidence = round(float(proba.max()) * 100, 1)

    log_prediction(emotion, confidence)

    return {
        "predicted_emotion": emotion,
        "confidence": confidence
    }


@app.post("/recommend")
def recommend_endpoint(eeg_input: EEGInput):
    features = np.array(eeg_input.features).reshape(1, -1)
    pred = model.predict(features)
    emotion = le.inverse_transform(pred)[0]

    valence, arousal = emotion_to_valence_arousal(emotion)
    recommendations = recommend_songs_by_distance(valence, arousal, music_df, n=5)

    recs_list = recommendations.to_dict(orient='records')
    prediction_id = log_prediction(emotion, 0.0)
    log_recommendations(prediction_id, recs_list)

    return {
        "predicted_emotion": emotion,
        "recommendations": recs_list
    }
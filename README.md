# EEG-based-digital-twin-music-recommendation-system 
# ThetaPlay — EEG-Based Digital Twin Music Recommendation System

ThetaPlay predicts a listener's emotional state from EEG brain signals and recommends music that matches (or shifts) that emotion in real time. The system combines signal processing, machine learning, and a content-based recommendation engine into a single end-to-end pipeline, exposed through a REST API and a simple web frontend.

## Overview

Traditional music recommenders rely on listening history or explicit ratings. ThetaPlay instead uses **EEG signals** as a direct, physiological signal of the listener's current emotional state — classifying it into one of four categories (Happy, Calm, Sad, Stressed) using the valence-arousal model of emotion, then matching that state against a database of emotion-tagged songs.

```
EEG Signal → Feature Extraction → Bagging Classifier → Emotion Label
                                                              ↓
                                                   Valence/Arousal Mapping
                                                              ↓
                                              Music Recommendation Engine
                                                              ↓
                                                    Ranked Song List
```

## Dataset

- **DEAP Dataset** — 32 subjects, 40 one-minute music video trials each, 32-channel EEG recorded at 128 Hz, with self-reported valence/arousal/dominance/liking ratings per trial.
- **Spotify Audio Features Dataset** — used to tag a large song catalog with valence and energy values, mapped onto the same four emotion classes as the EEG model.

## Methodology

### 1. Preprocessing
- Each 60-second EEG trial is split into 8-second windows with 50% overlap, producing a `(32, 1024)` array per window (32 channels × 1024 timepoints).
- Emotion labels are derived from each subject's own median valence/arousal split (rather than a fixed threshold), accounting for individual rating bias.

### 2. Feature Extraction
For each EEG window, 355 features are extracted:
- **Band power** (theta, alpha, beta, gamma) per channel — 4 × 32 = 128 features
- **Differential entropy** per band per channel — 4 × 32 = 128 features
- **Statistical features** (mean, variance, std) per channel — 3 × 32 = 96 features
- **Frontal alpha asymmetry** across 3 left/right electrode pairs — 3 features

### 3. Class Balancing
The raw dataset is moderately imbalanced across the four emotion classes. **SMOTE** (Synthetic Minority Oversampling) is applied to balance all classes before training.

### 4. Model Training
A **Bagging Classifier** (ensemble of Decision Trees) is trained on the balanced, scaled feature set, with:
- 10-fold stratified cross-validation
- Train/test split evaluation
- Per-class and macro/weighted **AUC-ROC** analysis

### 5. Music Emotion Mapping
Songs from the Spotify dataset are tagged into the same four emotion classes using their `valence` and `energy` audio features, mirroring the EEG labeling logic for consistency across both data sources.

### 6. Recommendation Engine
Two matching strategies are implemented:
- **Category-based** — returns songs sharing the same emotion label
- **Distance-based** — ranks songs by Euclidean distance in valence-arousal space to the predicted emotion, giving more precise, ranked recommendations

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  EEG Sample  │ ──▶ │   FastAPI    │ ──▶ │  Bagging Model  │
└─────────────┘     │   Backend    │     │ (trained, saved) │
                     └──────┬───────┘     └─────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Recommendation     │
                  │ Engine (distance-  │
                  │ based matching)    │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  SQLite Database   │
                  │ (logs predictions  │
                  │  & recommendations)│
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │  Web Frontend       │
                  │ (displays emotion + │
                  │  song results)      │
                  └────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| Signal Processing | NumPy, SciPy (Welch's method for PSD) |
| Class Balancing | imbalanced-learn (SMOTE) |
| Machine Learning | scikit-learn (BaggingClassifier, DecisionTreeClassifier) |
| Model Persistence | joblib |
| Backend API | FastAPI, Uvicorn |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript (Fetch API) |
| Data Source | DEAP Dataset, Spotify Audio Features |

## Project Structure

```
eeg-project/
├── api/
│   └── main.py                  # FastAPI app — endpoints & CORS config
├── data/
│   ├── raw/                     # Raw DEAP + Spotify data (gitignored)
│   ├── processed/               # Cleaned, emotion-tagged music data
│   └── thetaplay.db             # SQLite database (predictions, recommendations)
├── frontend/
│   └── index.html               # Simple web UI calling the API
├── models/
│   ├── emotion_model.pkl        # Trained Bagging classifier
│   └── label_encoder.pkl        # Label encoder for emotion classes
├── notebooks/                   # Exploratory data analysis notebooks
├── src/
│   ├── data/
│   │   ├── preprocess.py        # EEG windowing, feature extraction, caching
│   │   └── music_preprocess.py  # Spotify data cleaning & emotion tagging
│   ├── bagging.py               # Model training, evaluation, AUC-ROC analysis
│   ├── recommender/
│   │   ├── emotion_mapper.py    # Valence/arousal ↔ emotion label mapping
│   │   ├── recommend.py         # Category & distance-based recommendation logic
│   │   └── pipeline_test.py     # End-to-end pipeline test script
│   └── utils/
│       ├── config.py
│       └── database.py          # SQLite schema, logging, history retrieval
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~10-20 minutes)
```bash
python src/bagging.py
```
This loads the cached EEG features, applies SMOTE, trains the Bagging classifier, evaluates it, and saves `models/emotion_model.pkl` and `models/label_encoder.pkl`.

### 3. Preprocess the music dataset
```bash
python src/data/music_preprocess.py
```

### 4. Initialize the database
```bash
python src/utils/database.py
```

### 5. Start the API server
```bash
uvicorn api.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`

### 6. Open the frontend
Open `frontend/index.html` in a browser (with the API server running).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict-emotion` | Takes a 355-value EEG feature array, returns predicted emotion + confidence |
| POST | `/recommend` | Takes a 355-value EEG feature array, returns predicted emotion + 5 ranked song recommendations |

## Results

- 10-fold cross-validated accuracy and per-class AUC-ROC scores are reported by `src/bagging.py` on each run (see console output for exact figures on your trained model).
- The recommendation engine was validated end-to-end: real EEG samples produce emotion predictions that map to musically coherent song recommendations (e.g., a "Sad" prediction surfaces lower-valence, lower-energy tracks).

## Future Work

- **Digital Twin profile**: extend per-user prediction history (already logged in SQLite) into a queryable emotional profile endpoint, enabling personalization over multiple sessions rather than single-shot predictions.
- **Deep learning models**: the raw (unwindowed) EEG signal cache is already preserved separately for future CNN/LSTM experimentation, which may learn representations beyond hand-crafted band-power features.
- **Real-time EEG streaming** from consumer EEG headsets instead of pre-recorded DEAP samples.
- **Mirror vs. regulation recommendation strategies**: currently the system mirrors the detected emotion; a mood-regulation mode (recommending calming music during high-stress states) is a natural extension.

## Dataset Citation

Koelstra, S., et al. "DEAP: A Database for Emotion Analysis Using Physiological Signals." *IEEE Transactions on Affective Computing*, 2012.


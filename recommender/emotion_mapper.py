# src/recommender/emotion_mapper.py

def get_music_emotion_label(valence, energy, v_threshold=0.5, a_threshold=0.5):
    """
    Maps Spotify valence/energy values to emotion labels,
    matching the same 4 classes used by the EEG model:
    Happy, Calm, Sad, Stressed
    """
    v = 1 if valence > v_threshold else 0
    a = 1 if energy > a_threshold else 0

    emotion_map = {
        (1, 1): "Happy",
        (1, 0): "Relaxed",
        (0, 0): "Sad",
        (0, 1): "Stressed"
    }
    return emotion_map[(v, a)]


def tag_dataframe_with_emotion(df, valence_col='valence', energy_col='energy'):
    """
    Applies emotion labeling to every row in a music dataframe
    """
    df = df.copy()
    df['emotion'] = df.apply(
        lambda row: get_music_emotion_label(row[valence_col], row[energy_col]),
        axis=1
    )
    return df


# Add this to emotion_mapper.py

EMOTION_TO_COORDS = {
    "happy":    (0.75, 0.75),   # high valence, high arousal
    "relaxed":  (0.75, 0.25),   # high valence, low arousal
    "sad":      (0.25, 0.25),   # low valence, low arousal
    "stressed": (0.25, 0.75),   # low valence, high arousal
}

def emotion_to_valence_arousal(emotion_label):
    """
    Converts a predicted emotion category back into 
    approximate valence/arousal coordinates for distance matching
    """
    emotion_label = str(emotion_label).lower()
    return EMOTION_TO_COORDS.get(emotion_label, (0.5, 0.5))  # default = neutral
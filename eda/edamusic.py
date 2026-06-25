import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("D:/Projects/eeg-project/data/processed/cleaned_spotify_data.csv")

audio_features = [
    'energy', 'tempo', 'danceability', 'loudness',
    'liveness', 'valence', 'speechiness',
    'instrumentalness', 'acousticness'
]

# Create output directory for plots
os.makedirs("D:/Projects/eeg-project/plots", exist_ok=True)

# 1. Emotion distribution — are classes balanced?
df['emotion'].value_counts().plot(kind='bar', color=['#7F77DD','#1D9E75','#D85A30','#BA7517'])
plt.title("Emotion distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of songs")
plt.tight_layout()
plt.savefig("D:/Projects/eeg-project/plots/01_emotion_distribution.png", dpi=100)
print("✅ Saved: plots/01_emotion_distribution.png")
plt.close()

# 2. Correlation heatmap — which features are related?
plt.figure(figsize=(10, 8))
sns.heatmap(df[audio_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Audio feature correlations")
plt.tight_layout()
plt.savefig("D:/Projects/eeg-project/plots/02_correlations.png", dpi=100)
print("✅ Saved: plots/02_correlations.png")
plt.close()

# 3. Feature histograms — check normalization worked
df[audio_features].hist(bins=20, figsize=(14, 8))
plt.suptitle("Audio feature distributions (should all be 0–1)")
plt.tight_layout()
plt.savefig("D:/Projects/eeg-project/plots/03_feature_distributions.png", dpi=100)
print("✅ Saved: plots/03_feature_distributions.png")
plt.close()

# 4. Per-emotion averages — validate your emotion labels
df.groupby('emotion')[audio_features].mean().T.plot(kind='bar', figsize=(12, 5))
plt.title("Average audio features per emotion")
plt.ylabel("Mean value (0–1)")
plt.legend(title='Emotion')
plt.tight_layout()
plt.savefig("D:/Projects/eeg-project/plots/04_emotion_averages.png", dpi=100)
print("✅ Saved: plots/04_emotion_averages.png")
plt.close()

print("\n✅ EDA Complete! Plots saved to plots/ directory")
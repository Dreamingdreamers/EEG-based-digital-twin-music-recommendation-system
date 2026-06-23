import pickle
import numpy as np
from pathlib import Path

# Prefer repository-relative data file; fall back to cwd-relative location
default_path = Path(__file__).parent / 'data' / 'deap-dataset' / 'data_preprocessed_python' / 's01.dat'
cwd_path = Path.cwd() / 'data' / 'deap-dataset' / 'data_preprocessed_python' / 's01.dat'
d_drive_path = Path('D:/Projects/eeg-project/data/deap-dataset/data_preprocessed_python/s01.dat')

if default_path.exists():
    filepath = default_path
elif cwd_path.exists():
    filepath = cwd_path
elif d_drive_path.exists():
    filepath = d_drive_path
else:-
    raise FileNotFoundError(
        f"s01.dat not found. Looked in: {default_path}, {cwd_path}, and {d_drive_path}\n"
        "Please place the dataset under the project `data/deap-dataset/data_preprocessed_python/` folder or update the path in this script."
    )

with open(filepath, 'rb') as f:
    subject = pickle.load(f, encoding='latin1')

labels = subject['labels']  # (40, 4)

valence = labels[:, 0]
arousal = labels[:, 1]

print("Valence stats:")
print(f"  Min: {valence.min():.2f}")
print(f"  Max: {valence.max():.2f}")
print(f"  Mean: {valence.mean():.2f}")

print("\nArousal stats:")
print(f"  Min: {arousal.min():.2f}")
print(f"  Max: {arousal.max():.2f}")
print(f"  Mean: {arousal.mean():.2f}")

print("\nFirst 10 valence scores:", valence[:10])
print("First 10 arousal scores:", arousal[:10])
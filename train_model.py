import pandas as pd
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# === Load dataset ===
df = pd.read_excel("my_datasets.xlsx", sheet_name=0)

# === Define expected feature columns ===
feature_order = [
    "ENGLISH", "MATHEMATICS", "SOCIAL STUDIES", "AGRIC SCIENCE", "PHE",
    "BASIC TECH", "COMPUTER", "BUSINESS STUDIES", "IRS/CRS", "CCA", "YORUBA"
]

# === Filter and clean relevant features ===
available_subjects = [subj for subj in feature_order if subj in df.columns]
df[available_subjects] = df[available_subjects].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Apply rules to filter only meaningful students ===
def passed_all(subjects, row):
    return all(row.get(subj, 0) >= 60 for subj in subjects)

science = ['MATHEMATICS', 'BASIC TECH', 'COMPUTER', 'AGRIC SCIENCE', 'PHE']
commercial = ['MATHEMATICS', 'BUSINESS STUDIES', 'COMPUTER', 'SOCIAL STUDIES', 'ENGLISH']
arts = ['ENGLISH', 'YORUBA', 'SOCIAL STUDIES', 'CCA', 'IRS/CRS']

def is_valid(row):
    return (
        passed_all(science, row) or
        passed_all(commercial, row) or
        passed_all(arts, row)
    )

df = df[df.apply(is_valid, axis=1)]  # Remove "General Studies"

# === Standardize features ===
X = df[available_subjects].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train KMeans model ===
k = 3  # Now only 3 clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(X_scaled)

# === Save models and config ===
os.makedirs("models", exist_ok=True)

with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

career_config = {
    "feature_order": available_subjects,
    "cluster_labels": {
        0: "Arts",
        1: "Science",
        2: "Commercial"
    }
}

with open("models/career_config.pkl", "wb") as f:
    pickle.dump(career_config, f)

print(f"\n✅ Model trained with only Science, Commercial, and Arts clusters.")
sil_score = silhouette_score(X_scaled, model.labels_)
db_score = davies_bouldin_score(X_scaled, model.labels_)
inertia = model.inertia_

print("\n📊 Clustering Evaluation Metrics:")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")
print(f"Inertia: {inertia:.4f}")


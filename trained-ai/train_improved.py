import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import joblib

# --- CONFIGURATION ---
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. LOAD SEMANTIC EMBEDDER
# 'all-MiniLM-L6-v2' is fast and excellent for title similarity
print("Loading Semantic Encoder (MiniLM)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- PART 1: LOAD REAL DATASET ---
def load_real_titles():
    print("Loading NSF Award Search titles...")
    try:
        # Assuming your CSV is in the same directory
        df = pd.read_csv("NSF_Award_Search_cleaned.csv")
        return df['AwardTitle'].dropna().unique().tolist()
    except Exception as e:
        print(f"Error loading CSV: {e}. Using fallback examples.")
        return ["Advanced Research Project", "Sustainable Engineering Solution", "Deep Learning for Healthcare"]

REAL_GOOD_TITLES = load_real_titles()

# --- PART 2: GENERATE DATASET ---
def generate_data(n=3000):
    titles = []
    stats = [] # [duration, mooe, ps, co, total, agencies]
    labels = []
    
    bad_keywords = ["Purchase", "Procurement", "Acquisition", "Buying", "Shopping"]
    bad_items = ["Laptops", "Computers", "Office Supplies", "Furniture", "Vehicles"]

    for _ in range(n):
        is_good = np.random.rand() > 0.35 
        
        if is_good:
            t = np.random.choice(REAL_GOOD_TITLES)
        else:
            keyword = np.random.choice(bad_keywords)
            item = np.random.choice(bad_items)
            t = f"{keyword} of {item} for Department Office"

        # Generate Stats
        if is_good:
            duration = np.random.randint(12, 36)
            agencies = np.random.randint(1, 6) 
            ps_pct, mooe_pct = np.random.uniform(0.2, 0.4), np.random.uniform(0.3, 0.5)
        else:
            duration = np.random.randint(1, 9)
            agencies = np.random.randint(0, 2)
            ps_pct, mooe_pct = np.random.uniform(0.7, 0.9), np.random.uniform(0.01, 0.05)

        co_pct = 1.0 - (ps_pct + mooe_pct)
        base_budget = np.random.randint(500000, 5000000)
        ps, mooe, co = int(base_budget * ps_pct), int(base_budget * mooe_pct), int(base_budget * co_pct)
        total = ps + mooe + co

        titles.append(t)
        stats.append([duration, mooe, ps, co, total, agencies])
        labels.append(1 if is_good else 0)

    return np.array(titles), np.array(stats), np.array(labels)

titles, meta_data, labels = generate_data(n=max(3000, len(REAL_GOOD_TITLES)))

# --- PART 3: SEMANTIC VECTORIZATION (THE FIX) ---
print(f"Encoding {len(titles)} titles into semantic vectors...")
# This turns "Health Behavior" and "Math Functions" into very different vectors
title_embeddings = embedder.encode(titles, show_progress_bar=True)

# --- PART 4: PREPROCESSING & CLUSTERING ---
scaler = StandardScaler()
meta_data_scaled = scaler.fit_transform(meta_data)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print("Training Profiler (K-Means)...")
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(meta_data_scaled)
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_profiler.pkl"))

# --- PART 5: PREDICTIVE AI (NEURAL NETWORK) ---
print("Building Semantic Neural Network...")

X_train_emb, X_val_emb, X_train_meta, X_val_meta, y_train, y_val = train_test_split(
    title_embeddings, meta_data_scaled, labels, test_size=0.2, random_state=42
)

# Input layers
emb_input = keras.Input(shape=(384,), name='emb_input') # MiniLM vectors are 384-dim
meta_input = keras.Input(shape=(6,), name='meta_input')

# Text Processing Branch
x1 = layers.Dense(128, activation='relu')(emb_input)
x1 = layers.BatchNormalization()(x1)
x1 = layers.Dropout(0.2)(x1)

# Stats Processing Branch
x2 = layers.Dense(64, activation='relu')(meta_input)
x2 = layers.Dropout(0.2)(x2)

# Combined Reasoning
combined = layers.concatenate([x1, x2])
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dense(32, activation='relu')(z)
output = layers.Dense(1, activation='sigmoid', name='output')(z)

model = keras.Model(inputs=[emb_input, meta_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit(
    [X_train_emb, X_train_meta], y_train,
    validation_data=([X_val_emb, X_val_meta], y_val),
    epochs=12,
    batch_size=32,
    verbose=1
)

# --- PART 6: SAVE EVERYTHING ---
model.save(os.path.join(MODEL_DIR, "proposal_model.keras"))

# Save the Vector Database for the search/similarity functionality
# We store the titles and their corresponding semantic embeddings
vector_db = {
    "titles": titles.tolist(),
    "vectors": title_embeddings.tolist()
}
with open(os.path.join(MODEL_DIR, "vector_db.json"), 'w') as f:
    json.dump(vector_db, f)

# --- PART 7: EXPORT CLUSTER DESCRIPTIONS ---
print("Generating profiling descriptions...")
cluster_descriptions = {}
centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers):
    # center = [duration, mooe, ps, co, total, agencies]
    dur, mooe, ps, co, total, agencies = center
    
    desc = "Standard R&D Project"
    
    if ps > (total * 0.6): 
        desc = "High-Salary / Overhead Heavy" 
    elif agencies < 1: 
        desc = "Independent / Solo Research" 
    elif total > 3000000 and agencies > 2: 
        desc = "Large-Scale Collaborative Grant"
    elif dur < 10: 
        desc = "Short-Term / Quick Study"
    
    cluster_descriptions[str(i)] = desc
    print(f"   Cluster {i}: {desc}")

with open(os.path.join(MODEL_DIR, "cluster_descriptions.json"), 'w') as f:
    json.dump(cluster_descriptions, f)

print("\n" + "="*60)
print("RETRAINING COMPLETE")
print("The '99% Match' error should now be resolved by Semantic Embeddings.")
print("="*60)
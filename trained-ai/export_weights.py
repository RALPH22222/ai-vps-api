"""
Export trained Keras model weights to JSON for TypeScript inference.
Updated for Semantic Embeddings (SentenceTransformer).
"""

import os
import json
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

MODEL_DIR = "models"
OUTPUT_DIR = os.path.join("..", "src", "services", "ai-models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model and metadata...")
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "proposal_model.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_profiler.pkl"))

with open(os.path.join(MODEL_DIR, "cluster_descriptions.json"), "r") as f:
    cluster_descs = json.load(f)

# --- 1. Export Dense layer weights ---
print("Exporting dense layer weights...")
dense_layers = []
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        weights = layer.get_weights()  # [kernel, bias]
        dense_layers.append({
            "name": layer.name,
            "kernel": weights[0].tolist(),
            "bias": weights[1].tolist(),
            "activation": layer.get_config().get("activation", "linear"),
        })
        print(f"   Dense '{layer.name}': kernel={weights[0].shape}")

with open(os.path.join(OUTPUT_DIR, "dense_layers.json"), "w") as f:
    json.dump(dense_layers, f)

# --- 1b. Export BatchNormalization weights (for TypeScript inference) ---
print("Exporting batch normalization weights...")
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        weights = layer.get_weights()  # [gamma, beta, moving_mean, moving_variance]
        bn_data = {
            "gamma": weights[0].tolist(),
            "beta": weights[1].tolist(),
            "moving_mean": weights[2].tolist(),
            "moving_variance": weights[3].tolist(),
            "epsilon": layer.epsilon,
        }
        with open(os.path.join(OUTPUT_DIR, "batch_norm.json"), "w") as f:
            json.dump(bn_data, f)
        print(f"   BatchNorm '{layer.name}': size={len(weights[0])}")
        break  # Only one BN layer in this model

# --- 2. Export Scaler parameters ---
print("Exporting scaler...")
scaler_data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
}
with open(os.path.join(OUTPUT_DIR, "scaler.json"), "w") as f:
    json.dump(scaler_data, f)

# --- 3. Export KMeans centroids ---
print("Exporting kmeans...")
kmeans_data = {
    "centroids": kmeans.cluster_centers_.tolist(),
    "descriptions": cluster_descs,
} 
with open(os.path.join(OUTPUT_DIR, "kmeans.json"), "w") as f:
    json.dump(kmeans_data, f)

# --- 4. Export comparison database (Semantic) ---
print("Exporting comparison database from vector_db.json...")
vdb_path = os.path.join(MODEL_DIR, "vector_db.json")
if os.path.exists(vdb_path):
    with open(vdb_path, 'r') as f:
        vdb = json.load(f)
    
    comparison_db = {
        "titles": vdb['titles'],
        "vectors": vdb['vectors'],
    }
    with open(os.path.join(OUTPUT_DIR, "comparison_db.json"), "w") as f:
        json.dump(comparison_db, f)
    print(f"   Comparison DB: {len(vdb['titles'])} titles exported.")
else:
    print("Warning: vector_db.json not found. Novelty checks may be empty.")

print(f"\nAll files exported to {os.path.abspath(OUTPUT_DIR)}/")
print("Note: Vocabulary and Embedding JSONs were intentionally omitted as the model now uses SentenceTransformers.")

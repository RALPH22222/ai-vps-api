import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

MODEL_DIR = "models"
print("="*50)
print("AI SYSTEM QUICK TEST")
print("="*50)

try:
    print("Loading AI models...")
    model = keras.models.load_model(os.path.join(MODEL_DIR, "proposal_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_profiler.pkl"))

    with open(os.path.join(MODEL_DIR, "cluster_descriptions.json"), 'r') as f:
        cluster_descs = json.load(f)

    print("✅ Models loaded successfully!")

    # Test prediction
    print("\nTesting prediction...")
    test_title = "Development of AI System for Data Analysis"
    test_stats = scaler.transform([[24, 50000, 30000, 20000, 100000, 3]])  # 24 months, budget components, 3 agencies

    score = model.predict({
        'text_input': tf.constant([test_title], dtype=tf.string),
        'meta_input': test_stats
    }, verbose=0)[0][0] * 100

    print(f"✅ Prediction successful!")
    print(f"   Test Title: '{test_title}'")
    print(f"   Score: {score:.1f}%")

    # Test clustering
    cluster_id = kmeans.predict(test_stats)[0]
    profile = cluster_descs.get(str(cluster_id), "Unknown")
    print(f"   Profile: {profile}")

    print("\n🎉 AI system is working correctly!")
    print("Ready for proposal analysis!")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Check if all model files exist in the models/ directory")
import sys
import json
import os
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    title = sys.argv[1]
    duration = float(sys.argv[2])
    mooe = float(sys.argv[3])
    ps = float(sys.argv[4])
    co = float(sys.argv[5])
    total = float(sys.argv[6])
    agencies = float(sys.argv[7])

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

    # Load models
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "proposal_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    # Process
    title_vec = embedder.encode([title])
    stats = scaler.transform([[duration, mooe, ps, co, total, agencies]])

    # Predict
    score = model.predict({'emb_input': title_vec, 'meta_input': stats}, verbose=0)[0][0] * 100

    print(json.dumps({"score": int(score)}))
except Exception as e:
    print(json.dumps({"error": str(e), "score": 65}))

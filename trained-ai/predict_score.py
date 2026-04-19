import sys
import json
import os
import warnings

# Suppress ALL noise before heavy imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU mode (stops CUDA errors)
warnings.filterwarnings("ignore")

try:
    # Heavy imports inside try to catch any missing dependency issues
    import tensorflow as tf
    from sentence_transformers import SentenceTransformer
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError as e:
    print(json.dumps({"error": f"Missing dependency: {str(e)}"}))
    sys.exit(1)

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
    # Do not send a fake score — let the API use its heuristic fallback when `score` is absent.
    print(json.dumps({"error": str(e)}))

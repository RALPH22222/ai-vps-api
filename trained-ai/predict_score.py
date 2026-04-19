import sys
import json
import os
import warnings

# Suppress ALL noise before heavy imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU mode (stops CUDA errors)
warnings.filterwarnings("ignore")

# Redirect stderr to devnull to stop the "Unexpected" and "Warning" noise on VPS
sys.stderr = open(os.devnull, 'w')

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
    raw_pred = model.predict({'emb_input': title_vec, 'meta_input': stats}, verbose=0)[0][0]
    score = round(float(raw_pred) * 100)
    
    # Cap between 0 and 100
    final_score = max(0, min(100, score))

    # Detailed report for the Node.js logs
    print(json.dumps({
        "score": final_score,
        "metrics": {
            "title_vec_mean": float(title_vec.mean()),
            "meta_stats_sum": float(stats.sum()),
            "raw_output": float(raw_pred)
        }
    }))
except Exception as e:
    # Do not send a fake score — let the API use its heuristic fallback when `score` is absent.
    print(json.dumps({"error": str(e)}))

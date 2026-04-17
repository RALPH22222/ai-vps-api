import os
import re
import json
import numpy as np
import tensorflow as tf
import joblib
from pypdf import PdfReader
import docx
import docx2txt
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_DIR = "models"
DOCUMENT_FILE = "VAWC_CapsuleProposal-updated.pdf"  # Now supports .pdf, .docx, .doc
DATASET_FILE = "NSF_Award_Search_cleaned.csv"

print("Loading AI Models...")

try:
    # 1. Load Semantic Embedder (MiniLM)
    print("Loading Semantic Encoder (MiniLM)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Load Keras Brain & Scaler
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "proposal_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_profiler.pkl"))
    
    with open(os.path.join(MODEL_DIR, "cluster_descriptions.json"), 'r') as f:
        cluster_descs = json.load(f)

    # 3. Load Comparison Database
    try:
        # Load from Vector DB for semantic comparisons
        vdb_path = os.path.join(MODEL_DIR, "vector_db.json")
        if os.path.exists(vdb_path):
            with open(vdb_path, 'r') as f:
                vdb = json.load(f)
                db_vectors = np.array(vdb['vectors'])
                db_titles = vdb['titles']
            print(f"   ...Comparison Mode: Semantic Vector DB ({len(db_titles)} items)")
        else:
            df = pd.read_csv(DATASET_FILE)
            db_titles = df['AwardTitle'].dropna().unique().tolist()
            print(f"   ...Encoding {len(db_titles)} titles...")
            db_vectors = embedder.encode(db_titles, show_progress_bar=False)
        
        comparison_source = "MiniLM Semantic DB"
        
    except Exception as e:
        print(f"   ...Error loading dataset ({e}), limited functionality")
        db_vectors = np.array([])
        db_titles = []
        comparison_source = "None"
        
    print("System Ready.\n")



except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# --- PART 1: THE MULTI-FORMAT INTELLIGENT PARSER ---
def extract_text_from_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        elif ext == ".doc":
            # Basic extraction for .doc using docx2txt (requires antiword/similar usually, but handles text)
            text = docx2txt.process(file_path)
        else:
            print(f"Unsupported file format: {ext}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    return text

def parse_document_data(file_path):
    text = extract_text_from_document(file_path)
    if not text: return None
    
    print(f"Parsing Document: {file_path}...")

    data = {
        "title": "Unknown Project",
        "duration": 12, 
        "cooperating_agencies": 0,
        "total": 0.0,
        "mooe": 0.0,
        "ps": 0.0,
        "co": 0.0
    }

    # 1. Semantic-Aware Title Extraction
    # We look for various synonyms and phrasing for 'Project Title'
    title_patterns = [
        r"(?:Project Title|Title of Project|Proposed Title|Project Name|Research Title)[:\s]+(.+)",
        r"(?:Name of Project|Study Title|Title)[:\s]+(.+)"
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found_title = match.group(1).strip()
            if len(found_title) > 5: # Filter junk
                data["title"] = found_title
                break

    # 2. Extract Duration (Supports 'Duration: X months' or '(In months) X')
    duration_patterns = [
        r"\(In months\)\s*(\d+)",
        r"Duration[:\s]+(\d+)\s*(?:months)?",
        r"Period[:\s]+(\d+)\s*(?:months)?"
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if val < 120: 
                data["duration"] = val
                break

    # 3. Extract Cooperating Agencies
    agency_section = re.search(r"Cooperating Agencies.*?\n(.*?)(?=\n\(\d\)|$|Classification|Budget)", text, re.IGNORECASE | re.DOTALL)
    if agency_section:
        raw_agencies = agency_section.group(1).strip()
        if len(raw_agencies) > 3 and "N/A" not in raw_agencies:
            count = raw_agencies.count(",") + 1
            # Also check for bullet points or newlines as separators
            if count == 1:
                alt_count = raw_agencies.count("\n") + 1
                count = max(count, alt_count)
            data["cooperating_agencies"] = count

    # 4. Extract Budget (Fuzzy keyword matching)
    numbers = re.findall(r"([\d,]+\.\d{2})", text)
    if numbers:
        clean_nums = []
        for n in numbers:
            try:
                val = float(n.replace(",", "").strip())
                clean_nums.append(val)
            except: pass
        
        if clean_nums:
            data["total"] = max(clean_nums)
            
            # Look for PS/Staff costs
            ps_match = re.search(r"(?:Personnel Services|PS).*?([\d,]+\.\d{2})", text, re.IGNORECASE | re.DOTALL)
            if ps_match: 
                data["ps"] = float(ps_match.group(1).replace(",", ""))
            
            # Look for MOOE/Ops costs
            mooe_match = re.search(r"(?:MOOE|Maintenance).*?([\d,]+\.\d{2})", text, re.IGNORECASE | re.DOTALL)
            if mooe_match: 
                data["mooe"] = float(mooe_match.group(1).replace(",", ""))
            
            # Calculate CO if needed
            if data["ps"] + data["mooe"] < data["total"]:
                data["co"] = data["total"] - (data["ps"] + data["mooe"])

    return data

# --- PART 2: THE AI ANALYSIS (SEMANTIC EQUIVALENCE MODE) ---
def analyze_document(file_path):
    extracted = parse_document_data(file_path)
    if not extracted: return

    print("\nEXTRACTED PARAMETERS:")
    print(f"   Title:    {extracted['title'][:70]}...")
    print(f"   Duration: {extracted['duration']} months")
    print(f"   Agencies: {extracted['cooperating_agencies']}")
    print(f"   Budget:   PHP {extracted['total']:,.2f}")
    
    # 1. Prediction Score (Semantic Integration)
    title_vec = embedder.encode([extracted['title']])[0]
    
    stats = scaler.transform([[
        extracted['duration'], extracted['mooe'], 
        extracted['ps'], extracted['co'], 
        extracted['total'], extracted['cooperating_agencies']
    ]])
    
    score = model.predict({
        'emb_input': np.array([title_vec]),
        'meta_input': stats
    }, verbose=0)[0][0] * 100
    
    # 2. Semantic Duplicate Check
    if len(db_vectors) > 0:
        sim_scores = cosine_similarity([title_vec], db_vectors)[0]
        best_match_idx = np.argmax(sim_scores)
        max_sim = sim_scores[best_match_idx]
    else:
        max_sim = 0
        best_match_idx = 0
    
    percentage = int(max_sim * 100)
    
    # User-requested interpretation scale
    if percentage <= 20: sim_status = "Not Related"
    elif percentage <= 40: sim_status = "Slightly Related"
    elif percentage <= 60: sim_status = "Moderately Similar"
    elif percentage <= 80: sim_status = "Highly Similar"
    else: sim_status = "Very Similar / Duplicate"

    # Novelty: Inverse of Semantic Similarity
    novelty = int((1 - max_sim) * 100)

    # 3. Profiling
    cluster_id = kmeans.predict(stats)[0]
    profile = cluster_descs.get(str(cluster_id), "General R&D Project")

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print(f"AI ANALYSIS REPORT")
    print("=" * 50)
    print(f"CLASSIFICATION: {profile}")
    print(f"OVERALL SCORE:  {int(score)}%")
    print(f"NOVELTY SCORE:  {novelty}%")
    print(f"SEMANTIC MATCH: {percentage}% ({sim_status})")
    
    if percentage > 60: 
        print("\nSTATUS: REJECTED (Duplicate Risk)")
        print(f"   Matched with: \"{db_titles[best_match_idx]}\"")
    elif score >= 70:
        print("\nSTATUS: PASSED")
    else:
        print("\nSTATUS: REVISION SUGGESTED")

    print("=" * 50)


if __name__ == "__main__":
    # Test with standard or found file
    target = DOCUMENT_FILE
    if not os.path.exists(target):
        # Scan for any document if default is missing
        docs = [f for f in os.listdir('.') if f.endswith(('.pdf', '.docx', '.doc'))]
        if docs: target = docs[0]
        
    analyze_document(target)
ze_pdf()
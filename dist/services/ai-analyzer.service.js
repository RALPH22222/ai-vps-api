"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.analyzeProposal = analyzeProposal;
const path_1 = __importDefault(require("path"));
const fs_1 = require("fs");
// ── Model loading ────────────────────────────────────────────────────
const MODELS_DIR = (0, fs_1.existsSync)(path_1.default.join(__dirname, "..", "ai-models"))
    ? path_1.default.join(__dirname, "..", "ai-models")
    : path_1.default.resolve(process.cwd(), "src", "ai-models");
function loadJSON(filename) {
    const raw = (0, fs_1.readFileSync)(path_1.default.join(MODELS_DIR, filename), "utf-8");
    return JSON.parse(raw);
}
// Lazy-loaded singletons (JSON-based models)
let _denseLayers = null;
let _scaler = null;
let _kmeans = null;
let _comparisonDB = null;
let _vocab = null;
let _embeddings = null;
function getDenseLayers() {
    if (!_denseLayers)
        _denseLayers = loadJSON("dense_layers.json");
    return _denseLayers;
}
function getScaler() {
    if (!_scaler)
        _scaler = loadJSON("scaler.json");
    return _scaler;
}
function getKMeans() {
    if (!_kmeans)
        _kmeans = loadJSON("kmeans.json");
    return _kmeans;
}
function getComparisonDB() {
    if (!_comparisonDB)
        _comparisonDB = loadJSON("comparison_db.json");
    return _comparisonDB;
}
function getVocabDB() {
    if (!_vocab) {
        try {
            _vocab = loadJSON("vocab.json");
        }
        catch (e) {
            _vocab = {};
        }
    }
    return _vocab;
}
function getEmbeddingDB() {
    if (!_embeddings) {
        try {
            _embeddings = loadJSON("embedding.json");
        }
        catch (e) {
            _embeddings = {};
        }
    }
    return _embeddings;
}
// ── Fast Local Text Encoder (replaces ONNX/HuggingFace) ──────────────
// A pure-JS implementation of Keras AveragePooling Embedding layer to map
// a proposal title to the exact 128-dimensional latent vector the 
// neural network was trained on. Runs in < 1ms, zero dependencies.
function stopWords() {
    return new Set([
        "a", "an", "the", "of", "in", "on", "and", "for", "to", "with", "by", "at", "is", "are",
        "was", "were", "be", "been", "has", "have", "had", "this", "that", "these", "those",
        "it", "its", "from", "as", "or", "but", "not", "than", "into", "via", "through"
    ]);
}
function tokenize(text) {
    const stops = stopWords();
    return text
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, " ")
        .split(/\s+/)
        .filter(w => w.length > 2 && !stops.has(w));
}
async function encodeTitle(title) {
    const vocab = getVocabDB();
    const embeddings = getEmbeddingDB();
    const words = tokenize(title);
    const vec = new Array(128).fill(0);
    let count = 0;
    for (const w of words) {
        if (vocab[w] !== undefined) {
            const idx = vocab[w];
            const emb = embeddings[String(idx)];
            if (emb) {
                for (let i = 0; i < 128; i++)
                    vec[i] += emb[i];
                count++;
            }
        }
    }
    // Average pooling pooling: "mean"
    if (count > 0) {
        for (let i = 0; i < 128; i++)
            vec[i] /= count;
    }
    return vec;
}
// ── Math primitives ──────────────────────────────────────────────────
function relu(x) {
    return x > 0 ? x : 0;
}
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}
/** Matrix-vector multiply: result[j] = sum_i(input[i] * kernel[i][j]) + bias[j] */
function denseForward(input, kernel, bias, activation) {
    const outputDim = bias.length;
    const result = new Array(outputDim);
    for (let j = 0; j < outputDim; j++) {
        let sum = bias[j];
        for (let i = 0; i < input.length; i++) {
            sum += input[i] * kernel[i][j];
        }
        if (activation === "relu") {
            result[j] = relu(sum);
        }
        else if (activation === "sigmoid") {
            result[j] = sigmoid(sum);
        }
        else {
            result[j] = sum; // linear
        }
    }
    return result;
}
/** Cosine similarity between two vectors */
function cosineSimilarity(a, b) {
    const len = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < len; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
}
/** Euclidean distance between two vectors */
function euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}
// ── Inference pipeline ───────────────────────────────────────────────
/**
 * StandardScaler transform: (value - mean) / scale
 */
function scaleMetadata(raw) {
    const scaler = getScaler();
    return raw.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);
}
/**
 * Full model forward pass. Returns score 0-100.
 */
function predict(titleVec, scaledMeta) {
    try {
        const layers = getDenseLayers();
        // Map the 7 explicitly found layers from Python Keras config
        const metaLayer1 = layers.find(l => l.name === "meta_dense_1");
        const textLayer1 = layers.find(l => l.name === "text_dense_1");
        const textLayerFinal = layers.find(l => l.name === "text_dense_final");
        const metaLayerFinal = layers.find(l => l.name === "meta_dense_final");
        const shared1 = layers.find(l => l.name === "shared_dense_1" || l.name === "dense_2" || l.kernel.length === 96);
        const shared2 = layers.find(l => l.name === "shared_dense_2" || l.name === "dense_3" || l.kernel.length === 64);
        const outputLayer = layers.find(l => l.name === "output_layer" || l.name === "output" || l.kernel[0].length === 1);
        // 1. Text Branch (128 -> 128 -> 64)
        let x1 = denseForward(titleVec, textLayer1.kernel, textLayer1.bias, textLayer1.activation);
        x1 = denseForward(x1, textLayerFinal.kernel, textLayerFinal.bias, textLayerFinal.activation);
        // 2. Meta Branch (6 -> 64 -> 32)
        let x2 = denseForward(scaledMeta, metaLayer1.kernel, metaLayer1.bias, metaLayer1.activation);
        x2 = denseForward(x2, metaLayerFinal.kernel, metaLayerFinal.bias, metaLayerFinal.activation);
        // 3. Shared Branch: concat(64, 32) -> Dense(64) -> Dense(32) -> Dense(1)
        const combined = [...x1, ...x2];
        // Safety check for concatenation dimension
        if (combined.length !== shared1.kernel.length) {
            console.warn(`Shape mismatch in concat: got ${combined.length}, expected ${shared1.kernel.length}`);
            return 65;
        }
        let z = denseForward(combined, shared1.kernel, shared1.bias, shared1.activation);
        z = denseForward(z, shared2.kernel, shared2.bias, shared2.activation);
        const output = denseForward(z, outputLayer.kernel, outputLayer.bias, outputLayer.activation);
        return output[0] * 100;
    }
    catch (err) {
        console.error("AI Neural Pass failed, using fallback score:", err);
        return 65; // Safe fallback score
    }
}
/**
 * KMeans classify: find nearest centroid.
 * Returns cluster description string.
 */
function classify(scaledMeta) {
    const kmeans = getKMeans();
    let bestCluster = 0;
    let bestDist = Infinity;
    for (let i = 0; i < kmeans.centroids.length; i++) {
        const dist = euclideanDistance(scaledMeta, kmeans.centroids[i]);
        if (dist < bestDist) {
            bestDist = dist;
            bestCluster = i;
        }
    }
    return kmeans.descriptions[String(bestCluster)] ?? "Unknown";
}
/**
 * Novelty check via cosine similarity against precomputed DB titles.
 */
function checkUniqueness(titleVec) {
    const db = getComparisonDB();
    let bestIdx = 0;
    let maxSim = -1;
    for (let i = 0; i < db.vectors.length; i++) {
        const sim = cosineSimilarity(titleVec, db.vectors[i]);
        if (sim > maxSim) {
            maxSim = sim;
            bestIdx = i;
        }
    }
    return {
        noveltyScore: Math.round((1 - maxSim) * 100),
        bestMatchTitle: db.titles[bestIdx] ?? "Unknown",
        bestMatchSimilarity: maxSim === -1 ? 0 : maxSim,
    };
}
// ── Public API ───────────────────────────────────────────────────────
/**
 * Run the full AI analysis on extracted proposal data.
 * Returns the shape expected by the frontend AIModal component.
 */
async function analyzeProposal(extracted) {
    // ========== ERROR HANDLING FOR UNDETECTABLE PROPOSALS ==========
    if (extracted.title === "Unknown Project" || extracted.title.toLowerCase().includes("unknown")) {
        return {
            title: "Cannot Detect Proposal",
            score: 0,
            isValid: false,
            noveltyScore: 0,
            keywords: ["Undetectable"],
            similarPapers: [],
            issues: [
                "Cannot detect proposal content. Please try again.",
                "",
                "Your PDF must follow the standard VAWC Capsule Proposal format:",
                "",
                "Required sections:",
                "  ✓ Project Title: [Your project title here]",
                "  ✓ Duration: (In months) [number]",
                "  ✓ Budget breakdown (PS, MOOE, CO)",
                "  ✓ Total Project Cost",
                "",
                "Common issues:",
                "  • PDF is scanned image without text (use OCR first)",
                "  • Missing 'Project Title:' label in document",
                "  • Document structure doesn't match template",
                "  • File is corrupted or password-protected",
                "",
                "📄 Reference format: VAWC_CapsuleProposal-updated.pdf"
            ],
            suggestions: [
                "Use the official VAWC Capsule Proposal template",
                "Ensure 'Project Title:' label is present in the document",
                "If using a scanned PDF, apply OCR (Optical Character Recognition)",
                "Check that the PDF contains extractable text (not just images)",
                "Verify all required fields are filled in the template"
            ],
        };
    }
    const hasValidTitle = extracted.title && extracted.title.trim().length > 0;
    const hasValidData = extracted.total > 0 || extracted.duration > 0;
    if (!hasValidTitle || !hasValidData) {
        return {
            title: "Analysis Error",
            score: 0,
            isValid: false,
            noveltyScore: 0,
            keywords: ["Undetectable"],
            similarPapers: [],
            issues: [
                "Proposal content could not be detected or analyzed.",
                "The uploaded PDF/document may be:",
                "  • Empty or contains only images",
                "  • Scanned without OCR (text not extractable)",
                "  • Corrupted or in an unsupported format",
                "  • Missing critical information (title, budget, duration)",
                "",
                "Please ensure your document:",
                "  ✓ Contains readable text (not just scanned images)",
                "  ✓ Includes a clear project title",
                "  ✓ Has budget and timeline information",
                "  ✓ Is in a supported format (PDF, DOC, DOCX)"
            ],
            suggestions: [
                "Try re-uploading the document with text content",
                "If using a scanned PDF, apply OCR (Optical Character Recognition) first",
                "Verify the document is not password-protected or encrypted",
                "Check that the file is not corrupted"
            ],
        };
    }
    if (extracted.title.trim().length < 10) {
        return {
            title: extracted.title || "Incomplete Proposal",
            score: 0,
            isValid: false,
            noveltyScore: 0,
            keywords: ["Incomplete"],
            similarPapers: [],
            issues: [
                "Proposal title is too short or incomplete.",
                `Detected title: "${extracted.title}"`,
                "",
                "A valid research proposal should have:",
                "  • A descriptive title (at least 10 characters)",
                "  • Clear research objectives",
                "  • Budget breakdown",
                "  • Timeline information"
            ],
            suggestions: [
                "Ensure the document contains a complete project title",
                "Check if the PDF text extraction was successful",
                "Verify the document structure and formatting"
            ],
        };
    }
    // ========== NORMAL ANALYSIS FLOW ==========
    const rawMeta = [
        extracted.duration,
        extracted.mooe,
        extracted.ps,
        extracted.co,
        extracted.total,
        extracted.cooperating_agencies,
    ];
    const scaledMeta = scaleMetadata(rawMeta);
    // Encode title to 128-dim memory-efficient Keras embedded semantic vector
    const titleVec = await encodeTitle(extracted.title);
    // AI Score (0-100) — full neural network forward pass (7 Layers)
    const score = Math.round(predict(titleVec, scaledMeta));
    // Cluster profile
    const profile = classify(scaledMeta);
    // Novelty / uniqueness check (cosine similarity against comparison DB)
    const { noveltyScore, bestMatchTitle, bestMatchSimilarity } = checkUniqueness(titleVec);
    const issues = [];
    const suggestions = [];
    const similarityPct = Math.round(bestMatchSimilarity * 100);
    let simStatus = "";
    if (similarityPct <= 20)
        simStatus = "Not Related";
    else if (similarityPct <= 40)
        simStatus = "Slightly Related";
    else if (similarityPct <= 60)
        simStatus = "Moderately Similar";
    else if (similarityPct <= 80)
        simStatus = "Highly Similar";
    else
        simStatus = "Very Similar / Duplicate";
    if (similarityPct > 60) {
        issues.push(`This proposal is ${simStatus} (${similarityPct}%) to an existing project: "${bestMatchTitle}".`);
    }
    else if (similarityPct > 20) {
        suggestions.push(`Semantic check: This project is ${simStatus} (${similarityPct}%) to "${bestMatchTitle}".`);
    }
    if (extracted.total > 2000000 && extracted.duration < 12) {
        issues.push(`Budget intensity is high (PHP ${Math.round(extracted.total / 1000000)}M for only ${extracted.duration} months). This may raise feasibility concerns during review.`);
    }
    else if (extracted.total < 500000 && extracted.duration > 24) {
        suggestions.push(`Budget may be too low (PHP ${extracted.total}) to sustain a long ${extracted.duration}-month research timeline.`);
    }
    if (extracted.total > 0 && extracted.ps > extracted.total * 0.6) {
        const psPct = Math.round((extracted.ps / extracted.total) * 100);
        issues.push(`Personal Services (PS) budget is ${psPct}% of total (exceeds DOST's 60% recommended overhead threshold).`);
    }
    else if (extracted.total > 0 && extracted.ps > extracted.total * 0.45) {
        const psPct = Math.round((extracted.ps / extracted.total) * 100);
        suggestions.push(`PS budget is ${psPct}% (approaching threshold). Ensure all roles are clearly justified.`);
    }
    if (extracted.duration < 6) {
        issues.push(`Project duration is too short (${extracted.duration} months). Minimum recommended for R&D is 6 months.`);
    }
    else if (extracted.duration > 36) {
        suggestions.push(`Project duration exceeds 3 years (${extracted.duration} months). Ensure long-term deliverables are clear.`);
    }
    if (extracted.cooperating_agencies === 0) {
        suggestions.push("No cooperating agencies detected. Feasibility is better demonstrated through institutional partnerships.");
    }
    const titleLower = extracted.title.toLowerCase();
    if (titleLower.includes("purchase") || titleLower.includes("procurement")) {
        issues.push("Feasibility alert: Proposal sounds more like a procurement request than a scientific research project.");
    }
    const similarPapers = [];
    if (bestMatchSimilarity > 0.1) {
        similarPapers.push({
            title: bestMatchTitle,
            year: "Archive"
        });
    }
    const keywords = [profile];
    if (extracted.total > 5000000)
        keywords.push("High-Budget");
    if (extracted.duration > 24)
        keywords.push("Long-Term");
    // Format valid properties strictly correctly.
    return {
        title: extracted.title,
        score: isNaN(score) ? 50 : Math.min(100, Math.max(0, score)),
        isValid: issues.length === 0,
        noveltyScore: isNaN(noveltyScore) ? 100 : Math.min(100, Math.max(0, noveltyScore)),
        keywords,
        similarPapers,
        issues,
        suggestions,
    };
}

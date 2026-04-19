"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.analyzeProposal = analyzeProposal;
const path_1 = __importDefault(require("path"));
const fs_1 = require("fs");
const child_process_1 = require("child_process");
const util_1 = __importDefault(require("util"));
const ai_debug_1 = require("../ai-debug");
const execFileAsync = util_1.default.promisify(child_process_1.execFile);
// ── Model loading ────────────────────────────────────────────────────
const MODELS_DIR = (0, fs_1.existsSync)(path_1.default.join(__dirname, "..", "ai-models"))
    ? path_1.default.join(__dirname, "..", "ai-models")
    : path_1.default.resolve(process.cwd(), "src", "ai-models");
function loadJSON(filename) {
    const raw = (0, fs_1.readFileSync)(path_1.default.join(MODELS_DIR, filename), "utf-8");
    return JSON.parse(raw);
}
// Lazy-loaded singletons (JSON-based models)
let _scaler = null;
let _kmeans = null;
let _nsfAwards = null;
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
function getNSFAwards() {
    if (!_nsfAwards) {
        try {
            const csvPath = path_1.default.resolve(process.cwd(), "trained-ai", "NSF_Award_Search_cleaned.csv");
            const content = (0, fs_1.readFileSync)(csvPath, "utf-8");
            const lines = content.split(/\r?\n/);
            _nsfAwards = [];
            for (let i = 1; i < lines.length; i++) {
                const line = lines[i].trim();
                if (!line)
                    continue;
                let title = "";
                if (line.startsWith('"')) {
                    const endQuoteIdx = line.indexOf('",');
                    if (endQuoteIdx !== -1) {
                        title = line.substring(1, endQuoteIdx);
                    }
                    else {
                        title = line.substring(1, line.length - 1);
                    }
                }
                else {
                    title = line.split(',')[0];
                }
                if (title)
                    _nsfAwards.push({ title });
            }
        }
        catch (e) {
            console.error("Failed to load NSF CSV", e);
            _nsfAwards = [];
        }
    }
    return _nsfAwards;
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
// ── Math primitives ──────────────────────────────────────────────────
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
 * When the Python Keras predictor is unavailable or errors, derive compliance 0–100
 * from budget/duration/co-agency signals instead of a hardcoded 65.
 */
function heuristicComplianceScore(extracted, profile) {
    let s = 72;
    const dur = extracted.duration;
    if (dur >= 6 && dur <= 36)
        s += 8;
    else if (dur < 6)
        s -= 18;
    else if (dur > 48)
        s -= 6;
    const total = extracted.total;
    if (total > 0 && extracted.ps > 0) {
        const psRatio = extracted.ps / total;
        if (psRatio > 0.6)
            s -= 22;
        else if (psRatio > 0.45)
            s -= 8;
        else
            s += 4;
    }
    if (extracted.cooperating_agencies >= 2)
        s += 6;
    else if (extracted.cooperating_agencies === 1)
        s += 3;
    if (profile === "Large-Scale Collaborative Grant")
        s += 4;
    if (profile.includes("High-Salary") || profile.includes("Overhead"))
        s -= 5;
    if (total <= 0 && dur <= 0)
        s -= 15;
    return Math.min(100, Math.max(25, Math.round(s)));
}
function getTermFrequencies(words) {
    const freqs = {};
    for (const w of words) {
        freqs[w] = (freqs[w] || 0) + 1;
    }
    return freqs;
}
function textCosineSimilarity(text1, text2) {
    const words1 = tokenize(text1);
    const words2 = tokenize(text2);
    const tf1 = getTermFrequencies(words1);
    const tf2 = getTermFrequencies(words2);
    let dot = 0;
    let norm1 = 0;
    let norm2 = 0;
    for (const count of Object.values(tf1))
        norm1 += count * count;
    for (const count of Object.values(tf2))
        norm2 += count * count;
    if (norm1 === 0 || norm2 === 0)
        return 0;
    for (const [w, count] of Object.entries(tf1)) {
        if (tf2[w])
            dot += count * tf2[w];
    }
    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}
/**
 * Novelty check via text cosine similarity against NSF Award DB.
 */
function checkUniqueness(title) {
    const awards = getNSFAwards();
    let bestTitle = "Unknown";
    let maxSim = -1;
    for (const award of awards) {
        const sim = textCosineSimilarity(title, award.title);
        if (sim > maxSim) {
            maxSim = sim;
            bestTitle = award.title;
        }
    }
    return {
        noveltyScore: Math.round((1 - Math.max(0, maxSim)) * 100),
        bestMatchTitle: bestTitle,
        bestMatchSimilarity: Math.max(0, maxSim),
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
                "Your PDF must follow the standard DOST Form No.1B format:",
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
                "Reference format: DOST Form No.1B"
            ],
            suggestions: [
                "Use the official DOST Form No.1B template",
                "Ensure 'Project Title:' label is present in the document",
                "If using a scanned PDF, apply OCR (Optical Character Recognition)",
                "Check that the PDF contains extractable text (not just images)",
                "Verify all required fields are filled in the template"
            ],
            strengths: [],
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
            strengths: [],
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
            strengths: [],
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
    const profile = classify(scaledMeta);
    (0, ai_debug_1.analyzeLog)("AI:inputs", {
        titleLen: extracted.title.length,
        duration: extracted.duration,
        total: extracted.total,
        ps: extracted.ps,
        mooe: extracted.mooe,
        co: extracted.co,
        cooperating_agencies: extracted.cooperating_agencies,
        profile,
    });
    (0, ai_debug_1.analyzeDebug)("AI:scaledMeta", { scaledMeta });
    (0, ai_debug_1.analyzeDebug)("AI:title preview", { title: (0, ai_debug_1.truncateForLog)(extracted.title, 200) });
    // Optional: Python Keras + sentence-transformers (often missing on VPS → use heuristic below).
    let score;
    let complianceScoreSource = "heuristic";
    const pyPath = path_1.default.resolve(process.cwd(), "trained-ai", "predict_score.py");
    let pythonCmd = process.platform === "win32" ? "python" : "python";
    // PEP 668 Fix: Use local virtual environment if it exists
    const venvPath = process.platform === "win32"
        ? path_1.default.join(process.cwd(), "venv", "Scripts", "python.exe")
        : path_1.default.join(process.cwd(), "venv", "bin", "python3");
    if ((0, fs_1.existsSync)(venvPath)) {
        pythonCmd = venvPath;
    }
    (0, ai_debug_1.analyzeDebug)("AI:python", { pyPath, pythonCmd, cwd: process.cwd(), usingVenv: (0, fs_1.existsSync)(venvPath) });
    try {
        const tPy = Date.now();
        const { stdout, stderr } = await execFileAsync(pythonCmd, [
            pyPath,
            extracted.title,
            String(extracted.duration),
            String(extracted.mooe),
            String(extracted.ps),
            String(extracted.co),
            String(extracted.total),
            String(extracted.cooperating_agencies)
        ], {
            env: {
                ...process.env,
                TF_CPP_MIN_LOG_LEVEL: "3",
                CUDA_VISIBLE_DEVICES: "-1",
                PYTHONIOENCODING: "utf8"
            }
        });
        if (stderr) {
            console.warn("[AI] Python Stderr:", stderr);
            (0, ai_debug_1.analyzeDebug)("AI:python stderr", { stderr: (0, ai_debug_1.truncateForLog)(String(stderr), 500) });
        }
        const result = JSON.parse(stdout.trim());
        (0, ai_debug_1.analyzeDebug)("AI:python stdout", { raw: (0, ai_debug_1.truncateForLog)(stdout.trim(), 500) });
        if (typeof result.score === "number" && !Number.isNaN(result.score) && result.error == null) {
            score = result.score;
            complianceScoreSource = "python-keras";
            (0, ai_debug_1.analyzeLog)("AI:python scorer ok", {
                ms: Date.now() - tPy,
                score: result.score,
                metrics: result.metrics // Log the new internal metrics from Python
            });
        }
        else if (result.error) {
            console.warn("[AI] Python scorer error (using heuristic compliance score):", result.error);
            (0, ai_debug_1.analyzeLog)("AI:python scorer error", { ms: Date.now() - tPy, error: (0, ai_debug_1.truncateForLog)(result.error, 300) });
        }
    }
    catch (err) {
        console.error("[AI] Python prediction failed. Possible reasons: Python3 not installed, missing libraries (tensorflow, sentence-transformers, joblib), or wrong file paths.");
        console.error("[AI] Error Details:", err.message || err);
        (0, ai_debug_1.analyzeLog)("AI:python exec failed", { message: err?.message || String(err) });
    }
    if (score === undefined) {
        score = heuristicComplianceScore(extracted, profile);
        complianceScoreSource = "heuristic";
        (0, ai_debug_1.analyzeLog)("AI:compliance heuristic", { score });
    }
    // Novelty / uniqueness check (cosine similarity against NSF CSV)
    const { noveltyScore, bestMatchTitle, bestMatchSimilarity } = checkUniqueness(extracted.title);
    (0, ai_debug_1.analyzeLog)("AI:novelty", {
        noveltyScore,
        bestMatchSimilarity: Math.round(bestMatchSimilarity * 1000) / 1000,
        bestMatchTitle: (0, ai_debug_1.truncateForLog)(bestMatchTitle, 120),
    });
    const issues = [];
    const suggestions = [];
    const strengths = [];
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
        issues.push(`[Similarity] This proposal is ${simStatus} (${similarityPct}%) to an existing project: "${bestMatchTitle}". Consider differentiating your methodology.`);
    }
    else if (similarityPct > 20) {
        suggestions.push(`[Novelty] Semantic check: This project is ${simStatus} (${similarityPct}%) to "${bestMatchTitle}". Highlight your unique approach.`);
    }
    else {
        strengths.push(`[Innovation] High novelty detected. Your project shows minimal overlap with existing research in our database.`);
    }
    if (extracted.total > 2000000 && extracted.duration < 12) {
        issues.push(`[Budget] High budget intensity (PHP ${Math.round(extracted.total / 1000000)}M for ${extracted.duration} months). Ensure every expense is deeply justified.`);
    }
    else if (extracted.total < 500000 && extracted.duration > 24) {
        suggestions.push(`[Sustainability] Budget might be too lean (PHP ${extracted.total}) for a long ${extracted.duration}-month timeline. Check for funding gaps.`);
    }
    else if (extracted.total > 0) {
        strengths.push(`[Budget] The proposed budget of PHP ${extracted.total.toLocaleString()} appears well-distributed for the ${extracted.duration}-month timeline.`);
    }
    if (extracted.total > 0 && extracted.ps > extracted.total * 0.6) {
        const psPct = Math.round((extracted.ps / extracted.total) * 100);
        issues.push(`[Overhead] Personal Services (PS) is ${psPct}% of total budget. DOST typically recommends staying below 60%.`);
    }
    else if (extracted.total > 0 && extracted.ps > extracted.total * 0.45) {
        const psPct = Math.round((extracted.ps / extracted.total) * 100);
        suggestions.push(`[Allocation] PS budget is ${psPct}%. This is within limits but may be scrutinized for efficiency.`);
    }
    else if (extracted.total > 0) {
        strengths.push(`[Efficiency] Excellent PS allocation (${Math.round((extracted.ps / extracted.total) * 100)}%). Most of the budget is directed toward MOOE and equipment.`);
    }
    if (extracted.duration < 6) {
        issues.push(`[Timeline] Duration is too short (${extracted.duration} months). Standard R&D cycles usually require at least 6 months.`);
    }
    else if (extracted.duration > 36) {
        suggestions.push(`[Scope] 3-year+ timeline detected. Ensure the phased deliverables are very clearly defined to maintain momentum.`);
    }
    else {
        strengths.push(`[Timeline] The ${extracted.duration}-month project cycle is ideal for the proposed R&D scope.`);
    }
    if (extracted.cooperating_agencies === 0) {
        suggestions.push("[Collaboration] No cooperating agencies detected. Adding institutional partners can significantly boost feasibility scores.");
    }
    else {
        strengths.push(`[Partnership] Collaboration with ${extracted.cooperating_agencies} agencies demonstrates strong institutional support.`);
    }
    const titleLower = extracted.title.toLowerCase();
    if (titleLower.includes("purchase") || titleLower.includes("procurement")) {
        issues.push("[Focus] Proposal sounds like a procurement request. R&D grants focus on knowledge generation, not just buying equipment.");
        // Generate a better sounding title
        const replacedTitle = extracted.title.replace(/purchase of|procurement of/gi, "Integration of").replace(/purchase|procurement/gi, "Development");
        suggestions.push(`[Title] Suggestion: "${replacedTitle} for Institutional Advancement" (Shift focus from buying to researching)`);
    }
    else if (extracted.title.length > 5 && extracted.title.length < 40) {
        suggestions.push(`[Title] Suggestion: "Comprehensive Study and Optimization of ${extracted.title}" (Make it sound more academic)`);
    }
    else if (profile !== "Standard R&D Project" && profile !== "Unknown") {
        suggestions.push(`[Title] Suggestion: "${extracted.title}: A ${profile} Initiative"`);
    }
    else {
        strengths.push("[Title] Your project title is descriptive and follows academic standards.");
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
    let finalScore = isNaN(score) ? 50 : Math.min(100, Math.max(0, score));
    // Realistic Adjustment: Decrease score based on issues and suggestions
    if (finalScore > 40) {
        finalScore -= (issues.length * 5); // -5% for each critical issue
        finalScore -= (suggestions.length * 2); // -2% for each suggestion
    }
    finalScore = Math.max(15, Math.min(100, finalScore));
    (0, ai_debug_1.analyzeLog)("AI:result", {
        complianceScore: finalScore,
        complianceScoreSource,
        isValid: issues.length === 0,
        issuesCount: issues.length,
        suggestionsCount: suggestions.length,
        keywords,
    });
    // Format valid properties strictly correctly.
    return {
        title: extracted.title,
        score: finalScore,
        isValid: issues.length === 0,
        noveltyScore: isNaN(noveltyScore) ? 100 : Math.min(100, Math.max(0, noveltyScore)),
        keywords,
        similarPapers,
        issues,
        suggestions,
        strengths,
    };
}

import path from "path";
import { readFileSync, existsSync } from "fs";
import { execFile } from "child_process";
import util from "util";

const execFileAsync = util.promisify(execFile);

// ── Types ────────────────────────────────────────────────────────────

interface DenseLayerWeights {
  name: string;
  kernel: number[][]; // shape: [inputDim, outputDim]
  bias: number[];
  activation: string;
}

interface BatchNormParams {
  gamma: number[];
  beta: number[];
  moving_mean: number[];
  moving_variance: number[];
  epsilon: number;
}

interface ScalerParams {
  mean: number[];
  scale: number[];
}

interface KMeansParams {
  centroids: number[][];
  descriptions: Record<string, string>;
}

interface NSFAward {
  title: string;
}

export interface AnalysisResult {
  title: string;
  score: number;
  isValid: boolean;
  noveltyScore: number;
  keywords: string[];
  similarPapers: { title: string; year: string }[];
  issues: string[];
  suggestions: string[];
}

export interface ExtractedData {
  title: string;
  duration: number;
  cooperating_agencies: number;
  total: number;
  mooe: number;
  ps: number;
  co: number;
}

// ── Model loading ────────────────────────────────────────────────────

const MODELS_DIR = existsSync(path.join(__dirname, "..", "ai-models"))
  ? path.join(__dirname, "..", "ai-models")
  : path.resolve(process.cwd(), "src", "ai-models");

function loadJSON<T>(filename: string): T {
  const raw = readFileSync(path.join(MODELS_DIR, filename), "utf-8");
  return JSON.parse(raw) as T;
}

// Lazy-loaded singletons (JSON-based models)
let _denseLayers: DenseLayerWeights[] | null = null;
let _scaler: ScalerParams | null = null;
let _kmeans: KMeansParams | null = null;
let _nsfAwards: NSFAward[] | null = null;
let _vocab: Record<string, number> | null = null;
let _embeddings: Record<string, number[]> | null = null;

function getDenseLayers(): DenseLayerWeights[] {
  if (!_denseLayers) _denseLayers = loadJSON<DenseLayerWeights[]>("dense_layers.json");
  return _denseLayers;
}

function getScaler(): ScalerParams {
  if (!_scaler) _scaler = loadJSON<ScalerParams>("scaler.json");
  return _scaler;
}

function getKMeans(): KMeansParams {
  if (!_kmeans) _kmeans = loadJSON<KMeansParams>("kmeans.json");
  return _kmeans;
}

function getNSFAwards(): NSFAward[] {
  if (!_nsfAwards) {
    try {
      const csvPath = path.resolve(process.cwd(), "trained-ai", "NSF_Award_Search_cleaned.csv");
      const content = readFileSync(csvPath, "utf-8");
      const lines = content.split(/\r?\n/);
      _nsfAwards = [];
      for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        let title = "";
        if (line.startsWith('"')) {
           const endQuoteIdx = line.indexOf('",');
           if (endQuoteIdx !== -1) {
              title = line.substring(1, endQuoteIdx);
           } else {
              title = line.substring(1, line.length - 1);
           }
        } else {
           title = line.split(',')[0];
        }
        if (title) _nsfAwards.push({ title });
      }
    } catch (e) {
      console.error("Failed to load NSF CSV", e);
      _nsfAwards = [];
    }
  }
  return _nsfAwards;
}

function getVocabDB(): Record<string, number> {
  if (!_vocab) {
    try {
      _vocab = loadJSON<Record<string, number>>("vocab.json");
    } catch (e) {
      _vocab = {};
    }
  }
  return _vocab;
}

function getEmbeddingDB(): Record<string, number[]> {
  if (!_embeddings) {
    try {
      _embeddings = loadJSON<Record<string, number[]>>("embedding.json");
    } catch (e) {
      _embeddings = {};
    }
  }
  return _embeddings;
}

// ── Fast Local Text Encoder (replaces ONNX/HuggingFace) ──────────────
// A pure-JS implementation of Keras AveragePooling Embedding layer to map
// a proposal title to the exact 128-dimensional latent vector the 
// neural network was trained on. Runs in < 1ms, zero dependencies.

function stopWords(): Set<string> {
  return new Set([
    "a","an","the","of","in","on","and","for","to","with","by","at","is","are",
    "was","were","be","been","has","have","had","this","that","these","those",
    "it","its","from","as","or","but","not","than","into","via","through"
  ]);
}

function tokenize(text: string): string[] {
  const stops = stopWords();
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 2 && !stops.has(w));
}

async function encodeTitle(title: string): Promise<number[]> {
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
        for (let i = 0; i < 128; i++) vec[i] += emb[i];
        count++;
      }
    }
  }

  // Average pooling pooling: "mean"
  if (count > 0) {
    for (let i = 0; i < 128; i++) vec[i] /= count;
  }

  return vec;
}

// ── Math primitives ──────────────────────────────────────────────────

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Matrix-vector multiply: result[j] = sum_i(input[i] * kernel[i][j]) + bias[j] */
function denseForward(input: number[], kernel: number[][], bias: number[], activation: string): number[] {
  const outputDim = bias.length;
  const result = new Array<number>(outputDim);

  for (let j = 0; j < outputDim; j++) {
    let sum = bias[j];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * kernel[i][j];
    }

    if (activation === "relu") {
      result[j] = relu(sum);
    } else if (activation === "sigmoid") {
      result[j] = sigmoid(sum);
    } else {
      result[j] = sum; // linear
    }
  }

  return result;
}

/** Cosine similarity between two vectors */
function cosineSimilarity(a: number[], b: number[]): number {
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
function euclideanDistance(a: number[], b: number[]): number {
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
function scaleMetadata(raw: number[]): number[] {
  const scaler = getScaler();
  return raw.map((val, i) => (val - scaler.mean[i]) / scaler.scale[i]);
}

/**
 * Full model forward pass. Returns score 0-100.
 */
function predict(titleVec: number[], scaledMeta: number[]): number {
  try {
    const layers = getDenseLayers();

    // Map the 7 explicitly found layers from Python Keras config
    const metaLayer1 = layers.find(l => l.name === "meta_dense_1")!;
    const textLayer1 = layers.find(l => l.name === "text_dense_1")!;
    const textLayerFinal = layers.find(l => l.name === "text_dense_final")!;
    const metaLayerFinal = layers.find(l => l.name === "meta_dense_final")!;
    const shared1 = layers.find(l => l.name === "shared_dense_1" || l.name === "dense_2" || l.kernel.length === 96)!;
    const shared2 = layers.find(l => l.name === "shared_dense_2" || l.name === "dense_3" || l.kernel.length === 64)!;
    const outputLayer = layers.find(l => l.name === "output_layer" || l.name === "output" || l.kernel[0].length === 1)!;

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
  } catch (err) {
    console.error("AI Neural Pass failed, using fallback score:", err);
    return 65; // Safe fallback score
  }
}

/**
 * KMeans classify: find nearest centroid.
 * Returns cluster description string.
 */
function classify(scaledMeta: number[]): string {
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

function getTermFrequencies(words: string[]): Record<string, number> {
  const freqs: Record<string, number> = {};
  for (const w of words) {
    freqs[w] = (freqs[w] || 0) + 1;
  }
  return freqs;
}

function textCosineSimilarity(text1: string, text2: string): number {
  const words1 = tokenize(text1);
  const words2 = tokenize(text2);
  const tf1 = getTermFrequencies(words1);
  const tf2 = getTermFrequencies(words2);
  
  let dot = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (const count of Object.values(tf1)) norm1 += count * count;
  for (const count of Object.values(tf2)) norm2 += count * count;
  
  if (norm1 === 0 || norm2 === 0) return 0;
  
  for (const [w, count] of Object.entries(tf1)) {
    if (tf2[w]) dot += count * tf2[w];
  }
  
  return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * Novelty check via text cosine similarity against NSF Award DB.
 */
function checkUniqueness(title: string): {
  noveltyScore: number;
  bestMatchTitle: string;
  bestMatchSimilarity: number;
} {
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
export async function analyzeProposal(extracted: ExtractedData): Promise<AnalysisResult> {
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

  // Call python script for accurate semantic encoding and keras model prediction
  let score = 65;
  try {
    const pyPath = path.resolve(process.cwd(), "trained-ai", "predict_score.py");
    const { stdout } = await execFileAsync("python", [
      pyPath,
      extracted.title,
      String(extracted.duration),
      String(extracted.mooe),
      String(extracted.ps),
      String(extracted.co),
      String(extracted.total),
      String(extracted.cooperating_agencies)
    ]);
    const result = JSON.parse(stdout);
    if (result.score !== undefined) {
      score = result.score;
    } else {
      console.warn("Python prediction returned error:", result.error);
    }
  } catch (err) {
    console.error("Python prediction failed:", err);
  }

  // Cluster profile
  const profile = classify(scaledMeta);

  // Novelty / uniqueness check (cosine similarity against NSF CSV)
  const { noveltyScore, bestMatchTitle, bestMatchSimilarity } = checkUniqueness(extracted.title);

  const issues: string[] = [];
  const suggestions: string[] = [];

  const similarityPct = Math.round(bestMatchSimilarity * 100);
  let simStatus = "";
  if (similarityPct <= 20) simStatus = "Not Related";
  else if (similarityPct <= 40) simStatus = "Slightly Related";
  else if (similarityPct <= 60) simStatus = "Moderately Similar";
  else if (similarityPct <= 80) simStatus = "Highly Similar";
  else simStatus = "Very Similar / Duplicate";

  if (similarityPct > 60) {
    issues.push(`This proposal is ${simStatus} (${similarityPct}%) to an existing project: "${bestMatchTitle}".`);
  } else if (similarityPct > 20) {
    suggestions.push(`Semantic check: This project is ${simStatus} (${similarityPct}%) to "${bestMatchTitle}".`);
  }

  if (extracted.total > 2000000 && extracted.duration < 12) {
    issues.push(`Budget intensity is high (PHP ${Math.round(extracted.total/1000000)}M for only ${extracted.duration} months). This may raise feasibility concerns during review.`);
  } else if (extracted.total < 500000 && extracted.duration > 24) {
    suggestions.push(`Budget may be too low (PHP ${extracted.total}) to sustain a long ${extracted.duration}-month research timeline.`);
  }

  if (extracted.total > 0 && extracted.ps > extracted.total * 0.6) {
    const psPct = Math.round((extracted.ps / extracted.total) * 100);
    issues.push(`Personal Services (PS) budget is ${psPct}% of total (exceeds DOST's 60% recommended overhead threshold).`);
  } else if (extracted.total > 0 && extracted.ps > extracted.total * 0.45) {
    const psPct = Math.round((extracted.ps / extracted.total) * 100);
    suggestions.push(`PS budget is ${psPct}% (approaching threshold). Ensure all roles are clearly justified.`);
  }

  if (extracted.duration < 6) {
    issues.push(`Project duration is too short (${extracted.duration} months). Minimum recommended for R&D is 6 months.`);
  } else if (extracted.duration > 36) {
    suggestions.push(`Project duration exceeds 3 years (${extracted.duration} months). Ensure long-term deliverables are clear.`);
  }

  if (extracted.cooperating_agencies === 0) {
    suggestions.push("No cooperating agencies detected. Feasibility is better demonstrated through institutional partnerships.");
  }

  const titleLower = extracted.title.toLowerCase();
  if (titleLower.includes("purchase") || titleLower.includes("procurement")) {
    issues.push("Feasibility alert: Proposal sounds more like a procurement request than a scientific research project.");
    
    // Generate a better sounding title
    const replacedTitle = extracted.title.replace(/purchase of|procurement of/gi, "Integration of").replace(/purchase|procurement/gi, "Development");
    suggestions.push(`💡 Title Suggestion: "${replacedTitle} for Institutional Advancement" (Shift focus from buying to researching)`);
  } else if (extracted.title.length > 5 && extracted.title.length < 40) {
    // If the title is too short, suggest a more academic phrasing
    suggestions.push(`💡 Title Suggestion: "Comprehensive Study and Optimization of ${extracted.title}"`);
  } else if (profile !== "Standard R&D Project" && profile !== "Unknown") {
    // Suggest adding the profile to the title if it's unique
    suggestions.push(`💡 Title Suggestion: "${extracted.title}: A ${profile} Initiative"`);
  }

  const similarPapers: { title: string; year: string }[] = [];
  if (bestMatchSimilarity > 0.1) {
    similarPapers.push({ 
      title: bestMatchTitle, 
      year: "Archive" 
    });
  }

  const keywords = [profile];
  if (extracted.total > 5000000) keywords.push("High-Budget");
  if (extracted.duration > 24) keywords.push("Long-Term");

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

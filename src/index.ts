import express, { Request, Response } from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import { analyzeProposal } from "./services/ai-analyzer.service";
import { 
  extractTextFromFile, 
  extractDataFromText, 
  extractFormFields, 
  SUPPORTED_TYPES 
} from "./services/document-extractor.service";
import { analyzeLog, isAiDebug } from "./ai-debug";
// hello
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5001;

// Configure Multer for memory storage
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  }
});

app.use(cors());
app.use(express.json());

// Custom error handler for Multer
const handleMulterError = (err: any, req: Request, res: Response, next: any) => {
  if (err instanceof multer.MulterError) {
    console.error(`[AI] Multer Error: ${err.message} (Field: ${err.field})`);
    return res.status(400).json({ 
      message: "File upload error", 
      error: err.code,
      field: err.field 
    });
  }
  next(err);
};

// Health check
app.get("/health", (req: Request, res: Response) => {
  res.json({ status: "ok", service: "AI Analysis API", timestamp: new Date().toISOString() });
});

/**
 * POST /analyze
 * Handles both JSON data or File Upload (multipart/form-data)
 */
app.post("/analyze", (req, res, next) => {
  upload.any()(req, res, (err) => {
    if (err) return handleMulterError(err, req, res, next);
    next();
  });
}, async (req: Request, res: Response) => {
  const reqId = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  const t0 = Date.now();
  analyzeLog(`HTTP:begin`, { reqId, path: "/analyze", debugAi: isAiDebug() });
  try {
    let extractedData;
    let formFields = {};

    // Get files from upload.any() array
    const files = req.files as Express.Multer.File[] | undefined;
    const file = files?.[0]; // Default to the first file found

    // Scenario A: File Upload
    if (file) {
      analyzeLog(`HTTP:file`, {
        reqId,
        originalname: file.originalname,
        size: file.size,
        mimetype: file.mimetype,
      });
      
      if (!SUPPORTED_TYPES.includes(file.mimetype as any)) {
        console.warn(`[AI] Rejected file type: ${file.mimetype}`);
        return res.status(400).json({ 
          message: `Unsupported file format: ${file.mimetype}. Please upload PDF or DOCX.` 
        });
      }

      const tExtract = Date.now();
      const text = await extractTextFromFile(file.buffer, file.mimetype);
      analyzeLog(`HTTP:text ready`, { reqId, ms: Date.now() - tExtract, chars: text.length });
      extractedData = extractDataFromText(text);
      formFields = extractFormFields(text);
      const formKeys = Object.keys(formFields as object);
      analyzeLog(`HTTP:parsed`, { reqId, formFieldCount: formKeys.length, formKeys });
    } 
    // Scenario B: Direct JSON
    else if (req.body && (req.body.title || Object.keys(req.body).length > 0)) {
      analyzeLog(`HTTP:json body`, {
        reqId,
        title: req.body.title || "(no title)",
        keys: Object.keys(req.body),
      });
      extractedData = req.body;
    } 
    else {
      console.warn(`[AI] No file or body data received! Body:`, req.body);
      return res.status(400).json({ message: "No file or valid JSON data provided." });
    }

    const tAi = Date.now();
    const result = await analyzeProposal(extractedData);
    analyzeLog(`HTTP:analyze done`, { reqId, ms: Date.now() - tAi, score: result.score, isValid: result.isValid });

    analyzeLog(`HTTP:complete`, {
      reqId,
      totalMs: Date.now() - t0,
      title: result.title?.slice(0, 80),
    });
    // Merge form fields if they were extracted
    return res.json({ ...result, formFields });
    
  } catch (error: any) {
    console.error("[AI] Fatal Exception:", error);
    return res.status(500).json({ 
      message: "Internal AI Engine Error",
      error: error.message || String(error)
    });
  }
});

app.listen(PORT, () => {
  console.log(`AI Analysis VPS API running on http://localhost:${PORT}`);
  console.log(`Document parsing (mammoth/pdf-parse) enabled.`);
  console.log(
    `🔧 Verbose AI/extractor logs: set DEBUG_AI=1 (or true) — previews and python stdout/stderr snippets.`
  );
});

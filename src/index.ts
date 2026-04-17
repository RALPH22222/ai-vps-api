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
  console.log(`[AI] POST /analyze - ${new Date().toISOString()}`);
  try {
    let extractedData;
    let formFields = {};

    // Get files from upload.any() array
    const files = req.files as Express.Multer.File[] | undefined;
    const file = files?.[0]; // Default to the first file found

    // Scenario A: File Upload
    if (file) {
      console.log(`[AI] File detected: ${file.originalname}, Size: ${file.size}, Type: ${file.mimetype}`);
      
      if (!SUPPORTED_TYPES.includes(file.mimetype as any)) {
        console.warn(`[AI] Rejected file type: ${file.mimetype}`);
        return res.status(400).json({ 
          message: `Unsupported file format: ${file.mimetype}. Please upload PDF or DOCX.` 
        });
      }

      console.log(`[AI] Starting document extraction...`);
      const text = await extractTextFromFile(file.buffer, file.mimetype);
      console.log(`[AI] Extraction complete. Parsing metadata...`);
      extractedData = extractDataFromText(text);
      formFields = extractFormFields(text);
    } 
    // Scenario B: Direct JSON
    else if (req.body && (req.body.title || Object.keys(req.body).length > 0)) {
      console.log(`[AI] JSON body detected for: "${req.body.title || "Unknown"}"`);
      extractedData = req.body;
    } 
    else {
      console.warn(`[AI] No file or body data received! Body:`, req.body);
      return res.status(400).json({ message: "No file or valid JSON data provided." });
    }

    console.log(`[AI] Running neural analysis...`);
    // Run AI analysis
    const result = await analyzeProposal(extractedData);
    
    console.log(`[AI] Analysis successful for: "${result.title}"`);
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
  console.log(`🚀 AI Analysis VPS API running on http://localhost:${PORT}`);
  console.log(`📁 Document parsing (mammoth/pdf-parse) enabled.`);
});

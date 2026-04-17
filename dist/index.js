"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
const multer_1 = __importDefault(require("multer"));
const ai_analyzer_service_1 = require("./services/ai-analyzer.service");
const document_extractor_service_1 = require("./services/document-extractor.service");
dotenv_1.default.config();
const app = (0, express_1.default)();
const PORT = process.env.PORT || 5001;
// Configure Multer for memory storage
const upload = (0, multer_1.default)({
    storage: multer_1.default.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limit
    }
});
app.use((0, cors_1.default)());
app.use(express_1.default.json());
// Custom error handler for Multer
const handleMulterError = (err, req, res, next) => {
    if (err instanceof multer_1.default.MulterError) {
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
app.get("/health", (req, res) => {
    res.json({ status: "ok", service: "AI Analysis API", timestamp: new Date().toISOString() });
});
/**
 * POST /analyze
 * Handles both JSON data or File Upload (multipart/form-data)
 */
app.post("/analyze", (req, res, next) => {
    upload.any()(req, res, (err) => {
        if (err)
            return handleMulterError(err, req, res, next);
        next();
    });
}, async (req, res) => {
    console.log(`[AI] POST /analyze - ${new Date().toISOString()}`);
    try {
        let extractedData;
        let formFields = {};
        // Get files from upload.any() array
        const files = req.files;
        const file = files?.[0]; // Default to the first file found
        // Scenario A: File Upload
        if (file) {
            console.log(`[AI] File detected: ${file.originalname}, Size: ${file.size}, Type: ${file.mimetype}`);
            if (!document_extractor_service_1.SUPPORTED_TYPES.includes(file.mimetype)) {
                console.warn(`[AI] Rejected file type: ${file.mimetype}`);
                return res.status(400).json({
                    message: `Unsupported file format: ${file.mimetype}. Please upload PDF or DOCX.`
                });
            }
            console.log(`[AI] Starting document extraction...`);
            const text = await (0, document_extractor_service_1.extractTextFromFile)(file.buffer, file.mimetype);
            console.log(`[AI] Extraction complete. Parsing metadata...`);
            extractedData = (0, document_extractor_service_1.extractDataFromText)(text);
            formFields = (0, document_extractor_service_1.extractFormFields)(text);
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
        const result = await (0, ai_analyzer_service_1.analyzeProposal)(extractedData);
        console.log(`[AI] Analysis successful for: "${result.title}"`);
        // Merge form fields if they were extracted
        return res.json({ ...result, formFields });
    }
    catch (error) {
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

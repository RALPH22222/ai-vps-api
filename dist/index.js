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
const ai_debug_1 = require("./ai-debug");
// hello hello
dotenv_1.default.config();
const app = (0, express_1.default)();
const PORT = process.env.PORT || 5001;
// Configure Multer for memory storage
const upload = (0, multer_1.default)({
    storage: multer_1.default.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024, // 10MB limits
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
    const reqId = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    const t0 = Date.now();
    (0, ai_debug_1.analyzeLog)(`HTTP:begin`, { reqId, path: "/analyze", debugAi: (0, ai_debug_1.isAiDebug)() });
    try {
        let extractedData;
        let formFields = {};
        // Get files from upload.any() array
        const files = req.files;
        const file = files?.[0]; // Default to the first file found
        // Scenario A: File Upload
        if (file) {
            (0, ai_debug_1.analyzeLog)(`HTTP:file`, {
                reqId,
                originalname: file.originalname,
                size: file.size,
                mimetype: file.mimetype,
            });
            if (!document_extractor_service_1.SUPPORTED_TYPES.includes(file.mimetype)) {
                console.warn(`[AI] Rejected file type: ${file.mimetype}`);
                return res.status(400).json({
                    message: `Unsupported file format: ${file.mimetype}. Please upload PDF or DOCX.`
                });
            }
            const tExtract = Date.now();
            const text = await (0, document_extractor_service_1.extractTextFromFile)(file.buffer, file.mimetype);
            (0, ai_debug_1.analyzeLog)(`HTTP:text ready`, { reqId, ms: Date.now() - tExtract, chars: text.length });
            extractedData = (0, document_extractor_service_1.extractDataFromText)(text);
            formFields = (0, document_extractor_service_1.extractFormFields)(text);
            const formKeys = Object.keys(formFields);
            (0, ai_debug_1.analyzeLog)(`HTTP:parsed`, { reqId, formFieldCount: formKeys.length, formKeys });
        }
        // Scenario B: Direct JSON
        else if (req.body && (req.body.title || Object.keys(req.body).length > 0)) {
            (0, ai_debug_1.analyzeLog)(`HTTP:json body`, {
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
        const result = await (0, ai_analyzer_service_1.analyzeProposal)(extractedData);
        (0, ai_debug_1.analyzeLog)(`HTTP:analyze done`, { reqId, ms: Date.now() - tAi, score: result.score, isValid: result.isValid });
        (0, ai_debug_1.analyzeLog)(`HTTP:complete`, {
            reqId,
            totalMs: Date.now() - t0,
            title: result.title?.slice(0, 80),
        });
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
    console.log(`AI Analysis VPS API running on http://localhost:${PORT}`);
    console.log(`Document parsing (mammoth/pdf-parse) enabled.`);
    console.log(`🔧 Verbose AI/extractor logs: set DEBUG_AI=1 (or true) — previews and python stdout/stderr snippets.`);
});

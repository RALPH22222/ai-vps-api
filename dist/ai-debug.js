"use strict";
/**
 * Logging helpers for /analyze (VPS troubleshooting).
 *
 * - Always: concise `[Analyze]` lines (phase + timing + summary).
 * - Verbose: set `DEBUG_AI=1` (or `true`) in env for previews and structured detail.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.isAiDebug = isAiDebug;
exports.analyzeLog = analyzeLog;
exports.analyzeDebug = analyzeDebug;
exports.truncateForLog = truncateForLog;
exports.extractorLog = extractorLog;
exports.extractorDebug = extractorDebug;
function isAiDebug() {
    const v = process.env.DEBUG_AI;
    return v === "1" || v === "true" || v === "yes";
}
/** Always printed — keep one line, safe for production logs. */
function analyzeLog(phase, detail) {
    const extra = detail && Object.keys(detail).length ? ` ${JSON.stringify(detail)}` : "";
    console.log(`[Analyze] [${phase}] ${new Date().toISOString()}${extra}`);
}
/** Only when DEBUG_AI is set — previews, keys, paths. */
function analyzeDebug(phase, detail) {
    if (!isAiDebug())
        return;
    analyzeLog(`DEBUG:${phase}`, detail);
}
function truncateForLog(s, max = 240) {
    const t = s.replace(/\s+/g, " ").trim();
    if (t.length <= max)
        return t;
    return `${t.slice(0, max)}…`;
}
/** Same as `[Analyze]` but prefixed `[Extractor]` for document pipeline. */
function extractorLog(message, meta) {
    const extra = meta && Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : "";
    console.log(`[Extractor] ${new Date().toISOString()} ${message}${extra}`);
}
function extractorDebug(message, meta) {
    if (!isAiDebug())
        return;
    extractorLog(`DEBUG ${message}`, meta);
}

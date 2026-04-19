/**
 * Logging helpers for /analyze (VPS troubleshooting).
 *
 * - Always: concise `[Analyze]` lines (phase + timing + summary).
 * - Verbose: set `DEBUG_AI=1` (or `true`) in env for previews and structured detail.
 */

export function isAiDebug(): boolean {
  const v = process.env.DEBUG_AI;
  return v === "1" || v === "true" || v === "yes";
}

/** Always printed — keep one line, safe for production logs. */
export function analyzeLog(phase: string, detail?: Record<string, unknown>): void {
  const extra = detail && Object.keys(detail).length ? ` ${JSON.stringify(detail)}` : "";
  console.log(`[Analyze] [${phase}] ${new Date().toISOString()}${extra}`);
}

/** Only when DEBUG_AI is set — previews, keys, paths. */
export function analyzeDebug(phase: string, detail?: Record<string, unknown>): void {
  if (!isAiDebug()) return;
  analyzeLog(`DEBUG:${phase}`, detail);
}

export function truncateForLog(s: string, max = 240): string {
  const t = s.replace(/\s+/g, " ").trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max)}…`;
}

/** Same as `[Analyze]` but prefixed `[Extractor]` for document pipeline. */
export function extractorLog(message: string, meta?: Record<string, unknown>): void {
  const extra = meta && Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : "";
  console.log(`[Extractor] ${new Date().toISOString()} ${message}${extra}`);
}

export function extractorDebug(message: string, meta?: Record<string, unknown>): void {
  if (!isAiDebug()) return;
  extractorLog(`DEBUG ${message}`, meta);
}

import mammoth from "mammoth";
import { ExtractedData } from "./ai-analyzer.service";
import { extractorDebug, extractorLog, truncateForLog } from "../ai-debug";

// pdf-parse import can be tricky across different Node/TS environments.
// We try the standard import first, then fall back if needed.
let pdf: any;
try {
  pdf = require("pdf-parse");
} catch (e) {
  pdf = require("pdf-parse/lib/pdf-parse");
}

// eslint-disable-next-line @typescript-eslint/no-var-requires
const officeParser = require("officeparser");

/** Fields extracted from the DOST template for auto-filling the submission form. */
export interface FormExtractedFields {
  program_title?: string;
  project_title?: string;
  year?: string;
  agency_name?: string;
  agency_city?: string;
  agency_barangay?: string;
  agency_street?: string;
  telephone?: string;
  email?: string;
  priority_areas?: string;
  stand_classification?: string;
  cooperating_agency_names?: string[];
  research_station?: string;
  classification_type?: string; 
  class_input?: string; 
  sector?: string;
  discipline?: string;
  duration?: number;
  planned_start_month?: string;
  planned_start_year?: string;
  planned_end_month?: string;
  planned_end_year?: string;
  budget_sources?: { source: string; ps: number; mooe: number; co: number; total: number }[];
}

export const SUPPORTED_TYPES = [
  "application/pdf",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
] as const;

/**
 * Extract plain text from a document buffer based on its MIME type.
 */
export async function extractTextFromFile(buffer: Buffer, contentType: string): Promise<string> {
  const t0 = Date.now();
  extractorLog(`Parsing buffer`, { contentType, bytes: buffer.length });
  switch (contentType) {
    case "application/pdf": {
      try {
        const pdfData = await pdf(buffer);
        const text = pdfData.text;
        const ms = Date.now() - t0;
        extractorLog(`PDF parse done`, { ms, chars: text?.length ?? 0 });
        extractorDebug("PDF text preview", { preview: truncateForLog(text || "", 400) });
        return text;
      } catch (err) {
        console.error("[Extractor] PDF Parse failed:", err);
        throw err;
      }
    }
    case "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
      try {
        const result = await mammoth.extractRawText({ buffer });
        const text = result.value;
        const ms = Date.now() - t0;
        extractorLog(`DOCX parse done`, { ms, chars: text?.length ?? 0 });
        extractorDebug("DOCX text preview", { preview: truncateForLog(text || "", 400) });
        return text;
      } catch (err) {
        console.error("[Extractor] DOCX Parse failed:", err);
        throw err;
      }
    }
    case "application/msword": {
      try {
        const raw = await officeParser.parseOfficeAsync(buffer);
        const text = typeof raw === "string" ? raw : String(raw);
        const ms = Date.now() - t0;
        extractorLog(`DOC parse done`, { ms, chars: text.length });
        extractorDebug("DOC text preview", { preview: truncateForLog(text, 400) });
        return text;
      } catch (err) {
        console.error("[Extractor] DOC Parse failed:", err);
        throw err;
      }
    }
    default:
      console.warn(`[Extractor] Unsupported format: ${contentType}`);
      throw new Error(`Unsupported file format: ${contentType}`);
  }
}

/**
 * Extract proposal metadata from raw PDF text.
 */
export function extractDataFromText(text: string): ExtractedData {
  const data: ExtractedData = {
    title: "Unknown Project",
    duration: 12,
    cooperating_agencies: 0,
    total: 0,
    mooe: 0,
    ps: 0,
    co: 0,
  };

  const titleMatch = text.match(/Project\s+Title[:\s]*(.+)/i);
  if (titleMatch) {
    let title = titleMatch[1].trim();
    const matchEnd = (titleMatch.index ?? 0) + titleMatch[0].length;
    const restText = text.substring(matchEnd);
    const nextLineMatch = restText.match(/^\n([^\n]+)/);
    if (nextLineMatch) {
      const nextLine = nextLineMatch[1].trim();
      if (nextLine && !/^(Leader|Agency|Address|Telephone|Fax|Email|Program)/i.test(nextLine)) {
        title = title + " " + nextLine;
      }
    }
    data.title = title.replace(/\s{2,}/g, " ").trim();
  }

  const monthLabelMatch = text.match(/\(In months\)\s*(\d+)/i);
  const durationLabelMatch = text.match(/Duration[:\s]+(\d+)/i);

  if (monthLabelMatch) {
    data.duration = parseInt(monthLabelMatch[1], 10);
  } else if (durationLabelMatch) {
    const val = parseInt(durationLabelMatch[1], 10);
    if (val < 120) {
      data.duration = val;
    }
  }

  const agencySection = text.match(/Cooperating Agencies[^\n]*\n([\s\S]*?)(?=\n\(\d\)|$|Classification)/i);
  if (agencySection) {
    const rawAgencies = agencySection[1].trim();
    if (rawAgencies.length > 3 && !rawAgencies.includes("N/A")) {
      const count = (rawAgencies.match(/,/g) || []).length + 1;
      data.cooperating_agencies = count;
    }
  }

  const numbers = text.match(/([\d,]+\.\d{2})/g);
  if (numbers && numbers.length > 0) {
    const cleanNums: number[] = [];
    for (const n of numbers) {
      try {
        const val = parseFloat(n.replace(/,/g, ""));
        if (!isNaN(val)) cleanNums.push(val);
      } catch {
        // skip
      }
    }

    if (cleanNums.length > 0) {
      data.total = Math.max(...cleanNums);

      if (text.includes("PS")) {
        const psMatch = text.match(/PS.*?([\d,]+\.\d{2})/s);
        if (psMatch) {
          const val = parseFloat(psMatch[1].replace(/,/g, ""));
          if (val < data.total) data.ps = val;
        }
      }

      if (text.includes("MOOE")) {
        const mooeMatch = text.match(/MOOE.*?([\d,]+\.\d{2})/s);
        if (mooeMatch) {
          const val = parseFloat(mooeMatch[1].replace(/,/g, ""));
          if (val < data.total) data.mooe = val;
        }
      }

      if (data.co === 0 && data.total > 0) {
        const remainder = data.total - (data.ps + data.mooe);
        if (remainder > 0) data.co = remainder;
      }
    }
  }

  extractorLog(`Metadata (for AI analyzer)`, {
    titleLen: data.title.length,
    duration: data.duration,
    cooperating_agencies: data.cooperating_agencies,
    total: data.total,
    ps: data.ps,
    mooe: data.mooe,
    co: data.co,
  });
  extractorDebug("Project title preview", { title: truncateForLog(data.title, 160) });

  return data;
}

/**
 * Many PDFs export as one long line or reorder text — patterns anchored with ^ miss.
 * Grab Agency/Address … up to Telephone/Fax/Email from the raw string (no line anchors).
 */
function extractAgencyAddressBlobFromFullText(fullText: string): string | undefined {
  const patterns: RegExp[] = [
    /Agency\s*\/\s*Agency\s+Address\s*[:\s]*\s*([\s\S]+?)(?=\s*(?:Telephone\s*\/\s*Fax\s*\/\s*Email|Telephone\/Fax\/Email|Leader\b|Program\b|Project\b|Gender\b))/i,
    /Agency\s*\/\s*Address\s*[:\s]*\s*([\s\S]+?)(?=\s*(?:Telephone\s*\/\s*Fax\s*\/\s*Email|Telephone\/Fax\/Email|Leader\b|Program\b|Project\b|Gender\b))/i,
    /Implementing\s+Agency\s*[:\s]*\s*([\s\S]+?)(?=\s*(?:Address|Telephone|Leader\b))/i,
    /Agency\s*\/\s*(?:Agency\s+)?Address\s*[:\s]*\s*([\s\S]+?)(?=\s*Leader\s*\/\s*Gender\b)/i,
    /Agency\s*\/\s*(?:Agency\s+)?Address\s*[:\s]*\s*([\s\S]+?)(?=\s*\(\d+\)\s*[^\s])/i,
    /Implementing\s+Agency\s*\/\s*Address\s*[:\s]*\s*([\s\S]+?)(?=\s*(?:Telephone|Leader\b))/i,
    /Implementing\s+Agency\b[\s\S]*?Address\s*[:\s]*\s*([\s\S]+?)(?=\s*(?:Telephone|Leader\b))/i,
  ];
  for (const p of patterns) {
    const m = fullText.match(p);
    if (m?.[1]) {
      const v = m[1].replace(/\s{2,}/g, " ").trim();
      if (v.length > 2 && !/^Telephone/i.test(v)) return v;
    }
  }
  return undefined;
}

/** PDFs often repeat the section header; pick the first value that looks like real contact info. */
function pickBestTelephoneFaxEmailValue(
  fullText: string,
  cleanFn: (s: string) => string,
  isGarbage: (s: string) => boolean
): string | undefined {
  const re = /Telephone\s*\/\s*Fax\s*\/\s*Email\s*[:\s]*\s*([^\n\r]{1,1200})/gi;
  const candidates: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(fullText)) !== null) {
    candidates.push(cleanFn(m[1]));
  }
  const withEmail = candidates.find((c) => /[\w.+-]+@[\w.-]+\.\w+/.test(c) && !isGarbage(c));
  if (withEmail) return withEmail;
  const withPhone = candidates.find((c) => /[\d()+\-\s]{7,}/.test(c) && !isGarbage(c));
  if (withPhone) return withPhone;
  return candidates.find((c) => c.length > 2 && !isGarbage(c) && !/^N\/?A$/i.test(c));
}

/**
 * Extract form-fillable fields from the DOST Capsule Proposal template text.
 */
export function extractFormFields(text: string): FormExtractedFields {
  const fields: FormExtractedFields = {};
  const clean = (s: string) => s.replace(/\s{2,}/g, " ").trim();
  const textForScan = text.replace(/\u00a0/g, " ");
  const lines = textForScan
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  /** PDFs often merge the next block label into the same line as agency or contact. */
  const stripAfterContactLabel = (s: string): string =>
    clean(s.replace(/\bTelephone\s*\/\s*Fax\s*\/\s*Email\b[\s\S]*/i, ""));

  const isGarbageContactValue = (s: string): boolean =>
    !s ||
    /^Telephone\s*\/\s*Fax\s*\/\s*Email$/i.test(s) ||
    /Program\s+Title/i.test(s) ||
    /Project\s+Title/i.test(s) ||
    /^Leader\s*\/\s*Gender/i.test(s);

  const isGarbageAddressSegment = (s: string): boolean =>
    !s ||
    /^Telephone/i.test(s) ||
    /Fax\s*\/\s*Email/i.test(s) ||
    /Program\s+Title/i.test(s) ||
    /Project\s+Title/i.test(s);

  const readLabeledValue = (labelPatterns: RegExp[], stopPatterns: RegExp[] = []): string | undefined => {
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      for (const labelPattern of labelPatterns) {
        const match = line.match(labelPattern);
        if (!match) continue;

        let value = clean(match[1] || "");
        
        // If value is empty on this line, try the next line
        if (!value && i + 1 < lines.length) {
          const next = lines[i + 1];
          // Ensure next line isn't another label
          if (!/^(Program|Project|Leader|Agency|Address|Telephone|Fax|Email|Priority|Sector|Discipline|Duration|Classification)/i.test(next)) {
            value = clean(next);
          }
        }

        if (!value) continue;

        // Trim value when a known stop marker starts.
        for (const stop of stopPatterns) {
          const stopMatch = value.match(stop);
          if (stopMatch && typeof stopMatch.index === "number") {
            value = clean(value.slice(0, stopMatch.index));
          }
        }

        return value || undefined;
      }
    }
    return undefined;
  };

  const programMatch = textForScan.match(/Program\s+Title[:\s]*(.+)/i);
  if (programMatch) {
    const val = clean(programMatch[1]);
    if (val && !/^N\/?A$/i.test(val)) fields.program_title = val;
  }

  const projectMatch = textForScan.match(/Project\s+Title[:\s]*(.+)/i);
  if (projectMatch) {
    let title = projectMatch[1].trim();
    const matchEnd = (projectMatch.index ?? 0) + projectMatch[0].length;
    const rest = textForScan.substring(matchEnd);
    const nextLine = rest.match(/^\n([^\n]+)/);
    if (nextLine) {
      const nl = nextLine[1].trim();
      if (nl && !/^(Leader|Agency|Address|Telephone|Fax|Email|Program|\(\d)/i.test(nl)) {
        title = title + " " + nl;
      }
    }
    fields.project_title = clean(title);
  }

  const yearLabelMatch = textForScan.match(/\bYear[:\s]*([12]\d{3})\b/i);
  if (yearLabelMatch) {
    fields.year = yearLabelMatch[1].trim();
  }

  const applyAgencyFromRaw = (rawInput: string) => {
    const raw = stripAfterContactLabel(clean(rawInput));
    if (!raw) return;
    const parts = raw.split(/\s*\/\s*/).map((p) => p.trim()).filter(Boolean);
    if (parts.length >= 2) {
      const name = parts[0].trim();
      if (name) fields.agency_name = name;
      const addrParts = parts
        .slice(1)
        .join("/")
        .split(",")
        .map((s) => s.trim())
        .filter((p) => p && !isGarbageAddressSegment(p));
      if (addrParts.length >= 3) {
        fields.agency_street = addrParts[0];
        fields.agency_barangay = addrParts[1];
        fields.agency_city = addrParts[addrParts.length - 1];
      } else if (addrParts.length === 2) {
        fields.agency_barangay = addrParts[0];
        fields.agency_city = addrParts[1];
      } else if (addrParts.length === 1) {
        fields.agency_city = addrParts[0];
      }
    } else {
      fields.agency_name = raw;
    }
  };

  const agencyBlob = extractAgencyAddressBlobFromFullText(textForScan);
  if (agencyBlob) {
    applyAgencyFromRaw(agencyBlob);
  }
  if (!fields.agency_name) {
    const agencyName = readLabeledValue(
      [/^Implementing\s+Agency[:\s]*(.*)$/i, /^Agency\s+Name[:\s]*(.*)$/i, /^Agency[:\s]*(.*)$/i],
      [/Address/i, /Telephone/i, /Program/i, /Project/i]
    );
    if (agencyName) fields.agency_name = agencyName;
  }

  if (!fields.agency_city && !fields.agency_street) {
    const agencyAddress = readLabeledValue(
      [
        /^Agency\s*\/\s*Agency\s+Address[:\s]*(.*)$/i,
        /^Agency\s*\/\s*Address[:\s]*(.*)$/i,
        /^Agency\s+Address[:\s]*(.*)$/i,
        /^Address[:\s]*(.*)$/i,
      ],
      [/Telephone\s*\/\s*Fax\s*\/\s*Email/i, /Program\s+Title\s*:/i, /Project\s+Title\s*:/i, /Leader/i]
    );
    if (agencyAddress) applyAgencyFromRaw(agencyAddress);
  }

  let contactLine = pickBestTelephoneFaxEmailValue(textForScan, clean, isGarbageContactValue);
  if (!contactLine) {
    contactLine = readLabeledValue(
      [/^Telephone\s*\/\s*Fax\s*\/\s*Email[:\s]*(.*)$/i],
      [/Program\s+Title\s*:/i, /Project\s+Title\s*:/i, /Leader\s*\/\s*Gender\s*:/i, /Agency\s*\/\s*Address\s*:/i]
    );
  }
  if (!contactLine) {
    const loose = textForScan.match(/Telephone\s*\/\s*Fax\s*\/\s*Email[:\s]*([^\n\r]+)/i);
    if (loose?.[1]) contactLine = clean(loose[1]);
  }
  if (contactLine) {
    const raw = clean(contactLine);
    if (!isGarbageContactValue(raw)) {
      const emailInLine = raw.match(/[\w.+-]+@[\w.-]+\.\w+/);
      if (emailInLine) fields.email = emailInLine[0];
      const phoneInLine = raw.match(/[\d()+\-\s]{7,}/);
      if (phoneInLine) fields.telephone = phoneInLine[0].trim();
      if (!fields.email && !fields.telephone && raw.length > 3 && !isGarbageContactValue(raw)) {
        fields.telephone = raw;
      }
    }
  }

  // Fallback for PDFs where contact labels are malformed or split.
  if (!fields.email) {
    const emailFallback = textForScan.match(/[\w.+-]+@[\w.-]+\.\w+/);
    if (emailFallback) fields.email = emailFallback[0];
  }

  if (fields.telephone && isGarbageContactValue(fields.telephone)) {
    delete fields.telephone;
  }
  if (fields.agency_city && isGarbageAddressSegment(fields.agency_city)) {
    delete fields.agency_city;
  }
  if (fields.agency_street && isGarbageAddressSegment(fields.agency_street)) {
    delete fields.agency_street;
  }
  if (fields.agency_barangay && isGarbageAddressSegment(fields.agency_barangay)) {
    delete fields.agency_barangay;
  }
  if (fields.agency_name !== undefined && !String(fields.agency_name).trim()) {
    delete fields.agency_name;
  }

  // --- Priority Areas / STAND Classification ---
  // The PDF has a section header "(6) Priority Areas/STAND Classification" with
  // checkbox options on the FOLLOWING lines. We scan those lines for a filled
  // checkbox (non-underscore prefix like a flag icon or filled mark).
  {
    const sectionIdx = lines.findIndex(l => /Priority\s+Areas?\s*\/\s*STAND\s+Classification/i.test(l));
    if (sectionIdx !== -1) {
      // The known options in the DOST template
      const knownOptions = [
        "STAND",
        "Coconut Industry",
        "Export Winners",
        "Other Priority Areas",
        "Support Industries",
      ];
      const checkedOptions: string[] = [];
      // Scan the next 6 lines for any filled/checked option
      for (let i = sectionIdx + 1; i < Math.min(sectionIdx + 7, lines.length); i++) {
        const rawLine = lines[i];
        
        for (const opt of knownOptions) {
          // Find where the option text starts in the line
          const idx = rawLine.indexOf(opt);
          if (idx !== -1) {
            // Grab the ~12 characters right before the option
            const prefix = rawLine.substring(Math.max(0, idx - 12), idx);
            // Check if those characters contain a checkmark (x, X, ✓, ✔, or the unicode )
            // surrounded by spaces, brackets, or underscores
            const isChecked = /[\s_\[\]()]*[xX✓✔][\s_\[\]()]*$/.test(prefix);
            
            if (isChecked) {
              checkedOptions.push(opt);
            }
          }
        }
      }
      if (checkedOptions.length > 0) {
        const finalValue = checkedOptions.join(", ");
        fields.priority_areas = finalValue;
        fields.stand_classification = finalValue;
      }
    }
  }

  if (!fields.priority_areas && fields.sector) {
    fields.priority_areas = fields.sector;
  }
  if (!fields.stand_classification && fields.priority_areas) {
    fields.stand_classification = fields.priority_areas;
  }

  const coopMatch = textForScan.match(/Cooperating\s+Agenc(?:y|ies)[^\n]*\n?([\s\S]*?)(?=\n\s*\(\d\)|\n\s*R\s*&\s*D\s+Station|$)/i);
  if (coopMatch) {
    const raw = clean(coopMatch[1]);
    if (raw.length > 2 && !/^N\/?A$/i.test(raw)) {
      fields.cooperating_agency_names = raw.split(/,\s*/).map(s => s.trim()).filter(Boolean);
    }
  }

  const stationMatch = textForScan.match(/R\s*&?\s*D\s+Station[^\n]*\n?([\s\S]*?)(?=\n\s*\(\d\)|$)/i);
  if (stationMatch) {
    const val = clean(stationMatch[1]);
    if (val.length > 2) fields.research_station = val;
  }

  const classSection = textForScan.match(/Classification[^\n]*\n([\s\S]*?)(?=\n\s*\(\d\)\s*(?:Mode|Priority|Sector)|$)/i);
  if (classSection) {
    const classText = classSection[1];
    const hasBasic = /(?:_+|[xX✓✔])\s*Basic/i.test(classText);
    const hasApplied = /(?:_+|[xX✓✔])\s*Applied/i.test(classText);
    const hasPilot = /(?:_+|[xX✓✔])\s*Pilot/i.test(classText);
    const hasPromotion = /(?:_+|[xX✓✔])\s*(?:Tech|Promotion|Commercialization)/i.test(classText);

    if (hasBasic) { fields.classification_type = "research"; fields.class_input = "basic"; }
    else if (hasApplied) { fields.classification_type = "research"; fields.class_input = "applied"; }
    else if (hasPilot) { fields.classification_type = "development"; fields.class_input = "pilot_testing"; }
    else if (hasPromotion) { fields.classification_type = "development"; fields.class_input = "tech_promotion"; }
  }

  // Same-line PDF extracts often put the value after "Sector/Commodity:" on one line.
  let sectorVal = readLabeledValue(
    [/^Sector\s*\/\s*Commodity[:\s]*(.*)$/i],
    [/^\(\d+\)\s*/, /^Discipline\b/i, /\n\s*Discipline\b/i]
  );
  if (!sectorVal) {
    const sectorLoose = textForScan.match(/Sector\s*\/\s*Commodity[:\s]*([^\n\r]+)/i);
    if (sectorLoose?.[1]) sectorVal = clean(sectorLoose[1]);
  }
  if (sectorVal) {
    sectorVal = clean(sectorVal.split(/\bDiscipline\b/i)[0] ?? sectorVal);
    if (sectorVal.length > 2) fields.sector = sectorVal;
  }

  if (!fields.sector && fields.priority_areas) {
    fields.sector = fields.priority_areas;
  }

  let disciplineVal = readLabeledValue(
    [/^Discipline[:\s]*(.*)$/i],
    [/^\(\d+\)\s*/, /\n\s*\(\d+\)/]
  );
  if (!disciplineVal) {
    const discLoose = textForScan.match(/Discipline[:\s]*([^\n\r]+)/i);
    if (discLoose?.[1]) disciplineVal = clean(discLoose[1]);
  }
  if (disciplineVal && disciplineVal.length > 2) {
    fields.discipline = disciplineVal;
  }

  const monthsMatch = textForScan.match(/\(In\s+months\)\s*(\d+)/i);
  const durationAlt = textForScan.match(/Duration[:\s]*(\d+)/i);
  if (monthsMatch) {
    fields.duration = parseInt(monthsMatch[1], 10);
  } else if (durationAlt) {
    const val = parseInt(durationAlt[1], 10);
    if (val > 0 && val < 120) fields.duration = val;
  }

  const startMatch = textForScan.match(/Planned\s+[Ss]tart\s+[Dd]ate\s*[_\s]*([A-Za-z]+)[_\s]*(\d{4})/i);
  if (startMatch) {
    fields.planned_start_month = startMatch[1].trim();
    fields.planned_start_year = startMatch[2].trim();
  }

  const endMatch = textForScan.match(/Planned\s+(?:Completion|[Ee]nd)\s+[Dd]ate\s*[_\s]*([A-Za-z]+)[_\s]*(\d{4})/i);
  if (endMatch) {
    fields.planned_end_month = endMatch[1].trim();
    fields.planned_end_year = endMatch[2].trim();
  }

  if (!fields.year) {
    fields.year = new Date().getFullYear().toString();
  }

  const budgetSection = textForScan.match(/(?:Estimated\s+Budget|Source\s*\n?\s*Of\s+funds)([\s\S]*?)(?=Note:|$)/i);
  if (budgetSection) {
    const budgetText = budgetSection[1];
    const sources: FormExtractedFields["budget_sources"] = [];
    const lines = budgetText.split("\n").map(l => l.trim()).filter(Boolean);

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (/^(PS|MOOE|CO|TOTAL|Year|Source|Of\s+funds)/i.test(line)) continue;
      if (/^TOTAL\s*➔/i.test(line)) continue;

      const isSourceName = /^[A-Za-z]/.test(line) && !(/([\d,]+\.\d{2})/.test(line));
      if (isSourceName) {
        const sourceName = line.trim();
        const nextLine = i + 1 < lines.length ? lines[i + 1] : "";
        const nums = nextLine.match(/([\d,]+\.\d{2})/g);
        if (nums && nums.length >= 3) {
          const parsed = nums.map(n => parseFloat(n.replace(/,/g, "")));
          if (parsed.length >= 4) {
            sources.push({ source: sourceName, ps: parsed[0], mooe: parsed[1], co: parsed[2], total: parsed[3] });
          } else if (parsed.length === 3) {
            sources.push({ source: sourceName, ps: parsed[0], mooe: parsed[1], co: 0, total: parsed[2] });
          }
          i++; 
        }
      }
    }

    if (sources.length > 0) fields.budget_sources = sources;
  }

  const formKeys = Object.keys(fields) as (keyof FormExtractedFields)[];
  extractorLog(`Form autofill fields`, { count: formKeys.length, keys: formKeys });
  extractorDebug("Form field sample values", {
    sample: Object.fromEntries(
      formKeys.map((k) => {
        const v = fields[k];
        if (v === undefined) return [k, undefined];
        if (typeof v === "string") return [k, truncateForLog(v, 100)];
        if (Array.isArray(v)) return [k, v.length > 3 ? `[${v.length} items]` : v];
        return [k, v];
      })
    ),
  });

  return fields;
}

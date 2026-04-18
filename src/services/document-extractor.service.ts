import mammoth from "mammoth";
import { ExtractedData } from "./ai-analyzer.service";

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
  console.log(`[Extractor] Parsing file of type: ${contentType}, size: ${buffer.length} bytes`);
  switch (contentType) {
    case "application/pdf": {
      try {
        const pdfData = await pdf(buffer);
        console.log(`[Extractor] PDF parsed successfully, characters: ${pdfData.text?.length}`);
        return pdfData.text;
      } catch (err) {
        console.error("[Extractor] PDF Parse failed:", err);
        throw err;
      }
    }
    case "application/vnd.openxmlformats-officedocument.wordprocessingml.document": {
      try {
        const result = await mammoth.extractRawText({ buffer });
        console.log(`[Extractor] DOCX parsed successfully, characters: ${result.value?.length}`);
        return result.value;
      } catch (err) {
        console.error("[Extractor] DOCX Parse failed:", err);
        throw err;
      }
    }
    case "application/msword": {
      try {
        const text = await officeParser.parseOfficeAsync(buffer);
        console.log(`[Extractor] DOC parsed successfully`);
        return typeof text === "string" ? text : String(text);
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

  return data;
}

/**
 * Extract form-fillable fields from the DOST Capsule Proposal template text.
 */
export function extractFormFields(text: string): FormExtractedFields {
  const fields: FormExtractedFields = {};
  const clean = (s: string) => s.replace(/\s{2,}/g, " ").trim();
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const readLabeledValue = (labelPatterns: RegExp[], stopPatterns: RegExp[] = []): string | undefined => {
    for (const line of lines) {
      for (const labelPattern of labelPatterns) {
        const match = line.match(labelPattern);
        if (!match) continue;

        let value = clean(match[1] || "");
        if (!value) return undefined;

        // Some PDF extracts concatenate the next label into the same line.
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

  const programMatch = text.match(/Program\s+Title[:\s]*(.+)/i);
  if (programMatch) {
    const val = clean(programMatch[1]);
    if (val && !/^N\/?A$/i.test(val)) fields.program_title = val;
  }

  const projectMatch = text.match(/Project\s+Title[:\s]*(.+)/i);
  if (projectMatch) {
    let title = projectMatch[1].trim();
    const matchEnd = (projectMatch.index ?? 0) + projectMatch[0].length;
    const rest = text.substring(matchEnd);
    const nextLine = rest.match(/^\n([^\n]+)/);
    if (nextLine) {
      const nl = nextLine[1].trim();
      if (nl && !/^(Leader|Agency|Address|Telephone|Fax|Email|Program|\(\d)/i.test(nl)) {
        title = title + " " + nl;
      }
    }
    fields.project_title = clean(title);
  }

  const yearLabelMatch = text.match(/\bYear[:\s]*([12]\d{3})\b/i);
  if (yearLabelMatch) {
    fields.year = yearLabelMatch[1].trim();
  }

  const agencyAddress = readLabeledValue(
    [/^Agency\s*\/\s*Address[:\s]*(.*)$/i, /^Agency\s+Address[:\s]*(.*)$/i],
    [/Telephone\s*\/\s*Fax\s*\/\s*Email/i, /Program\s+Title\s*:/i, /Project\s+Title\s*:/i]
  );
  if (agencyAddress) {
    const raw = clean(agencyAddress);
    const parts = raw.split(/\s*\/\s*/);
    if (parts.length >= 2) {
      fields.agency_name = parts[0].trim();
      const addrParts = parts.slice(1).join("/").split(",").map(s => s.trim());
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
  }

  const contactLine = readLabeledValue(
    [/^Telephone\s*\/\s*Fax\s*\/\s*Email[:\s]*(.*)$/i, /^Telephone[:\s]*(.*)$/i],
    [/Program\s+Title\s*:/i, /Project\s+Title\s*:/i, /Leader\s*\/\s*Gender\s*:/i, /Agency\s*\/\s*Address\s*:/i]
  );
  if (contactLine) {
    const raw = clean(contactLine);
    const emailInLine = raw.match(/[\w.+-]+@[\w.-]+\.\w+/);
    if (emailInLine) fields.email = emailInLine[0];
    const phoneInLine = raw.match(/[\d()+\-\s]{7,}/);
    if (phoneInLine) fields.telephone = phoneInLine[0].trim();
    if (!fields.email && !fields.telephone && raw.length > 3) {
      fields.telephone = raw;
    }
  }

  // Fallback for PDFs where contact labels are malformed or split.
  if (!fields.email) {
    const emailFallback = text.match(/[\w.+-]+@[\w.-]+\.\w+/);
    if (emailFallback) fields.email = emailFallback[0];
  }

  if (fields.telephone && /Program\s+Title/i.test(fields.telephone)) {
    fields.telephone = undefined;
  }

  const priorityAreas = readLabeledValue(
    [/^Priority\s+Areas?[:\s]*(.*)$/i, /^Priority\s+Area[:\s]*(.*)$/i],
    [/Sector\s*\/\s*Commodity/i, /Discipline/i, /\(\d\)/i]
  );
  if (priorityAreas && !/^N\/?A$/i.test(priorityAreas)) {
    fields.priority_areas = priorityAreas;
  }

  const coopMatch = text.match(/Cooperating\s+Agenc(?:y|ies)[^\n]*\n?([\s\S]*?)(?=\n\s*\(\d\)|\n\s*R\s*&\s*D\s+Station|$)/i);
  if (coopMatch) {
    const raw = clean(coopMatch[1]);
    if (raw.length > 2 && !/^N\/?A$/i.test(raw)) {
      fields.cooperating_agency_names = raw.split(/,\s*/).map(s => s.trim()).filter(Boolean);
    }
  }

  const stationMatch = text.match(/R\s*&?\s*D\s+Station[^\n]*\n?([\s\S]*?)(?=\n\s*\(\d\)|$)/i);
  if (stationMatch) {
    const val = clean(stationMatch[1]);
    if (val.length > 2) fields.research_station = val;
  }

  const classSection = text.match(/Classification[^\n]*\n([\s\S]*?)(?=\n\s*\(\d\)\s*(?:Mode|Priority|Sector)|$)/i);
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

  const sectorMatch = text.match(/Sector\/Commodity[^\n]*\n?([\s\S]*?)(?=\n\s*\(\d\)|$)/i);
  if (sectorMatch) {
    const val = clean(sectorMatch[1]);
    if (val.length > 2) fields.sector = val;
  }

  if (!fields.sector && fields.priority_areas) {
    fields.sector = fields.priority_areas;
  }

  const discMatch = text.match(/Discipline[^\n]*\n?([\s\S]*?)(?=\n\s*\(\d\)|$)/i);
  if (discMatch) {
    const val = clean(discMatch[1]);
    if (val.length > 2) fields.discipline = val;
  }

  const monthsMatch = text.match(/\(In\s+months\)\s*(\d+)/i);
  const durationAlt = text.match(/Duration[:\s]*(\d+)/i);
  if (monthsMatch) {
    fields.duration = parseInt(monthsMatch[1], 10);
  } else if (durationAlt) {
    const val = parseInt(durationAlt[1], 10);
    if (val > 0 && val < 120) fields.duration = val;
  }

  const startMatch = text.match(/Planned\s+[Ss]tart\s+[Dd]ate\s*[_\s]*([A-Za-z]+)[_\s]*(\d{4})/i);
  if (startMatch) {
    fields.planned_start_month = startMatch[1].trim();
    fields.planned_start_year = startMatch[2].trim();
  }

  const endMatch = text.match(/Planned\s+(?:Completion|[Ee]nd)\s+[Dd]ate\s*[_\s]*([A-Za-z]+)[_\s]*(\d{4})/i);
  if (endMatch) {
    fields.planned_end_month = endMatch[1].trim();
    fields.planned_end_year = endMatch[2].trim();
  }

  if (!fields.year) {
    fields.year = fields.planned_start_year || fields.planned_end_year;
  }

  const budgetSection = text.match(/(?:Estimated\s+Budget|Source\s*\n?\s*Of\s+funds)([\s\S]*?)(?=Note:|$)/i);
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

  return fields;
}

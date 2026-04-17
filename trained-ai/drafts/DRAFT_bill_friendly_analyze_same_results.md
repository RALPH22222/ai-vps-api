# Draft: Cheaper Lambda, same UI numbers as `test_ai.py` (for later)

You said you are **not** adopting this yet. This file is a **blueprint** you can implement when ready.

**Goal:** Keep **MiniLM 384-d embeddings + cosine vs `comparison_db.json` + Keras JSON forward pass** so the website matches `test_ai.py` in practice, but **remove `@huggingface/transformers` from Lambda** so the function is smaller, faster, and cheaper.

**Idea:** The **browser** runs `Xenova/all-MiniLM-L6-v2` (same family as Python `all-MiniLM-L6-v2`). It sends the **384-float title embedding** plus the same **budget/meta** the server already extracts—or you send the file and let the server extract text only (no embedding on server).

Two variants:

| Variant | Who embeds title | Lambda does |
|--------|------------------|-------------|
| **A (recommended draft)** | Browser | Text extract + parse meta from uploaded file **or** accept pre-parsed meta + embedding in JSON |
| **B** | Lambda (current) | Everything (what you have now) |

Below is **Variant A**.

---

## 1. API contract (draft)

**Option A1 — Multipart (minimal change to current flow)**  
Keep `POST /proposal/analyze` as multipart with `file`, **add** a field `titleEmbedding` = JSON string of `number[]` (length **384**).

**Option A2 — JSON body**  
`POST /proposal/analyze-embedded` with:

```json
{
  "titleEmbedding": [ /* 384 floats */ ],
  "extracted": {
    "title": "string",
    "duration": 12,
    "cooperating_agencies": 0,
    "total": 0,
    "mooe": 0,
    "ps": 0,
    "co": 0
  }
}
```

If you still want **auto-fill from PDF**, you can keep sending the **file** in multipart **and** `titleEmbedding` from the client (client embeds the **same** title string it shows after extraction—or you embed after server returns parsed title in two steps; simplest is **one request** with file + embedding where embedding is from the **user-confirmed** title in the form).

Simplest UX draft: user clicks “Analyze”; frontend already has `project_title` from the form → **embed that string** → send multipart: `file` + `titleEmbedding`.

---

## 2. Frontend draft (React + Vite)

Install when you implement:

```bash
cd frontend
npm install @huggingface/transformers
```

**Sketch: lazy-load pipeline once, embed title, call existing API.**

```typescript
// drafts/embedTitle.ts  (new file when you implement)

import { pipeline, env } from "@huggingface/transformers";

// Optional: use local WASM cache in browser
// env.allowLocalModels = true;

let extractor: Awaited<ReturnType<typeof pipeline>> | null = null;

export async function embedProposalTitle(title: string): Promise<number[]> {
  if (!extractor) {
    extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  const t = title.trim();
  if (!t) throw new Error("Empty title");
  const tensor = await extractor(t, { pooling: "mean", normalize: false });
  const vec = Array.from(tensor.data as Float32Array);
  if (vec.length !== 384) {
    throw new Error(`Expected 384-d embedding, got ${vec.length}`);
  }
  return vec;
}
```

**Sketch: call analyze API with `FormData`**

```typescript
// When calling analyzeProposalWithAI(file) — extend to pass embedding from form title

import { embedProposalTitle } from "./embedTitle";

async function analyzeWithClientEmbedding(file: File, formProjectTitle: string) {
  const titleEmbedding = await embedProposalTitle(formProjectTitle);
  const fd = new FormData();
  fd.append("file", file);
  fd.append("titleEmbedding", JSON.stringify(titleEmbedding));
  // ... same fetch URL / auth headers as today ...
}
```

**Note:** First run downloads the ONNX model to the user’s browser (one-time per cache). That **shifts cost off AWS** to the user’s device.

**Trust:** A user could tamper with `titleEmbedding`. For a closed university system you may accept that; otherwise keep server-side embedding.

---

## 3. Backend: `analyze-proposal.ts` (draft snippet)

After multipart parse, read optional embedding:

```typescript
// Inside handler, after multipart.parse
const rawEmb = payload.titleEmbedding;
let clientTitleEmbedding: number[] | undefined;
if (rawEmb && typeof rawEmb === "string") {
  try {
    const parsed = JSON.parse(rawEmb) as number[];
    if (Array.isArray(parsed) && parsed.length === 384) {
      clientTitleEmbedding = parsed.map(Number);
    }
  } catch {
    /* ignore */
  }
}

const extracted = extractDataFromText(text);

// Prefer client embedding for semantic path; fallback to server if missing
const result = clientTitleEmbedding?.length === 384
  ? await analyzeProposalWithEmbedding(extracted, clientTitleEmbedding)
  : await analyzeProposal(extracted);
```

---

## 4. Backend: `ai-analyzer.service.ts` (draft new export)

Add a function that **skips** `encodeTitleMiniLM` and uses the passed vector:

```typescript
/**
 * Same math as analyzeProposal() but title embedding computed client-side (bill-friendly Lambda).
 */
export async function analyzeProposalWithEmbedding(
  extracted: ExtractedData,
  titleEmbedding384: number[]
): Promise<AnalysisResult> {
  if (titleEmbedding384.length !== 384) {
    throw new Error("titleEmbedding384 must be length 384");
  }
  // ... copy the same validation branches as analyzeProposal for unknown/short title ...

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

  const titleVec = titleEmbedding384;
  const score = Math.round(predict(titleVec, scaledMeta));
  const { noveltyScore, bestMatchTitle, bestMatchSimilarity } =
    checkUniquenessSemantic(titleVec);

  // ... same issues/suggestions/return object as analyzeProposal ...
}
```

Refactor tip: extract shared `buildAnalysisResult(extracted, titleVec)` used by both `analyzeProposal` and `analyzeProposalWithEmbedding`.

---

## 5. CDK draft (when you drop server Transformers)

When **all** clients send `titleEmbedding` (or you accept fallback only for admins):

- Remove `nodeModules: ["@huggingface/transformers"]` from `analyze-proposal` bundling.
- Remove onnx/sharp shim hooks **if** nothing else in that bundle needs them.
- Try **512 MB** memory + **30–60 s** timeout after profiling.

```typescript
// proposal-lambdas.ts — draft
this.analyzeProposal = new NodejsFunction(this, "analyze-proposal", {
  ...defaults,
  functionName: "pms-analyze-proposal",
  memorySize: 512,
  timeout: Duration.seconds(60),
  entry: path.resolve("src", "handlers", "proposal", "analyze-proposal.ts"),
  role: sharedRole,
  environment: sharedEnv,
  bundling: {
    commandHooks: {
      afterBundling(inputDir, outputDir) {
        return process.platform === "win32"
          ? [`xcopy "${inputDir}\\src\\ai-models" "${outputDir}\\ai-models" /E /I /Y`]
          : [`cp -r ${inputDir}/src/ai-models ${outputDir}/ai-models`];
      },
    },
  },
});
```

---

## 6. Optional: Python sidecar draft (exact `test_ai.py`)

If you want **identical** PyTorch numbers:

1. Small **FastAPI** app with `SentenceTransformer`, `tensorflow` or `keras` load `proposal_model.keras`, same `parse_data_from_text` logic.
2. Expose `POST /analyze` with file upload.
3. Frontend or Lambda **HTTP** calls that URL with `Authorization` secret.

Pseudo-structure:

```
trained-ai/
  sidecar/
    main.py        # FastAPI, mount test_ai logic
    requirements.txt
Dockerfile
```

You would copy functions from `test_ai.py` into importable modules instead of `input()`.

---

## 7. Checklist when you implement

- [ ] Frontend: embed **same string** you use as “project title” for analysis (aligned with extracted PDF title if you show both).
- [ ] Use `pooling: "mean", normalize: false` to stay closest to default `SentenceTransformer.encode` behavior.
- [ ] Keep **`comparison_db.json`** (384-d vectors) and **`dense_layers.json`** in `ai-models` in sync via `export_weights.py`.
- [ ] Load-test Lambda **memory** with full `comparison_db.json` parse (still large JSON).
- [ ] Decide policy if `titleEmbedding` missing: reject, or fallback to current server-side MiniLM.

---

*End of draft — safe to delete this file if you do not plan to use it.*

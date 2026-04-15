/**
 * JSON Data Agent — RAG Server
 * -----------------------------
 * Full Retrieval-Augmented Generation pipeline:
 *   1. Reads JSON files from ./data on startup + every 5 mins
 *   2. Generates embeddings via HuggingFace (free)
 *   3. Stores vectors in Qdrant Cloud (free)
 *   4. On each user query: embeds question → searches Qdrant → top 5 results to Groq
 *
 * ENV VARS (set in Railway dashboard):
 *   GROQ_API_KEY        — groq.com API key
 *   QDRANT_URL          — e.g. https://xxxx.us-east4-0.gcp.cloud.qdrant.io
 *   QDRANT_API_KEY      — Qdrant Cloud API key
 *   HF_API_KEY          — HuggingFace token (free, for embeddings)
 *   PORT                — set automatically by Railway
 */

const http   = require("http");
const https  = require("https");
const fs     = require("fs");
const path   = require("path");
const url    = require("url");

const PORT           = process.env.PORT || 3456;
const WATCH_DIR      = path.resolve(process.env.DATA_DIR || "./data");
const POLL_INTERVAL  = 5 * 60 * 1000;
const GROQ_API_KEY   = process.env.GROQ_API_KEY   || "";
const GROQ_MODEL     = "llama-3.3-70b-versatile";
const QDRANT_URL     = (process.env.QDRANT_URL     || "").replace(/\/$/, "");
const QDRANT_API_KEY = process.env.QDRANT_API_KEY  || "";
const HF_API_KEY     = process.env.HF_API_KEY      || "";
const COLLECTION     = "articles";
const EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2";
const TOP_K          = 5;

let indexedFiles  = {};
let totalArticles = 0;
let lastIndexed   = null;
let qdrantReady   = false;

function log(msg) { console.log(`[${new Date().toISOString()}] ${msg}`); }

// Generic HTTPS helper
function httpsRequest(options, body) {
  return new Promise((resolve, reject) => {
    const req = https.request(options, res => {
      let data = "";
      res.on("data", c => data += c);
      res.on("end", () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(data) }); }
        catch { resolve({ status: res.statusCode, body: data }); }
      });
    });
    req.on("error", reject);
    if (body) req.write(typeof body === "string" ? body : JSON.stringify(body));
    req.end();
  });
}

function qdrantHost() {
  try { return new URL(QDRANT_URL).hostname; } catch { return ""; }
}

// HuggingFace embeddings
async function embed(texts) {
  const payload = JSON.stringify({ inputs: texts });
  const res = await httpsRequest({
    hostname: "router.huggingface.co",
    path: `/hf-inference/models/${EMBED_MODEL}/pipeline/feature-extraction`,
    method: "POST",
    headers: {
      "Authorization": `Bearer ${HF_API_KEY}`,
      "Content-Type": "application/json",
      "x-wait-for-model": "true",
      "Content-Length": Buffer.byteLength(payload)
    }
  }, payload);
  if (res.status !== 200) throw new Error(`HuggingFace error ${res.status}: ${JSON.stringify(res.body)}`);
  const raw = res.body;

  // HF can return several shapes — normalize everything to: Array of flat float arrays
  // Shape A: [ [f,f,f,...], [f,f,f,...] ]  — batch of vectors (what we want)
  // Shape B: [ f, f, f, ... ]              — single bare vector
  // Shape C: [ [[f,f,...]], [[f,f,...]] ]   — batch wrapped in extra dim (token-level)
  // Shape D: [ [ [f,f,...] ] ]             — single wrapped in two extra dims

  function flatten(v) {
    // Recursively unwrap until we get a flat number array
    while (Array.isArray(v) && Array.isArray(v[0])) v = v[0];
    return v;
  }

  // If it's a batch (array of items)
  if (Array.isArray(raw)) {
    if (typeof raw[0] === "number") {
      // bare single vector [f,f,f,...]
      return [raw];
    }
    if (Array.isArray(raw[0])) {
      if (typeof raw[0][0] === "number") {
        // [[f,f,...],[f,f,...]] — proper batch
        return raw;
      }
      // Nested deeper — flatten each item
      return raw.map(item => flatten(item));
    }
  }
  return [flatten(raw)];
}

// Qdrant helpers
async function qdrantReq(method, endpoint, body) {
  const payload = body ? JSON.stringify(body) : null;
  return httpsRequest({
    hostname: qdrantHost(),
    path: endpoint,
    method,
    headers: {
      "api-key": QDRANT_API_KEY,
      "Content-Type": "application/json",
      ...(payload ? { "Content-Length": Buffer.byteLength(payload) } : {})
    }
  }, payload);
}

async function ensureCollection() {
  // Delete and recreate to ensure correct vector dimensions
  await qdrantReq("DELETE", `/collections/${COLLECTION}`);
  log(`Dropped old collection "${COLLECTION}" (if existed)`);

  // Probe actual embedding size with a test string
  const testVecs = await embed(["test"]);
  const dim = testVecs[0].length;
  log(`Detected embedding dimension: ${dim} (vec sample: ${JSON.stringify(testVecs[0].slice(0,3))}...)`);
  if (dim < 10) throw new Error(`Embedding dim too small (${dim}) — check HF response shape`);

  const res = await qdrantReq("PUT", `/collections/${COLLECTION}`, {
    vectors: { size: dim, distance: "Cosine" }
  });
  if (res.status === 200 || res.status === 201) log(`Collection "${COLLECTION}" created (dim=${dim})`);
  else throw new Error(`Failed to create collection: ${JSON.stringify(res.body)}`);
}

async function upsertPoints(points) {
  const res = await qdrantReq("PUT", `/collections/${COLLECTION}/points?wait=true`, { points });
  if (res.status !== 200) throw new Error(`Upsert failed: ${JSON.stringify(res.body)}`);
}

async function searchQdrant(vec) {
  const res = await qdrantReq("POST", `/collections/${COLLECTION}/points/search`, {
    vector: vec, limit: TOP_K, with_payload: true
  });
  if (res.status !== 200) throw new Error(`Search failed: ${JSON.stringify(res.body)}`);
  return res.body.result || [];
}

async function deleteByFile(file) {
  await qdrantReq("POST", `/collections/${COLLECTION}/points/delete`, {
    filter: { must: [{ key: "source_file", match: { value: file } }] }
  });
}

function articleToText(a) {
  return [a.title, a.content, (a.market_categories||[]).join(", "), (a.other_labels||[]).join(", ")]
    .filter(Boolean).join(" | ").slice(0, 1000);
}

function makeId(filename, idx) {
  let hash = 0;
  const str = filename + "_" + idx;
  for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
  return Math.abs(hash);
}

async function indexFile(file) {
  const filePath = path.join(WATCH_DIR, file);
  let data;
  try { data = JSON.parse(fs.readFileSync(filePath, "utf8")); }
  catch (e) { log(`Skipped ${file}: ${e.message}`); return 0; }

  const articles = Array.isArray(data) ? data : [data];
  try { await deleteByFile(file); } catch (e) { log(`Warning deleting old ${file}: ${e.message}`); }

  const BATCH = 32;
  let indexed = 0;
  for (let i = 0; i < articles.length; i += BATCH) {
    const batch = articles.slice(i, i + BATCH);
    let vectors;
    try { vectors = await embed(batch.map(articleToText)); }
    catch (e) { log(`Embed error ${file}[${i}]: ${e.message}`); continue; }

    const points = batch.map((a, j) => ({
      id: makeId(file, i + j),
      vector: vectors[j],
      payload: {
        source_file: file,
        title:             a.title             || "",
        content:           a.content           || "",
        date:              a.date              || "",
        author:            a.author            || "",
        market_categories: a.market_categories || [],
        other_labels:      a.other_labels      || [],
        post_link:         a.post_link         || "",
        source:            a.source            || ""
      }
    }));

    try { await upsertPoints(points); indexed += points.length; }
    catch (e) { log(`Upsert error ${file}[${i}]: ${e.message}`); }
  }
  log(`Indexed: ${file} → ${indexed} articles`);
  return indexed;
}

async function indexFolder() {
  if (!qdrantReady) return;
  if (!fs.existsSync(WATCH_DIR)) { log(`Folder not found: ${WATCH_DIR}`); return; }
  let files;
  try { files = fs.readdirSync(WATCH_DIR).filter(f => f.endsWith(".json")); }
  catch (e) { log(`Read error: ${e.message}`); return; }

  let changed = false;
  for (const file of files) {
    let stat;
    try { stat = fs.statSync(path.join(WATCH_DIR, file)); } catch { continue; }
    if (indexedFiles[file]?.updatedAt === stat.mtimeMs) continue;
    const count = await indexFile(file);
    indexedFiles[file] = { updatedAt: stat.mtimeMs, count };
    changed = true;
  }
  for (const file of Object.keys(indexedFiles)) {
    if (!files.includes(file)) {
      try { await deleteByFile(file); } catch (e) { log(`Delete error ${file}: ${e.message}`); }
      delete indexedFiles[file];
      changed = true;
    }
  }
  totalArticles = Object.values(indexedFiles).reduce((a, v) => a + (v.count || 0), 0);
  if (changed) lastIndexed = Date.now();
}

// RAG answer
async function ragAnswer(question, history) {
  const qVecs = await embed([question]);
  const hits  = await searchQdrant(qVecs[0]);

  const context = hits.map((h, i) => {
    const p = h.payload;
    const date = p.date ? new Date(p.date).toDateString() : "unknown";
    return `[${i+1}] "${p.title}" | ${date} | ${p.market_categories.join(", ")}
Link: ${p.post_link || "—"}
${p.content}`;
  }).join("\n\n---\n\n");

  const system = `You are a market intelligence assistant (Llama 3.3 70B via Groq).
Answer using ONLY the retrieved articles below. Cite title + date for each reference.
Sort by most recent unless asked otherwise. Include links when available.

Retrieved articles (most relevant first):
${context || "No relevant articles found for this query."}`;

  const payload = JSON.stringify({
    model: GROQ_MODEL,
    max_tokens: 1000,
    messages: [
      { role: "system", content: system },
      ...history,
      { role: "user", content: question }
    ]
  });

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: "api.groq.com",
      path: "/openai/v1/chat/completions",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
        "Authorization": `Bearer ${GROQ_API_KEY}`
      }
    }, res => {
      let data = "";
      res.on("data", c => data += c);
      res.on("end", () => {
        try {
          const r = JSON.parse(data);
          if (r.error) return reject(new Error(r.error.message));
          resolve(r.choices?.[0]?.message?.content || "");
        } catch (e) { reject(e); }
      });
    });
    req.on("error", reject);
    req.write(payload);
    req.end();
  });
}

// HTTP server
function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}
function respond(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

const server = http.createServer((req, res) => {
  cors(res);
  if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }
  const { pathname } = url.parse(req.url, true);

  if (pathname === "/sources" && req.method === "GET") {
    const sources = Object.entries(indexedFiles).map(([file, info]) => ({
      name: file.replace(".json", ""), file, records: info.count || 0, updatedAt: info.updatedAt
    }));
    respond(res, 200, { sources, lastIndexed, watchDir: WATCH_DIR, total: sources.length, totalArticles, rag: true });
    return;
  }

  if (pathname === "/status" && req.method === "GET") {
    respond(res, 200, { ok: true, qdrantReady, totalArticles, lastIndexed, model: GROQ_MODEL });
    return;
  }

  if (pathname === "/chat" && req.method === "POST") {
    if (!GROQ_API_KEY)  { respond(res, 400, { error: { message: "GROQ_API_KEY not set" } }); return; }
    if (!qdrantReady)   { respond(res, 503, { error: { message: "Vector index initializing, please wait..." } }); return; }
    let body = "";
    req.on("data", c => body += c);
    req.on("end", async () => {
      try {
        const { messages } = JSON.parse(body);
        const question = messages?.[messages.length - 1]?.content || "";
        const history  = (messages || []).slice(0, -1);
        if (!question) { respond(res, 400, { error: { message: "No question" } }); return; }
        const answer = await ragAnswer(question, history);
        respond(res, 200, { content: [{ type: "text", text: answer }] });
      } catch (e) {
        log(`Chat error: ${e.message}`);
        respond(res, 500, { error: { message: e.message } });
      }
    });
    return;
  }

  if (pathname === "/" || pathname === "/health") {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end(`RAG Agent | Articles: ${totalArticles} | Qdrant: ${qdrantReady ? "ready" : "init"}`);
    return;
  }

  respond(res, 404, { error: "Not found" });
});

// Boot
async function boot() {
  log("Booting RAG Data Agent...");
  if (!QDRANT_URL || !QDRANT_API_KEY || !HF_API_KEY) {
    log("ERROR: Missing QDRANT_URL, QDRANT_API_KEY, or HF_API_KEY env vars");
  } else {
    try {
      await ensureCollection();
      qdrantReady = true;
      await indexFolder();
      log(`Ready. ${totalArticles} articles indexed.`);
      setInterval(indexFolder, POLL_INTERVAL);
    } catch (e) {
      log(`Boot error: ${e.message}`);
    }
  }
  server.listen(PORT, () => log(`Listening on port ${PORT}`));
}

boot();

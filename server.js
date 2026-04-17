/**
 * JSON Data Agent — RAG Server with Auth0 JWT + Tier-Based Access
 * ---------------------------------------------------------------
 * ENV VARS (set in Railway dashboard):
 *   GROQ_API_KEY        — groq.com API key
 *   QDRANT_URL          — Qdrant Cloud cluster URL
 *   QDRANT_API_KEY      — Qdrant Cloud API key
 *   HF_API_KEY          — HuggingFace token
 *   AUTH0_DOMAIN        — e.g. dev-xxx.us.auth0.com
 *   AUTH0_AUDIENCE      — your Auth0 API identifier
 *   PRICE_API_URL       — live price CSV endpoint
 *   PORT                — set automatically by Railway
 */

const http   = require("http");
const https  = require("https");
const fs     = require("fs");
const path   = require("path");
const url    = require("url");
const crypto = require("crypto");

// ── Config ─────────────────────────────────────────────────────────────────────
const PORT           = process.env.PORT || 3456;
const WATCH_DIR      = path.resolve(process.env.DATA_DIR || "./data");
const POLL_INTERVAL  = 5 * 60 * 1000;
const GROQ_API_KEY   = process.env.GROQ_API_KEY   || "";
const GROQ_MODEL     = (process.env.GROQ_MODEL || "llama-3.1-8b-instant").replace(/^["']|["']$/g, "");
const QDRANT_URL     = (process.env.QDRANT_URL     || "").replace(/\/$/, "");
const QDRANT_API_KEY = process.env.QDRANT_API_KEY  || "";
const HF_API_KEY     = process.env.HF_API_KEY      || "";
const AUTH0_DOMAIN   = process.env.AUTH0_DOMAIN    || "dev-uxt65wdnctuy40v8.us.auth0.com";
const AUTH0_AUDIENCE = process.env.AUTH0_AUDIENCE  || "";
const COLLECTION     = "articles";
const EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2";
const TOP_K          = 3;
const PRICE_API_URL  = process.env.PRICE_API_URL || "";
const PRICE_REFRESH  = 4 * 60 * 60 * 1000;

// ── Usage limits ──────────────────────────────────────────────────────────────
const DAILY_LIMITS = { free: 10, essential: 50, enterprise: Infinity };
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || "ccarbon-admin-2026";
const CONTACT_EMAIL  = process.env.CONTACT_EMAIL  || "info@ccarbon.info";

// ── Tier definitions ───────────────────────────────────────────────────────────
const TIER_POST_TYPES = {
  free:       ["news", "post"],
  essential:  ["news", "post", "pricecommentary", "insight"],
  enterprise: ["news", "post", "pricecommentary", "insight", "webinar", "report"]
};

function getTierFromRoles(roles) {
  if (!roles || !roles.length) return "free";
  if (roles.includes("enterprise")) return "enterprise";
  if (roles.includes("essential"))  return "essential";
  return "free";
}

function getAllowedPostTypes(tier) {
  return TIER_POST_TYPES[tier] || TIER_POST_TYPES.free;
}

// ── State ──────────────────────────────────────────────────────────────────────
let indexedFiles     = {};
let totalArticles    = 0;
let lastIndexed      = null;
let qdrantReady      = false;
let priceData        = [];
let priceLastFetched = null;
let priceFetchError  = null;
let jwksCache        = null;
let jwksCacheTime    = 0;
// Usage tracking: { userId: { date: "YYYY-MM-DD", count: N, email, tier, lastSeen } }
let usageStore       = {};

function log(msg) { console.log(`[${new Date().toISOString()}] ${msg}`); }

// ── Usage tracking ────────────────────────────────────────────────────────────
function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function getUsage(userId) {
  const today = todayStr();
  if (!usageStore[userId] || usageStore[userId].date !== today) {
    usageStore[userId] = { ...( usageStore[userId] || {}), date: today, count: 0 };
  }
  return usageStore[userId];
}

function incrementUsage(userId, email, tier) {
  const usage = getUsage(userId);
  usage.count++;
  usage.email = email || usage.email || userId;
  usage.tier  = tier;
  usage.lastSeen = new Date().toISOString();
  return usage.count;
}

function checkLimit(userId, tier) {
  const limit = DAILY_LIMITS[tier] || DAILY_LIMITS.free;
  if (limit === Infinity) return { allowed: true, count: 0, limit };
  const usage = getUsage(userId);
  return { allowed: usage.count < limit, count: usage.count, limit };
}

function getAllUsageStats() {
  const today = todayStr();
  return Object.entries(usageStore).map(([userId, data]) => ({
    userId,
    email:    data.email    || userId,
    tier:     data.tier     || "free",
    today:    data.date === today ? data.count : 0,
    total:    data.count    || 0,
    lastSeen: data.lastSeen || "—"
  })).sort((a, b) => b.today - a.today);
}

// ── HTTPS helper ───────────────────────────────────────────────────────────────
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

// ── Auth0 JWT Verification ─────────────────────────────────────────────────────
async function getJWKS() {
  const now = Date.now();
  if (jwksCache && now - jwksCacheTime < 60 * 60 * 1000) return jwksCache;
  const res = await httpsRequest({
    hostname: AUTH0_DOMAIN,
    path: "/.well-known/jwks.json",
    method: "GET"
  });
  if (res.status !== 200) throw new Error("Failed to fetch JWKS");
  jwksCache = res.body;
  jwksCacheTime = now;
  return jwksCache;
}

function base64urlDecode(str) {
  str = str.replace(/-/g, "+").replace(/_/g, "/");
  while (str.length % 4) str += "=";
  return Buffer.from(str, "base64");
}

async function verifyJWT(token) {
  if (!token) throw new Error("No token provided");

  const parts = token.split(".");
  if (parts.length !== 3) throw new Error("Invalid JWT format");

  const header  = JSON.parse(base64urlDecode(parts[0]).toString());
  const payload = JSON.parse(base64urlDecode(parts[1]).toString());

  // Check expiry
  if (payload.exp && payload.exp < Math.floor(Date.now() / 1000)) {
    throw new Error("Token expired");
  }

  // Check issuer
  if (payload.iss !== `https://${AUTH0_DOMAIN}/`) {
    throw new Error("Invalid token issuer");
  }

  // Verify signature using JWKS
  const jwks = await getJWKS();
  const jwk  = jwks.keys.find(k => k.kid === header.kid);
  if (!jwk) throw new Error("No matching key found in JWKS");

  const pubKey = crypto.createPublicKey({ key: jwk, format: "jwk" });
  const signingInput = Buffer.from(`${parts[0]}.${parts[1]}`);
  const signature    = base64urlDecode(parts[2]);

  const valid = crypto.verify("sha256", signingInput, pubKey, signature);
  if (!valid) throw new Error("Invalid token signature");

  return payload;
}

function extractToken(req) {
  const auth = req.headers["authorization"] || "";
  if (auth.startsWith("Bearer ")) return auth.slice(7);
  return null;
}

// Get user tier from JWT payload
// Auth0 stores custom claims with a namespace
function getUserTier(payload) {
  const ns = `https://ccarbon.info/roles`;
  const roles = payload[ns] || payload["https://ccarbon/roles"] || payload.roles || [];
  return getTierFromRoles(Array.isArray(roles) ? roles : [roles]);
}

// ── Qdrant ─────────────────────────────────────────────────────────────────────
function qdrantHost() {
  try { return new URL(QDRANT_URL).hostname; } catch { return ""; }
}

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
  await qdrantReq("DELETE", `/collections/${COLLECTION}`);
  log(`Dropped old collection "${COLLECTION}"`);
  const testVecs = await embed(["test"]);
  const dim = testVecs[0].length;
  log(`Detected embedding dimension: ${dim}`);
  if (dim < 10) throw new Error(`Embedding dim too small (${dim})`);
  const res = await qdrantReq("PUT", `/collections/${COLLECTION}`, {
    vectors: { size: dim, distance: "Cosine" }
  });
  if (res.status === 200 || res.status === 201) log(`Collection "${COLLECTION}" created (dim=${dim})`);
  else throw new Error(`Failed to create collection: ${JSON.stringify(res.body)}`);

  // Create payload index on post_type for tier-based filtering
  const idxRes = await qdrantReq("PUT", `/collections/${COLLECTION}/index`, {
    field_name: "post_type",
    field_schema: "keyword"
  });
  if (idxRes.status === 200 || idxRes.status === 201) log(`Index created on post_type`);
  else log(`Warning: could not create post_type index: ${JSON.stringify(idxRes.body)}`);
}

async function upsertPoints(points) {
  const res = await qdrantReq("PUT", `/collections/${COLLECTION}/points?wait=true`, { points });
  if (res.status !== 200) throw new Error(`Upsert failed: ${JSON.stringify(res.body)}`);
}

// Search with tier-based filter
async function searchQdrant(vec, allowedPostTypes) {
  const body = {
    vector: vec,
    limit: TOP_K,
    with_payload: true
  };

  // Add post_type filter if not enterprise (enterprise sees everything)
  if (allowedPostTypes && allowedPostTypes.length < TIER_POST_TYPES.enterprise.length) {
    body.filter = {
      must: [{
        key: "post_type",
        match: { any: allowedPostTypes }
      }]
    };
  }

  const res = await qdrantReq("POST", `/collections/${COLLECTION}/points/search`, body);
  if (res.status !== 200) throw new Error(`Search failed: ${JSON.stringify(res.body)}`);
  return res.body.result || [];
}

async function deleteByFile(file) {
  await qdrantReq("POST", `/collections/${COLLECTION}/points/delete`, {
    filter: { must: [{ key: "source_file", match: { value: file } }] }
  });
}

// ── Embeddings ─────────────────────────────────────────────────────────────────
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
  function flatten(v) { while (Array.isArray(v) && Array.isArray(v[0])) v = v[0]; return v; }
  if (Array.isArray(raw)) {
    if (typeof raw[0] === "number") return [raw];
    if (Array.isArray(raw[0])) {
      if (typeof raw[0][0] === "number") return raw;
      return raw.map(item => flatten(item));
    }
  }
  return [flatten(raw)];
}

// ── Indexing ───────────────────────────────────────────────────────────────────
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
        source_file:       file,
        title:             a.title             || "",
        content:           a.content           || "",
        date:              a.date              || "",
        author:            a.author            || "",
        post_type:         a.post_type         || "news",   // ← used for tier filtering
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

// ── RAG answer ─────────────────────────────────────────────────────────────────
async function ragAnswer(question, history, tier) {
  const allowedPostTypes = getAllowedPostTypes(tier);
  const qVecs = await embed([question]);
  const hits  = await searchQdrant(qVecs[0], allowedPostTypes);

  // Filter out low-confidence matches (score < 0.3 means barely relevant)
  const relevantHits = hits.filter(h => h.score >= 0.3);
  const context = relevantHits.map((h, i) => {
    const p = h.payload;
    const date = p.date ? new Date(p.date).toDateString() : "unknown";
    const snippet = (p.content || "").slice(0, 500);
    return `[${i+1}] "${p.title}" | ${date} | ${(p.market_categories||[]).join(", ")} | Type: ${p.post_type||"—"}
Author: ${p.author||"—"} | Link: ${p.post_link || "—"}
${snippet}`;
  }).join("\n\n---\n\n");

  const priceContext = isPriceQuery(question)
    ? "\n\n=== LIVE LCFS PRICE DATA ===\n" + priceDataSummary()
    : "";

  const priceInstructions = isPriceQuery(question)
    ? "\nPRICE DATA: Present the price data as a markdown table (Date|Benchmark|Spot|Front Nodal). Write a short trend summary after the table. Prefix the table with [PRICE_TABLE]."
    : "";

  const tierNote = tier === "free"
    ? "\n\nNote: This user is on the Free tier. Only news and public posts are available to them."
    : tier === "essential"
    ? "\n\nNote: This user is on the Essential tier. News, price commentaries, and insights are available."
    : "";

  const system = `You are a strict market intelligence assistant for cCarbon. You have access to a set of retrieved documents below.

CRITICAL RULES — MUST FOLLOW:
1. ONLY use information from the "Retrieved documents" section below. NEVER use your training knowledge.
2. If the retrieved documents do not contain relevant information for the query, respond ONLY with: "No relevant documents were found in the cCarbon library for this query."
3. Do NOT mention, cite, or reference any external sources, reports, organizations, or data that is not explicitly present in the retrieved documents.
4. Do NOT invent titles, authors, dates, statistics, or URLs.
5. Every fact you state must be traceable to one of the numbered retrieved documents.

OUTPUT FORMAT — follow exactly when relevant documents ARE found:

**Executive Summary**
• [2-3 bullet points summarizing key findings from the retrieved documents only]

---
**[Exact title from document]**
**Type:** [post_type] | **Author:** [author] | **Date:** [readable date]
**Summary:** [2-3 sentence prose paragraph using ONLY information from this document]
[SOURCE_LINK: post_link_url]

RULES:
- Only output document blocks for documents actually retrieved below.
- [SOURCE_LINK: url] must be the exact post_link value from the document. If post_link is empty, omit the SOURCE_LINK line.
- No bullet points inside Summary — prose only.
- Dates in readable format (e.g. "17 February 2026").
- If documents contradict each other, note the difference.
- Never say "please let me know" or "feel free to ask" — end the response after the last document.
${tierNote}

Retrieved documents (most relevant first):
${context || "No relevant documents found for this query."}${priceContext}${priceInstructions}`;

  const payload = JSON.stringify({
    model: GROQ_MODEL,
    max_tokens: 1000,
    messages: [
      { role: "system", content: system },
      ...history,
      { role: "user", content: question }
    ]
  });

  async function callGroq(retries = 3) {
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
        res.on("end", async () => {
          try {
            const r = JSON.parse(data);
            if (r.error) {
              // Rate limit — extract wait time and retry
              if (r.error.code === "rate_limit_exceeded" && retries > 0) {
                const match = (r.error.message || "").match(/try again in ([\d.]+)s/);
                const wait = match ? Math.ceil(parseFloat(match[1]) * 1000) + 500 : 5000;
                log(`Groq rate limit hit — retrying in ${wait}ms (${retries} retries left)`);
                setTimeout(async () => {
                  try { resolve(await callGroq(retries - 1)); }
                  catch (e) { reject(e); }
                }, wait);
                return;
              }
              return reject(new Error(r.error.message));
            }
            resolve(r.choices?.[0]?.message?.content || "");
          } catch (e) { reject(e); }
        });
      });
      req.on("error", reject);
      req.write(payload);
      req.end();
    });
  }

  return callGroq();
}

// ── Price API ──────────────────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
  return lines.slice(1).map(line => {
    const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g, ""));
    const row = {};
    headers.forEach((h, i) => { row[h] = vals[i] || ""; });
    return row;
  }).filter(r => r[headers[0]]);
}

async function fetchPriceData() {
  if (!PRICE_API_URL) { log("PRICE_API_URL not set, skipping price fetch"); return; }
  try {
    log("Fetching live price data...");
    const data = await new Promise((resolve, reject) => {
      const urlObj = new URL(PRICE_API_URL);
      const req = https.request({
        hostname: urlObj.hostname,
        path: urlObj.pathname + urlObj.search,
        method: "GET",
        headers: { "User-Agent": "cCarbon-Agent/1.0" }
      }, res => {
        let raw = "";
        res.on("data", c => raw += c);
        res.on("end", () => resolve(raw));
      });
      req.on("error", reject);
      req.end();
    });
    const rows = parseCSV(data);
    if (rows.length === 0) throw new Error("Empty CSV response");
    rows.sort((a, b) => new Date(b.Date || "") - new Date(a.Date || ""));
    priceData = rows;
    priceLastFetched = Date.now();
    priceFetchError = null;
    log(`Price data fetched: ${rows.length} rows, latest: ${rows[0]?.Date}`);
  } catch (e) {
    priceFetchError = e.message;
    log(`Price API error: ${e.message}`);
  }
}

function priceDataSummary() {
  if (!priceData.length) return "No price data available.";
  const latest = priceData[0];
  const fetched = priceLastFetched ? new Date(priceLastFetched).toLocaleString() : "unknown";
  const rows = priceData.slice(0, 30);
  const csvText = ["Date,Benchmark,Spot,Front (Nodal)"]
    .concat(rows.map(r => `${r.Date},${(r.Benchmark||"").replace(/[$]/g,"")},${(r.Spot||"").replace(/[$]/g,"")},${(r["Front (Nodal)"]||"").replace(/[$]/g,"")}`))
    .join("\n");
  return "Latest LCFS price data (as of " + fetched + "):\nMost recent date: " + latest.Date + "\n\n" + csvText;
}

function isPriceQuery(q) {
  return /price|benchmark|spot|nodal|lcfs credit|credit value|\$|cost|rate|trading|market price|latest price|current price/.test(q.toLowerCase());
}

// ── HTTP server ────────────────────────────────────────────────────────────────
function cors(res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
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

  // GET /prices — return cached price data
  if (pathname === "/prices" && req.method === "GET") {
    respond(res, 200, {
      data: priceData.slice(0, 400),
      lastFetched: priceLastFetched,
      error: priceFetchError,
      count: priceData.length
    });
    return;
  }

  // GET /auth/config — return Auth0 config for the frontend
  if (pathname === "/auth/config" && req.method === "GET") {
    respond(res, 200, {
      domain: AUTH0_DOMAIN,
      clientId: "Vkeg6vFZjoyoarltYjgu7TQLgxcGAcRz",
      audience: AUTH0_AUDIENCE || `https://${AUTH0_DOMAIN}/api/v2/`
    });
    return;
  }

  // POST /chat — RAG pipeline with auth
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

        // Verify JWT and get tier
        let tier = "free";
        let userId = "anonymous";
        let userEmail = "";
        const token = extractToken(req);
        if (token) {
          try {
            const payload = await verifyJWT(token);
            tier      = getUserTier(payload);
            userId    = payload.sub || "anonymous";
            userEmail = payload.email || payload.name || userId;
            log(`Chat request — user: ${userEmail} tier: ${tier}`);
          } catch (e) {
            log(`JWT verification failed: ${e.message} — defaulting to free tier`);
          }
        }

        // Check daily usage limit
        const limitCheck = checkLimit(userId, tier);
        if (!limitCheck.allowed) {
          respond(res, 429, {
            error: {
              message: "LIMIT_REACHED",
              count: limitCheck.count,
              limit: limitCheck.limit,
              tier,
              contactEmail: CONTACT_EMAIL
            }
          });
          return;
        }

        const answer = await ragAnswer(question, history, tier);

        // Increment usage after successful answer
        const newCount = incrementUsage(userId, userEmail, tier);
        log(`Usage: ${userEmail} — ${newCount}/${DAILY_LIMITS[tier] === Infinity ? "∞" : DAILY_LIMITS[tier]} today`);

        respond(res, 200, {
          content: [{ type: "text", text: answer }],
          tier,
          usage: { count: newCount, limit: DAILY_LIMITS[tier] === Infinity ? null : DAILY_LIMITS[tier] }
        });
      } catch (e) {
        log(`Chat error: ${e.message}`);
        respond(res, 500, { error: { message: e.message } });
      }
    });
    return;
  }

  if (pathname === "/" || pathname === "/health") {
    const htmlPath = path.join(__dirname, "agent.html");
    if (fs.existsSync(htmlPath)) {
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end(fs.readFileSync(htmlPath));
    } else {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end(`RAG Agent | Articles: ${totalArticles} | Qdrant: ${qdrantReady ? "ready" : "init"}`);
    }
    return;
  }

  // GET /admin — admin dashboard HTML
  if (pathname === "/admin" && req.method === "GET") {
    const authHeader = req.headers["authorization"] || "";
    const b64 = authHeader.replace("Basic ", "");
    let authed = false;
    try {
      const decoded = Buffer.from(b64, "base64").toString();
      authed = decoded === `admin:${ADMIN_PASSWORD}`;
    } catch(e) {}
    if (!authed) {
      res.writeHead(401, { "WWW-Authenticate": 'Basic realm="cCarbon Admin"', "Content-Type": "text/plain" });
      res.end("Unauthorized");
      return;
    }
    const stats = getAllUsageStats();
    const today = todayStr();
    const totalToday = stats.reduce((a, s) => a + s.today, 0);
    const tierCounts = { free: 0, essential: 0, enterprise: 0 };
    stats.forEach(s => { if (tierCounts[s.tier] !== undefined) tierCounts[s.tier]++; });

    const rows = stats.map(s => `
      <tr>
        <td>${s.email}</td>
        <td><span class="badge ${s.tier}">${s.tier}</span></td>
        <td>${s.today}</td>
        <td>${DAILY_LIMITS[s.tier] === Infinity ? "∞" : DAILY_LIMITS[s.tier]}</td>
        <td>${s.lastSeen !== "—" ? new Date(s.lastSeen).toLocaleString() : "—"}</td>
      </tr>`).join("");

    const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>cCarbon Admin</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3; font-size: 14px; }
  .header { background: #203A6B; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  .header h1 { font-size: 18px; font-weight: 600; color: white; }
  .header span { font-size: 12px; color: rgba(255,255,255,0.6); font-family: monospace; }
  .content { padding: 24px; }
  .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px; }
  .card-label { font-size: 11px; color: #7d8590; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
  .card-val { font-size: 28px; font-weight: 600; font-family: monospace; }
  .card-val.orange { color: #E3662B; }
  .card-val.blue   { color: #6699cc; }
  .card-val.green  { color: #3fb950; }
  .section-title { font-size: 13px; font-weight: 600; color: #7d8590; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }
  th { padding: 10px 16px; text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #7d8590; border-bottom: 1px solid #30363d; background: #1c2128; }
  td { padding: 10px 16px; border-bottom: 1px solid #21262d; color: #e6edf3; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #1c2128; }
  .badge { font-size: 10px; padding: 2px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; font-family: monospace; }
  .badge.free       { background: #21262d; color: #7d8590; }
  .badge.essential  { background: #0d1e35; color: #6699cc; }
  .badge.enterprise { background: #3d1f0f; color: #E3662B; }
  .refresh { float: right; padding: 6px 14px; background: #21262d; border: 1px solid #30363d; border-radius: 6px; color: #e6edf3; cursor: pointer; font-size: 12px; text-decoration: none; }
  .refresh:hover { border-color: #E3662B; color: #E3662B; }
  .empty { padding: 40px; text-align: center; color: #7d8590; }
</style>
</head>
<body>
<div class="header">
  <h1>cCarbon Admin Dashboard</h1>
  <span>Today: ${today}</span>
  <a href="/admin" class="refresh" style="margin-left:auto;">↻ Refresh</a>
</div>
<div class="content">
  <div class="cards">
    <div class="card"><div class="card-label">Queries Today</div><div class="card-val orange">${totalToday}</div></div>
    <div class="card"><div class="card-label">Total Users</div><div class="card-val">${stats.length}</div></div>
    <div class="card"><div class="card-label">Enterprise</div><div class="card-val orange">${tierCounts.enterprise}</div></div>
    <div class="card"><div class="card-label">Essential</div><div class="card-val blue">${tierCounts.essential}</div></div>
  </div>
  <div class="section-title">User Activity</div>
  <table>
    <thead><tr><th>User</th><th>Tier</th><th>Queries Today</th><th>Daily Limit</th><th>Last Seen</th></tr></thead>
    <tbody>${rows || '<tr><td colspan="5" class="empty">No activity yet today</td></tr>'}</tbody>
  </table>
</div>
</body>
</html>`;
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end(html);
    return;
  }

  // GET /admin/usage — JSON usage stats API
  if (pathname === "/admin/usage" && req.method === "GET") {
    const authHeader = req.headers["authorization"] || "";
    const b64 = authHeader.replace("Basic ", "");
    let authed = false;
    try { authed = Buffer.from(b64, "base64").toString() === `admin:${ADMIN_PASSWORD}`; } catch(e) {}
    if (!authed) { respond(res, 401, { error: "Unauthorized" }); return; }
    respond(res, 200, {
      stats: getAllUsageStats(),
      today: todayStr(),
      limits: DAILY_LIMITS
    });
    return;
  }

  respond(res, 404, { error: "Not found" });
});

// ── Boot ───────────────────────────────────────────────────────────────────────
async function boot() {
  log("Booting RAG Data Agent with Auth0...");
  log(`Auth0 domain: ${AUTH0_DOMAIN}`);
  log(`Tier mapping: free=${TIER_POST_TYPES.free.join(",")} | essential=${TIER_POST_TYPES.essential.join(",")} | enterprise=all`);

  if (!QDRANT_URL || !QDRANT_API_KEY || !HF_API_KEY) {
    log("ERROR: Missing QDRANT_URL, QDRANT_API_KEY, or HF_API_KEY");
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

  await fetchPriceData();
  setInterval(fetchPriceData, PRICE_REFRESH);

  server.listen(PORT, () => log(`Listening on port ${PORT}`));
}

boot();

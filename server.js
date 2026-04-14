/**
 * JSON Data Agent — Server
 * ------------------------
 * Reads JSON files from ./data folder, serves them to the chat UI.
 * Deployed on Railway. Data updated by pushing to GitHub.
 *
 * ENV VARS (set in Railway dashboard):
 *   GROQ_API_KEY   — your Groq API key (required)
 *   PORT           — set automatically by Railway
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const url = require("url");

// ── Config ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3456;
const WATCH_DIR = path.resolve(process.env.DATA_DIR || "./data");
const POLL_INTERVAL_MS = 5000;
const GROQ_API_KEY = process.env.GROQ_API_KEY || "";
const GROQ_MODEL = "llama-3.3-70b-versatile";

// ── State ──────────────────────────────────────────────────────────────────────
let index = {};
let lastIndexed = null;

// ── Helpers ────────────────────────────────────────────────────────────────────
function log(msg) {
  console.log(`[${new Date().toISOString()}] ${msg}`);
}

function indexFolder() {
  let changed = false;
  const newIndex = {};

  if (!fs.existsSync(WATCH_DIR)) {
    log(`Data folder not found: ${WATCH_DIR}`);
    return false;
  }

  let files;
  try {
    files = fs.readdirSync(WATCH_DIR).filter(f => f.endsWith(".json"));
  } catch (e) {
    log(`Cannot read folder: ${e.message}`);
    return false;
  }

  for (const file of files) {
    const filePath = path.join(WATCH_DIR, file);
    let stat;
    try { stat = fs.statSync(filePath); } catch { continue; }

    const prev = index[file];
    const updatedAt = stat.mtimeMs;

    if (prev && prev.updatedAt === updatedAt) {
      newIndex[file] = prev;
      continue;
    }

    let data;
    try {
      const raw = fs.readFileSync(filePath, "utf8");
      data = JSON.parse(raw);
      changed = true;
      log(`Indexed: ${file} (${Array.isArray(data) ? data.length + " records" : typeof data})`);
    } catch (e) {
      log(`Skipped ${file}: invalid JSON — ${e.message}`);
      continue;
    }

    newIndex[file] = {
      name: file.replace(".json", ""),
      file,
      data,
      size: stat.size,
      updatedAt,
      records: Array.isArray(data) ? data.length : null
    };
  }

  for (const file of Object.keys(index)) {
    if (!newIndex[file]) { changed = true; log(`Removed: ${file}`); }
  }

  index = newIndex;
  if (changed) lastIndexed = Date.now();
  return changed;
}

// ── Groq proxy ─────────────────────────────────────────────────────────────────
function proxyToGroq(incoming, res) {
  const groqMessages = [];
  if (incoming.system) groqMessages.push({ role: "system", content: incoming.system });
  (incoming.messages || []).forEach(m => groqMessages.push(m));

  const groqBody = {
    model: GROQ_MODEL,
    max_tokens: incoming.max_tokens || 1000,
    messages: groqMessages
  };

  const payload = JSON.stringify(groqBody);
  const options = {
    hostname: "api.groq.com",
    path: "/openai/v1/chat/completions",
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(payload),
      "Authorization": `Bearer ${GROQ_API_KEY}`
    }
  };

  const apiReq = https.request(options, apiRes => {
    let data = "";
    apiRes.on("data", chunk => data += chunk);
    apiRes.on("end", () => {
      try {
        const groqResp = JSON.parse(data);
        if (groqResp.error) {
          res.writeHead(apiRes.statusCode, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
          res.end(JSON.stringify({ error: groqResp.error }));
          return;
        }
        const normalized = {
          content: [{ type: "text", text: groqResp.choices?.[0]?.message?.content || "" }],
          model: groqResp.model,
          usage: groqResp.usage
        };
        res.writeHead(200, { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" });
        res.end(JSON.stringify(normalized));
      } catch (e) {
        respond(res, 502, { error: { message: "Bad response from Groq: " + e.message } });
      }
    });
  });

  apiReq.on("error", e => respond(res, 502, { error: { message: "Groq unreachable: " + e.message } }));
  apiReq.write(payload);
  apiReq.end();
}

// ── HTTP server ────────────────────────────────────────────────────────────────
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
    const sources = Object.values(index).map(s => ({
      name: s.name, file: s.file, records: s.records, updatedAt: s.updatedAt, data: s.data
    }));
    respond(res, 200, { sources, lastIndexed, watchDir: WATCH_DIR, total: sources.length });
    return;
  }

  if (pathname === "/status" && req.method === "GET") {
    respond(res, 200, {
      ok: true,
      watchDir: WATCH_DIR,
      sourceCount: Object.keys(index).length,
      lastIndexed,
      model: GROQ_MODEL,
      apiKeySet: !!GROQ_API_KEY
    });
    return;
  }

  if (pathname === "/chat" && req.method === "POST") {
    if (!GROQ_API_KEY) {
      respond(res, 400, { error: { message: "GROQ_API_KEY not set in Railway environment variables." } });
      return;
    }
    let body = "";
    req.on("data", chunk => body += chunk);
    req.on("end", () => {
      try { proxyToGroq(JSON.parse(body), res); }
      catch (e) { respond(res, 400, { error: { message: "Invalid request body" } }); }
    });
    return;
  }

  // Health check for Railway
  if (pathname === "/" || pathname === "/health") {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.end("JSON Data Agent running. Sources: " + Object.keys(index).length);
    return;
  }

  respond(res, 404, { error: "Not found" });
});

// ── Start ──────────────────────────────────────────────────────────────────────
indexFolder();
setInterval(indexFolder, POLL_INTERVAL_MS);

server.listen(PORT, () => {
  log(`Server listening on port ${PORT}`);
  log(`Watching: ${WATCH_DIR}`);
  log(`Model: ${GROQ_MODEL}`);
  log(`API key: ${GROQ_API_KEY ? "✓ set" : "✗ NOT SET — add GROQ_API_KEY in Railway variables"}`);
});

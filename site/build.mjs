// Builds the static site in ./public.
//
//   node build.mjs               copy browser libraries + compute knowledge base + demo data
//   node build.mjs --vendor-only copy browser libraries only (what Render runs on deploy)
//   node build.mjs --record      also re-record the demo answers with a local Ollama model
//
// The knowledge base is embedded here with the SAME model and library the browser uses
// (transformers.js + bge-small), so question vectors and passage vectors match exactly.

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { EMBED_MODEL, QUERY_PREFIX, SYSTEM_PROMPT, LLM_MODEL_LABEL, buildPrompt, cosineScores, project } from "./public/shared.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = join(HERE, "public");
const ROOT = join(HERE, "..");
const args = new Set(process.argv.slice(2));

const EXAMPLES = [
  "What is a token?",
  "How does attention work in a transformer?",
  "Why do AI models hallucinate?",
  "What does temperature do?",
  "How is a model trained?",
];
const CHUNK_CHARS = 700;
const CHUNK_OVERLAP = 120;
const TOP_K = 4;
const OLLAMA_MODEL = "llama3.2:3b"; // same model as the in-browser Llama-3.2-3B

/* ---------- 1. browser libraries (served from our own site, not a CDN) ---------- */

function copyVendor() {
  const vendor = join(PUBLIC, "vendor");
  mkdirSync(vendor, { recursive: true });
  const nm = join(HERE, "node_modules");
  const files = [
    [join(nm, "@huggingface/transformers/dist/transformers.min.js"), "transformers.min.js"],
    [join(nm, "@mlc-ai/web-llm/lib/index.js"), "web-llm.js"],
  ];
  const ortDist = join(nm, "onnxruntime-web/dist");
  for (const f of readdirSync(ortDist)) {
    if (/^ort-wasm-simd-threaded(\.(jsep|asyncify|jspi))?\.(mjs|wasm)$/.test(f)) files.push([join(ortDist, f), f]);
  }
  for (const [from, to] of files) copyFileSync(from, join(vendor, to));
  copyFileSync(join(ROOT, "static/style.css"), join(PUBLIC, "style.css")); // shared look with the Python app
  console.log(`vendor: copied ${files.length} files`);
}

/* ---------- 2. chunking (mirrors app/rag.py) ---------- */

function splitLong(text, size, overlap) {
  if (text.length <= size) return [text];
  const sentences = text.split(/(?<=[.!?])\s+/);
  const pieces = [];
  let current = "";
  for (const sentence of sentences) {
    if (current && current.length + sentence.length + 1 > size) {
      pieces.push(current.trim());
      current = current.slice(-overlap) + " " + sentence;
    } else {
      current = current ? `${current} ${sentence}` : sentence;
    }
  }
  if (current.trim()) pieces.push(current.trim());
  return pieces;
}

function chunkMarkdown(md) {
  const title = (md.match(/^#\s+(.+)$/m) || [, "Untitled"])[1].trim();
  const out = [];
  for (const block of md.split(/^##\s+/m).slice(1)) {
    const nl = block.indexOf("\n");
    const heading = block.slice(0, nl).trim();
    const body = block.slice(nl + 1).split(/\s+/).filter(Boolean).join(" ");
    for (const piece of splitLong(body, CHUNK_CHARS, CHUNK_OVERLAP)) out.push([heading, piece]);
  }
  return [title, out];
}

/* ---------- 3. PCA to 2D (power iteration, no dependencies) ---------- */

function pca2(vectors) {
  const n = vectors.length, d = vectors[0].length;
  const mean = new Array(d).fill(0);
  for (const v of vectors) for (let j = 0; j < d; j++) mean[j] += v[j] / n;
  const X = vectors.map((v) => v.map((x, j) => x - mean[j]));
  const total = X.reduce((s, row) => s + row.reduce((a, x) => a + x * x, 0), 0);
  const comps = [], explained = [];
  for (let c = 0; c < 2; c++) {
    let w = Array.from({ length: d }, (_, j) => Math.sin(j + 1 + c * 7)); // deterministic start
    for (let it = 0; it < 500; it++) {
      const Xw = X.map((row) => row.reduce((a, x, j) => a + x * w[j], 0));
      const next = new Array(d).fill(0);
      X.forEach((row, i) => row.forEach((x, j) => (next[j] += x * Xw[i])));
      for (const prev of comps) { // keep orthogonal to earlier components
        const dot = next.reduce((a, x, j) => a + x * prev[j], 0);
        for (let j = 0; j < d; j++) next[j] -= dot * prev[j];
      }
      const norm = Math.hypot(...next);
      w = next.map((x) => x / norm);
    }
    comps.push(w);
    const proj = X.map((row) => row.reduce((a, x, j) => a + x * w[j], 0));
    explained.push(proj.reduce((a, p) => a + p * p, 0) / total);
  }
  return { mean, components: comps, explained };
}

const round = (arr, dp) => arr.map((x) => Number(x.toFixed(dp)));

/* ---------- 4. knowledge base + demo data ---------- */

async function buildData() {
  const { pipeline } = await import("@huggingface/transformers");
  const extractor = await pipeline("feature-extraction", EMBED_MODEL, { dtype: "q8" });
  const embed = async (texts) => (await extractor(texts, { pooling: "cls", normalize: true })).tolist();

  const chunks = [];
  const docs = [];
  for (const file of readdirSync(join(ROOT, "knowledge")).filter((f) => f.endsWith(".md")).sort()) {
    const [title, pieces] = chunkMarkdown(readFileSync(join(ROOT, "knowledge", file), "utf8"));
    docs.push(title);
    for (const [section, text] of pieces) chunks.push({ id: chunks.length, doc: title, section, text });
  }
  const vectors = (await embed(chunks.map((c) => `${c.section}: ${c.text}`))).map((v) => round(v, 5));
  const pca = pca2(vectors);
  chunks.forEach((c, i) => ([c.x, c.y] = project(vectors[i], pca)));

  const kb = {
    embed_model: EMBED_MODEL,
    dimensions: vectors[0].length,
    docs,
    chunks,
    vectors,
    pca: { mean: round(pca.mean, 6), components: pca.components.map((c) => round(c, 6)), explained: pca.explained },
  };
  mkdirSync(join(PUBLIC, "data"), { recursive: true });
  writeFileSync(join(PUBLIC, "data/kb.json"), JSON.stringify(kb));
  console.log(`kb: ${docs.length} docs, ${chunks.length} chunks, ${kb.dimensions} dims`);

  // Demo: precompute every step for the example questions.
  const answersPath = join(HERE, "demo-answers.json");
  const answers = existsSync(answersPath) ? JSON.parse(readFileSync(answersPath, "utf8")) : {};
  const demo = [];
  for (const question of EXAMPLES) {
    const [qvec] = await embed([QUERY_PREFIX + question]);
    const scores = cosineScores(qvec, kb.vectors);
    const top_ids = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, TOP_K);
    const prompt = buildPrompt(question, top_ids.map((i) => chunks[i]));
    if (args.has("--record")) answers[question] = await recordAnswer(prompt);
    if (!answers[question]) throw new Error(`No recorded answer for "${question}". Run: npm run record`);
    const [x, y] = project(qvec, kb.pca);
    demo.push({ question, vector: round(qvec, 4), point: { x, y }, scores: round(scores, 4), top_ids, prompt, ...answers[question] });
  }
  if (args.has("--record")) writeFileSync(answersPath, JSON.stringify(answers, null, 2) + "\n");
  writeFileSync(join(PUBLIC, "data/demo.json"), JSON.stringify({ system: SYSTEM_PROMPT, examples: demo }));
  console.log(`demo: ${demo.length} recorded examples`);
}

async function recordAnswer(prompt) {
  const res = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      stream: false,
      options: { temperature: 0.2 },
      messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`Ollama returned ${res.status}. Is it running, and is ${OLLAMA_MODEL} pulled?`);
  const r = await res.json();
  console.log(`  recorded (${r.eval_count} tokens)`);
  return { answer: r.message.content.trim(), model: LLM_MODEL_LABEL, input_tokens: r.prompt_eval_count, output_tokens: r.eval_count };
}

copyVendor();
if (!args.has("--vendor-only")) await buildData();

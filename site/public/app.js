import { EMBED_MODEL, QUERY_PREFIX, SYSTEM_PROMPT, LLM_MODEL_LABEL, buildPrompt, cosineScores, project as pcaProject } from "./shared.js";

/* The language-model options ("backends"):
 *   webllm - Llama 3.2 3B downloaded from Hugging Face and run on the GPU with WebLLM
 *   chrome - the browser's own built-in model (e.g. Gemini Nano in Chrome), if already installed
 *   ollama - the Ollama app on the visitor's computer, only contacted after they click
 * The 34 MB search model (transformers.js) is needed for all of them, and is cached after first use.
 */
const LLM_F16 = { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", size: "1.8 GB", memory: "2.5 GB" };
const LLM_F32 = { id: "Llama-3.2-3B-Instruct-q4f32_1-MLC", size: "1.8 GB", memory: "3 GB" }; // for GPUs without 16-bit float support
const OLLAMA_BASES = ["http://localhost:11434", "http://127.0.0.1:11434"];
const CHROME_AI_OPTIONS = { expectedInputs: [{ type: "text", languages: ["en"] }], expectedOutputs: [{ type: "text", languages: ["en"] }] };
const MODE_KEY = "ragscope-mode";
const TOP_K = 4;
const PALETTE = ["#4f46e5", "#d9622b", "#0f9d76", "#c2367a", "#2b7bd9", "#b58b00", "#7a4fd1", "#5f7a2b"];
const SVG_NS = "http://www.w3.org/2000/svg";
const W = 400, H = 300, PAD = 20;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const withTimeout = (promise, ms, fallback) => Promise.race([promise, sleep(ms).then(() => fallback)]);

let kb, demo, toSvg;
let mode = null;            // "demo" | "webllm" | "chrome" | "ollama"
let ollama = null;          // { base, model } when mode === "ollama"
let gpu = { ok: false, f16: false, reason: "" };
let llmChoice = LLM_F16;
let found = { searchModel: false, webllm: false, chrome: false };
let extractor = null;       // transformers.js embedding pipeline
let engine = null;          // WebLLM engine
let llmReady = null;        // resolves when the chosen language model is ready
let busy = false;
let currentTop = [];
let queryPoint = null;

/* ---------- small helpers ---------- */

const store = {
  get() { try { return localStorage.getItem(MODE_KEY); } catch { return null; } },
  set(v) { try { v ? localStorage.setItem(MODE_KEY, v) : localStorage.removeItem(MODE_KEY); } catch {} },
};

function el(tag, attrs = {}, text) {
  const node = tag.startsWith("svg:") ? document.createElementNS(SVG_NS, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStep(name, status) {
  const step = $(`step-${name}`);
  step.classList.remove("active", "done");
  if (status) step.classList.add(status);
}

function setLoading(text, fraction) {
  $("loading").hidden = text === null;
  if (text !== null) {
    $("loading-text").textContent = text;
    $("loading-bar").style.width = `${Math.round((fraction ?? 0) * 100)}%`;
  }
}

function setInputs({ typing, examples }) {
  $("q").disabled = !typing;
  $("go").disabled = !typing;
  document.querySelectorAll("#examples button").forEach((b) => (b.disabled = !examples));
}

function llmLabel() {
  if (mode === "chrome") return "your browser's built-in AI";
  if (mode === "ollama") return `${ollama.model} (Ollama)`;
  return LLM_MODEL_LABEL;
}

/* ---------- drawing ---------- */

function setupMap() {
  const xs = kb.chunks.map((c) => c.x), ys = kb.chunks.map((c) => c.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  toSvg = (x, y) => [
    clamp(PAD + ((x - minX) / (maxX - minX || 1)) * (W - 2 * PAD), 6, W - 6),
    clamp(H - PAD - ((y - minY) / (maxY - minY || 1)) * (H - 2 * PAD), 6, H - 6),
  ];
  $("legend").replaceChildren(
    ...kb.docs.map((d, i) => {
      const s = el("span");
      s.append(el("i", { style: `background:${PALETTE[i % PALETTE.length]}` }), d.split(":")[0]);
      return s;
    })
  );
}

function drawMap(point, topIds = []) {
  const svg = $("map");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const top = new Set(topIds);
  let qx, qy;
  if (point) {
    [qx, qy] = toSvg(point.x, point.y);
    for (const id of topIds) {
      const [x, y] = toSvg(kb.chunks[id].x, kb.chunks[id].y);
      svg.append(el("svg:line", { class: "link", x1: qx, y1: qy, x2: x, y2: y }));
    }
  }
  for (const c of kb.chunks) {
    const [x, y] = toSvg(c.x, c.y);
    const dot = el("svg:circle", {
      cx: x, cy: y, r: top.has(c.id) ? 7 : 4.5, fill: PALETTE[kb.docs.indexOf(c.doc) % PALETTE.length],
      opacity: topIds.length && !top.has(c.id) ? 0.3 : 0.9, "data-id": c.id,
    });
    dot.append(el("svg:title", {}, `${c.doc} > ${c.section}`));
    svg.append(dot);
  }
  if (point) {
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const r = i % 2 ? 4.5 : 11, a = (Math.PI / 5) * i - Math.PI / 2;
      pts.push(`${qx + r * Math.cos(a)},${qy + r * Math.sin(a)}`);
    }
    const star = el("svg:polygon", { class: "star", points: pts.join(" ") });
    star.append(el("svg:title", {}, "Your question"));
    svg.append(star);
  }
}

function drawVector(vec) {
  const max = Math.max(...vec.map(Math.abs)) || 1;
  const css = getComputedStyle(document.documentElement);
  const pos = css.getPropertyValue("--pos").trim(), neg = css.getPropertyValue("--neg").trim();
  $("vector").replaceChildren(
    ...vec.map((v, i) => {
      const s = el("span", { title: `dimension ${i}: ${v.toFixed(3)}` });
      if (v !== 0) {
        s.style.background = v > 0 ? pos : neg;
        s.style.opacity = (0.15 + (0.85 * Math.abs(v)) / max).toFixed(2);
      }
      return s;
    })
  );
}

function drawScores(scores, topIds) {
  const best = Math.max(...scores);
  $("scores").replaceChildren(
    ...topIds.map((id, rank) => {
      const c = kb.chunks[id];
      const li = el("li", { "data-id": id });
      const meta = el("div", { class: "meta" });
      meta.append(el("span", {}, `[${rank + 1}] ${c.section}`), el("span", {}, scores[id].toFixed(3)));
      const bar = el("div", { class: "bar" });
      bar.append(el("div", { style: `width:${(scores[id] / best) * 100}%` }));
      li.append(meta, el("div", { class: "src" }, c.doc), bar, el("div", { class: "text" }, c.text));
      return li;
    })
  );
}

function renderAnswer(text, streaming) {
  const box = $("answer");
  box.replaceChildren();
  for (const part of text.split(/(\[\d+\])/g)) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m && currentTop[Number(m[1]) - 1] !== undefined) {
      const n = Number(m[1]);
      const chip = el("span", { class: "cite", "data-rank": n }, n);
      chip.title = kb.chunks[currentTop[n - 1]].section;
      box.append(chip);
    } else {
      box.append(document.createTextNode(part));
    }
  }
  box.classList.toggle("caret", streaming);
}

function highlight(rank) {
  const id = rank ? currentTop[rank - 1] : null;
  document.querySelectorAll("#scores li").forEach((li) => li.classList.toggle("hl", Number(li.dataset.id) === id));
  document.querySelectorAll("#map circle").forEach((c) => {
    c.setAttribute("stroke", Number(c.dataset.id) === id ? "var(--ink)" : "none");
    c.setAttribute("stroke-width", 2);
  });
}

function showError(message) {
  $("answer").classList.add("error");
  $("answer").classList.remove("caret");
  $("answer").textContent = message;
  setStep("generate", "done");
}

/* ---------- the four steps, shared by every mode ---------- */

function resetSteps() {
  ["embed", "retrieve", "prompt", "generate"].forEach((s) => setStep(s, null));
  ["t-embed", "t-prompt", "t-gen", "stats"].forEach((id) => ($(id).textContent = ""));
  $("answer").classList.remove("error");
  renderAnswer("", false);
  $("demo-note").hidden = mode !== "demo";
  $("llm").textContent = mode === "demo" ? LLM_MODEL_LABEL : llmLabel();
}

function showEmbed(vector, point, ms) {
  drawVector(vector);
  queryPoint = point;
  drawMap(point);
  $("t-embed").textContent = ms === null ? "recorded" : `${ms} ms`;
  setStep("embed", "done");
  setStep("retrieve", "active");
}

function showRetrieve(scores, topIds) {
  currentTop = topIds;
  drawMap(queryPoint, topIds);
  drawScores(scores, topIds);
  setStep("retrieve", "done");
  setStep("prompt", "active");
}

function showPrompt(prompt) {
  $("system").textContent = SYSTEM_PROMPT;
  $("prompt").textContent = prompt;
  const chars = SYSTEM_PROMPT.length + prompt.length;
  $("t-prompt").textContent = `${chars.toLocaleString()} characters ≈ ${Math.round(chars / 4).toLocaleString()} tokens`;
  setStep("prompt", "done");
  setStep("generate", "active");
}

/* ---------- finding AI that's already on this device ---------- */

/** Has the 34 MB search model been downloaded by this site before? */
async function searchModelCached() {
  try {
    if (!(await caches.has("transformers-cache"))) return false;
    const keys = await (await caches.open("transformers-cache")).keys();
    return keys.some((r) => r.url.includes("bge-small-en-v1.5") && r.url.endsWith(".onnx"));
  } catch {
    return false;
  }
}

/** Is RAGScope's language model fully in the cache? Same check as WebLLM's hasModelInCache,
 *  done with the Cache API directly so we don't load the 6.5 MB library just to look. */
async function webllmCached(modelId) {
  try {
    if (!(await caches.has("webllm/model"))) return false;
    const cache = await caches.open("webllm/model");
    const base = `https://huggingface.co/mlc-ai/${modelId}/resolve/main/`;
    const manifest = await cache.match(base + "tensor-cache.json");
    if (!manifest) return false;
    const { records } = await manifest.json();
    const hits = await Promise.all(records.map((r) => cache.match(new URL(r.dataPath, base).href)));
    return hits.every(Boolean); // every shard present = the download finished
  } catch {
    return false;
  }
}

/** Does the browser have its own AI model installed and ready (e.g. Gemini Nano in Chrome)? */
async function chromeAIAvailable() {
  try {
    if (!("LanguageModel" in self)) return false;
    return (await withTimeout(LanguageModel.availability(CHROME_AI_OPTIONS), 3000, "unavailable")) === "available";
  } catch {
    return false;
  }
}

/** Ask the visitor's own Ollama app which models it has. Only ever called after a click
 *  (or when they chose Ollama on an earlier visit). */
async function findOllama() {
  for (const base of OLLAMA_BASES) {
    try {
      const res = await fetch(`${base}/api/tags`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) continue;
      const models = (await res.json()).models.map((m) => m.name).filter((n) => !/embed/i.test(n));
      return { base, models };
    } catch {}
  }
  return null;
}

function preferredOllamaModel(models) {
  const order = [/^llama3\.2:(3b|latest)$/, /^llama3/, /^qwen/, /^gemma/, /^(mistral|phi)/];
  for (const re of order) {
    const hit = models.find((m) => re.test(m));
    if (hit) return hit;
  }
  return models[0];
}

const searchNote = () => (found.searchModel ? "Nothing to download." : "Only the 34 MB search model needs downloading.");

function foundItem(title, detail, buttonText, onClick) {
  const li = el("li", { class: "found-item" });
  const text = el("div");
  text.append(el("b", {}, title), el("span", { class: "small" }, detail));
  const btn = el("button", { type: "button" }, buttonText);
  btn.addEventListener("click", onClick);
  li.append(text, btn);
  return li;
}

function renderFound() {
  const items = [];
  if (found.webllm) {
    items.push(foundItem(
      `${LLM_MODEL_LABEL}, downloaded from RAGScope before`,
      `Saved in this browser from an earlier visit. ${searchNote()}`,
      "Use it", () => enterMode("webllm")));
  }
  if (found.chrome) {
    items.push(foundItem(
      "Your browser's built-in AI",
      `Your browser already has its own AI model installed. ${searchNote()}`,
      "Use it", () => enterMode("chrome")));
  }
  $("found-list").replaceChildren(...items);
  $("found-status").textContent = items.length
    ? "Good news: you can use one of these without the big download."
    : "No AI models found in this browser yet.";
}

async function checkOllamaClicked() {
  const status = $("ollama-status");
  status.replaceChildren(el("span", {}, "Checking for Ollama on this computer…"));
  $("btn-ollama").disabled = true;
  const result = await findOllama();
  $("btn-ollama").disabled = false;

  if (!result) {
    status.replaceChildren(
      el("span", {}, "Couldn't reach Ollama. If it isn't installed, get it from ollama.com. If it's running, it needs permission to talk to this website. On a Mac, run this in Terminal, then quit and reopen Ollama:"),
      el("pre", {}, `launchctl setenv OLLAMA_ORIGINS "${location.origin}"`),
      el("span", {}, "On Windows or Linux, set the environment variable OLLAMA_ORIGINS to the same address, then restart Ollama.")
    );
    return;
  }
  if (!result.models.length) {
    status.replaceChildren(el("span", {}, "Ollama is running but has no chat models. Run this in a terminal, then check again:"), el("pre", {}, "ollama pull llama3.2"));
    return;
  }
  const select = el("select", { "aria-label": "Ollama model" });
  for (const m of result.models) select.append(el("option", { value: m }, m));
  select.value = preferredOllamaModel(result.models);
  const use = el("button", { type: "button" }, "Use Ollama");
  use.addEventListener("click", () => {
    ollama = { base: result.base, model: select.value };
    enterMode("ollama");
  });
  const row = el("div");
  row.append(select, document.createTextNode(" "), use);
  status.replaceChildren(el("span", {}, `Found Ollama with ${result.models.length} model(s). Questions go only to the Ollama app on this computer. ${searchNote()}`), row);
}

/* ---------- loading models ---------- */

async function loadSearchModel() {
  if (extractor) return;
  setLoading(found.searchModel ? "Loading the search model from your browser's cache…" : "Downloading the search model (about 34 MB)…", 0);
  const { pipeline, env } = await import("./vendor/transformers.min.js");
  env.allowLocalModels = false;
  env.backends.onnx.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
  const files = {};
  extractor = await pipeline("feature-extraction", EMBED_MODEL, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files[p.file] = [p.loaded, p.total];
        const [l, t] = Object.values(files).reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
        setLoading(`Loading the search model… ${Math.round((l / t) * 100)}%`, (l / t) * 0.05);
      }
    },
  });
  found.searchModel = true;
}

async function startLive() {
  await loadSearchModel();
  setInputs({ typing: true, examples: true }); // steps 1-3 work while the language model gets ready

  if (mode === "webllm") {
    setLoading(found.webllm ? "Loading the language model from your browser's cache…" : `Downloading the language model (about ${llmChoice.size}). This only happens once.`, 0.05);
    const webllm = await import("./vendor/web-llm.js");
    llmReady = webllm.CreateMLCEngine(llmChoice.id, {
      initProgressCallback: (r) => setLoading(`Loading the language model… ${Math.round(r.progress * 100)}%`, 0.05 + r.progress * 0.95),
    }).then((e) => { engine = e; found.webllm = true; });
  } else {
    llmReady = Promise.resolve(); // built-in AI and Ollama are already installed
  }
  await llmReady;
  setLoading(null);
  const where = mode === "ollama" ? "the Ollama app on this computer" : "your device";
  $("mode-label").textContent = `Live mode: answers by ${llmLabel()}, running on ${where}. Your questions stay on this device.`;
}

/* ---------- generating answers (one function per backend) ---------- */

async function generateWebLLM(prompt, onText) {
  const stream = await engine.chat.completions.create({
    messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: prompt }],
    temperature: 0.2,
    max_tokens: 400,
    stream: true,
    stream_options: { include_usage: true },
  });
  let usage = null;
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content;
    if (text) onText(text);
    if (chunk.usage) usage = { input: chunk.usage.prompt_tokens, output: chunk.usage.completion_tokens };
  }
  return usage;
}

async function generateChrome(prompt, onText) {
  const session = await LanguageModel.create({
    ...CHROME_AI_OPTIONS,
    initialPrompts: [{ role: "system", content: SYSTEM_PROMPT }],
  });
  try {
    let soFar = "";
    for await (const chunk of session.promptStreaming(prompt)) {
      // Older Chrome versions sent the whole text so far; newer ones send only the new part.
      const piece = chunk.startsWith(soFar) && soFar ? chunk.slice(soFar.length) : chunk;
      soFar += piece;
      if (piece) onText(piece);
    }
  } finally {
    session.destroy();
  }
  return null;
}

async function generateOllama(prompt, onText) {
  const res = await fetch(`${ollama.base}/api/chat`, {
    method: "POST",
    body: JSON.stringify({
      model: ollama.model,
      stream: true,
      options: { temperature: 0.2, num_predict: 400 },
      messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: prompt }],
    }),
  });
  if (!res.ok) throw new Error(`Ollama replied with error ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", usage = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines.filter(Boolean)) {
      const msg = JSON.parse(line);
      if (msg.error) throw new Error(msg.error);
      if (msg.message?.content) onText(msg.message.content);
      if (msg.done) usage = { input: msg.prompt_eval_count, output: msg.eval_count };
    }
  }
  return usage;
}

/* ---------- asking ---------- */

async function askLive(question) {
  resetSteps();
  setStep("embed", "active");
  const t0 = performance.now();
  const [qvec] = (await extractor([QUERY_PREFIX + question], { pooling: "cls", normalize: true })).tolist();
  const [x, y] = pcaProject(qvec, kb.pca);
  showEmbed(qvec, { x, y }, Math.round(performance.now() - t0));

  const scores = cosineScores(qvec, kb.vectors);
  const topIds = [...scores.keys()].sort((a, b) => scores[b] - scores[a]).slice(0, TOP_K);
  showRetrieve(scores, topIds);

  const prompt = buildPrompt(question, topIds.map((i) => kb.chunks[i]));
  showPrompt(prompt);

  if (mode === "webllm" && !engine) $("answer").textContent = "Waiting for the language model to finish loading…";
  await llmReady;
  const t1 = performance.now();
  let firstMs = null, answer = "";
  const onText = (text) => {
    firstMs ??= Math.round(performance.now() - t1);
    answer += text;
    renderAnswer(answer, true);
  };
  const generate = { webllm: generateWebLLM, chrome: generateChrome, ollama: generateOllama }[mode];
  const usage = await generate(prompt, onText);
  renderAnswer(answer, false);
  $("t-gen").textContent = `first token after ${firstMs} ms`;
  const secs = ((performance.now() - t1) / 1000).toFixed(1);
  $("stats").textContent = usage
    ? `${llmLabel()} read ${usage.input} tokens and wrote ${usage.output} tokens in ${secs} s, on your device.`
    : `${llmLabel()} answered in ${secs} s, on your device.`;
  setStep("generate", "done");
}

async function askDemo(example) {
  resetSteps();
  setStep("embed", "active");
  await sleep(500);
  showEmbed(example.vector, example.point, null);
  await sleep(900);
  showRetrieve(example.scores, example.top_ids);
  await sleep(900);
  showPrompt(example.prompt);
  await sleep(700);
  let shown = "";
  for (const w of example.answer.split(/(\s+)/)) {
    shown += w;
    renderAnswer(shown, true);
    await sleep(w.trim() ? 35 : 0);
  }
  renderAnswer(example.answer, false);
  $("t-gen").textContent = "recorded";
  $("stats").textContent = `Recorded from ${example.model}: it read ${example.input_tokens} tokens and wrote ${example.output_tokens} tokens.`;
  setStep("generate", "done");
}

/* ---------- choosing a mode ---------- */

function saveMode() {
  store.set(mode === "ollama" ? `ollama|${ollama.base}|${ollama.model}` : mode);
}

async function stopLive() {
  try { await engine?.unload(); } catch {}
  engine = null; llmReady = null;
}

async function enterMode(next) {
  await stopLive();
  mode = next;
  saveMode();
  $("consent").close();
  $("modebar").hidden = false;
  $("btn-delete").hidden = next === "demo";
  resetSteps();
  if (next === "demo") {
    $("mode-label").textContent = "Demo mode: recorded answers, nothing downloaded. Pick an example question.";
    $("q").placeholder = "Demo mode: pick an example question below";
    setInputs({ typing: false, examples: true });
    return;
  }
  $("mode-label").textContent = "Live mode: getting the AI ready…";
  $("q").placeholder = "e.g. Why do AI models make things up?";
  setInputs({ typing: false, examples: false });
  try {
    await startLive();
  } catch (err) {
    console.error(err);
    setLoading(null);
    $("mode-label").textContent = "The AI couldn't start on this device. Click Change to pick another option or the demo.";
    showError(`The AI couldn't load: ${err.message || err}`);
    setInputs({ typing: false, examples: false });
  }
}

function openConsent() {
  $("agree").checked = false;
  $("btn-live").disabled = true;
  renderFound();
  $("consent").showModal();
}

async function deleteModels() {
  if (!confirm("Delete the AI models this site saved in your browser? (This doesn't touch Ollama or your browser's built-in AI.)")) return;
  await stopLive();
  extractor = null;
  try { for (const name of await caches.keys()) await caches.delete(name); } catch {}
  try { for (const db of (await indexedDB.databases?.()) || []) if (db.name) indexedDB.deleteDatabase(db.name); } catch {}
  found.searchModel = false;
  found.webllm = false;
  store.set(null);
  mode = null;
  setInputs({ typing: false, examples: false });
  $("modebar").hidden = true;
  alert("Downloaded models deleted.");
  openConsent();
}

/** On a return visit, go straight back to the visitor's last choice, but only if it
 *  needs no new download. Otherwise ask again. */
async function resumeSavedChoice() {
  const saved = store.get() || "";
  if (saved === "demo") return enterMode("demo"), true;
  if ((saved === "webllm" || saved === "live") && gpu.ok && found.webllm && found.searchModel) return enterMode("webllm"), true;
  if (saved === "chrome" && found.chrome && found.searchModel) return enterMode("chrome"), true;
  if (saved.startsWith("ollama|") && found.searchModel) {
    const [, base, model] = saved.split("|");
    const result = await findOllama(); // they allowed this on an earlier visit
    if (result?.base === base && result.models.includes(model)) {
      ollama = { base, model };
      return enterMode("ollama"), true;
    }
  }
  return false;
}

/* ---------- wiring ---------- */

$("agree").addEventListener("change", () => ($("btn-live").disabled = !($("agree").checked && gpu.ok)));
$("btn-live").addEventListener("click", () => enterMode("webllm"));
$("btn-demo").addEventListener("click", () => enterMode("demo"));
$("btn-ollama").addEventListener("click", checkOllamaClicked);
$("btn-change").addEventListener("click", () => { if (!busy) openConsent(); });
$("btn-delete").addEventListener("click", deleteModels);
$("consent").addEventListener("cancel", (e) => { if (!mode) e.preventDefault(); }); // must choose before continuing

$("answer").addEventListener("mouseover", (e) => e.target.classList.contains("cite") && highlight(Number(e.target.dataset.rank)));
$("answer").addEventListener("mouseout", (e) => e.target.classList.contains("cite") && highlight(null));

async function run(question) {
  if (busy || !mode) return;
  busy = true;
  setInputs({ typing: false, examples: false });
  try {
    if (mode === "demo") {
      const example = demo.examples.find((e) => e.question === question);
      if (example) await askDemo(example);
    } else {
      await askLive(question);
    }
  } catch (err) {
    console.error(err);
    showError(`Something went wrong: ${err.message || err}`);
  } finally {
    busy = false;
    setInputs({ typing: mode !== "demo" && !!extractor, examples: true });
  }
}

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q.length >= 2) run(q);
});

async function checkGpu() {
  if (!("gpu" in navigator)) return { ok: false, f16: false, reason: "This browser doesn't support WebGPU, which this download option needs. You can still use an AI that's already on your device, or the demo." };
  try {
    // Some browsers never answer this call, so give up after 5 seconds instead of hanging.
    const adapter = await withTimeout(navigator.gpu.requestAdapter(), 5000, null);
    if (!adapter) return { ok: false, f16: false, reason: "WebGPU is turned on, but no usable graphics chip was found." };
    return { ok: true, f16: adapter.features.has("shader-f16"), reason: "" };
  } catch {
    return { ok: false, f16: false, reason: "WebGPU couldn't start on this device." };
  }
}

async function init() {
  [kb, demo] = await Promise.all([
    fetch("./data/kb.json").then((r) => r.json()),
    fetch("./data/demo.json").then((r) => r.json()),
  ]);
  document.querySelectorAll(".dimsref").forEach((n) => (n.textContent = kb.dimensions));
  const pct = Math.round((kb.pca.explained[0] + kb.pca.explained[1]) * 100);
  $("kbinfo").textContent = `Knowledge base: ${kb.docs.length} documents split into ${kb.chunks.length} passages. The 2D map keeps about ${pct}% of the variation in the original ${kb.dimensions} dimensions.`;
  setupMap();
  drawMap();
  drawVector(new Array(kb.dimensions).fill(0));
  $("examples").replaceChildren(
    ...demo.examples.map(({ question }) => {
      const b = el("button", { type: "button" }, question);
      b.addEventListener("click", () => { $("q").value = question; run(question); });
      return b;
    })
  );
  setInputs({ typing: false, examples: false });

  // Silent checks: they only read this browser's own storage and capabilities. No network.
  gpu = await checkGpu();
  llmChoice = gpu.f16 ? LLM_F16 : LLM_F32;
  [found.searchModel, found.webllm, found.chrome] = await Promise.all([searchModelCached(), webllmCached(llmChoice.id), chromeAIAvailable()]);

  $("dl-size").textContent = llmChoice.size;
  $("dl-llm").textContent = llmChoice.size;
  $("dl-mem").textContent = llmChoice.memory;
  if (gpu.ok) {
    $("device-check").textContent = "Your browser supports WebGPU, so this should work.";
  } else {
    $("device-check").textContent = gpu.reason;
    $("choice-live").classList.add("unavailable");
    $("agree").disabled = true;
  }

  if (!(await resumeSavedChoice())) openConsent();
}

init().catch((err) => {
  console.error(err);
  $("kbinfo").textContent = "Could not load the site's data. Please refresh the page.";
});

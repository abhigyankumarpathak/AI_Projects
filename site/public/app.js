import { EMBED_MODEL, QUERY_PREFIX, SYSTEM_PROMPT, LLM_MODEL_LABEL, buildPrompt, cosineScores, project as pcaProject } from "./shared.js";

const LLM_F16 = { id: "Llama-3.2-3B-Instruct-q4f16_1-MLC", size: "1.8 GB", memory: "2.5 GB" };
const LLM_F32 = { id: "Llama-3.2-3B-Instruct-q4f32_1-MLC", size: "1.8 GB", memory: "3 GB" }; // for GPUs without 16-bit float support
const MODE_KEY = "ragscope-mode";
const TOP_K = 4;
const PALETTE = ["#4f46e5", "#d9622b", "#0f9d76", "#c2367a", "#2b7bd9", "#b58b00", "#7a4fd1", "#5f7a2b"];
const SVG_NS = "http://www.w3.org/2000/svg";
const W = 400, H = 300, PAD = 20;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let kb, demo, toSvg;
let mode = null;            // "live" | "demo"
let gpu = { ok: false, f16: false, reason: "" };
let llmChoice = LLM_F16;
let extractor = null;       // transformers.js embedding pipeline (live mode)
let engine = null;          // WebLLM engine (live mode)
let llmReady = null;        // promise that resolves when the language model is loaded
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

/* ---------- drawing (same visuals as the Python version) ---------- */

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

/* ---------- the four steps, shared by live and demo ---------- */

function resetSteps() {
  ["embed", "retrieve", "prompt", "generate"].forEach((s) => setStep(s, null));
  ["t-embed", "t-prompt", "t-gen", "stats"].forEach((id) => ($(id).textContent = ""));
  $("answer").classList.remove("error");
  renderAnswer("", false);
  $("demo-note").hidden = mode !== "demo";
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

function showDone(stats) {
  $("stats").textContent = stats;
  setStep("generate", "done");
}

/* ---------- demo mode: replay recorded data, downloads nothing ---------- */

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
  const words = example.answer.split(/(\s+)/);
  let shown = "";
  for (const w of words) {
    shown += w;
    renderAnswer(shown, true);
    await sleep(w.trim() ? 35 : 0);
  }
  renderAnswer(example.answer, false);
  $("t-gen").textContent = "recorded";
  showDone(`Recorded from ${example.model}: it read ${example.input_tokens} tokens and wrote ${example.output_tokens} tokens.`);
}

/* ---------- live mode: real models in the browser ---------- */

async function checkGpu() {
  if (!("gpu" in navigator)) return { ok: false, f16: false, reason: "This browser doesn't support WebGPU, which is needed to run the AI. Try a recent Chrome, Edge or Safari on a laptop or desktop." };
  try {
    // Some browsers never answer this call, so give up after 5 seconds instead of hanging.
    const adapter = await Promise.race([navigator.gpu.requestAdapter(), sleep(5000).then(() => null)]);
    if (!adapter) return { ok: false, f16: false, reason: "WebGPU is turned on, but no usable graphics chip was found." };
    return { ok: true, f16: adapter.features.has("shader-f16"), reason: "" };
  } catch {
    return { ok: false, f16: false, reason: "WebGPU couldn't start on this device." };
  }
}

async function startLive() {
  setLoading("Loading the search model (about 34 MB)…", 0);
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

  setLoading(`Loading the language model (about ${llmChoice.size}). This only happens on the first visit.`, 0.05);
  setInputs({ typing: true, examples: true }); // steps 1-3 already work while the big model downloads
  const webllm = await import("./vendor/web-llm.js");
  llmReady = webllm
    .CreateMLCEngine(llmChoice.id, {
      initProgressCallback: (r) => setLoading(`Loading the language model… ${Math.round(r.progress * 100)}%`, 0.05 + r.progress * 0.95),
    })
    .then((e) => {
      engine = e;
      setLoading(null);
      $("mode-label").textContent = `Live mode: AI running on your device (${LLM_MODEL_LABEL}). Your questions stay on this device.`;
      return e;
    });
  await llmReady;
}

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

  if (!engine) $("answer").textContent = "Waiting for the language model to finish downloading…";
  await llmReady;
  const t1 = performance.now();
  let firstMs = null, answer = "", usage = null;
  const stream = await engine.chat.completions.create({
    messages: [{ role: "system", content: SYSTEM_PROMPT }, { role: "user", content: prompt }],
    temperature: 0.2,
    max_tokens: 400,
    stream: true,
    stream_options: { include_usage: true },
  });
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content || "";
    if (text) {
      firstMs ??= Math.round(performance.now() - t1);
      answer += text;
      renderAnswer(answer, true);
    }
    if (chunk.usage) usage = chunk.usage;
  }
  renderAnswer(answer, false);
  $("t-gen").textContent = `first token after ${firstMs} ms`;
  const secs = ((performance.now() - t1) / 1000).toFixed(1);
  showDone(usage
    ? `${LLM_MODEL_LABEL} read ${usage.prompt_tokens} tokens and wrote ${usage.completion_tokens} tokens in ${secs} s, on your device.`
    : `Finished in ${secs} s, on your device.`);
}

/* ---------- choosing a mode ---------- */

async function enterMode(next) {
  mode = next;
  store.set(next);
  $("consent").close();
  $("modebar").hidden = false;
  $("btn-delete").hidden = next !== "live";
  resetSteps();
  if (next === "demo") {
    $("mode-label").textContent = "Demo mode: recorded answers, nothing downloaded. Pick an example question.";
    $("q").placeholder = "Demo mode: pick an example question below";
    setInputs({ typing: false, examples: true });
    return;
  }
  $("mode-label").textContent = "Live mode: downloading models…";
  $("q").placeholder = "e.g. Why do AI models make things up?";
  setInputs({ typing: false, examples: false });
  try {
    await startLive();
  } catch (err) {
    console.error(err);
    setLoading(null);
    $("mode-label").textContent = "Live mode couldn't start on this device. Click Change to try the demo.";
    showError(`The AI couldn't load: ${err.message || err}. Your device may not have enough memory or graphics support.`);
    setInputs({ typing: false, examples: false });
  }
}

function openConsent() {
  $("agree").checked = false;
  $("btn-live").disabled = true;
  $("consent").showModal();
}

async function deleteModels() {
  if (!confirm("Delete the downloaded AI models from this browser? You'll need to download them again to use live mode.")) return;
  try { await engine?.unload(); } catch {}
  engine = null; extractor = null; llmReady = null;
  try {
    for (const name of await caches.keys()) await caches.delete(name);
  } catch {}
  try {
    for (const db of (await indexedDB.databases?.()) || []) if (db.name) indexedDB.deleteDatabase(db.name);
  } catch {}
  store.set(null);
  mode = null;
  setInputs({ typing: false, examples: false });
  $("modebar").hidden = true;
  alert("Downloaded models deleted.");
  openConsent();
}

/* ---------- wiring ---------- */

$("agree").addEventListener("change", () => ($("btn-live").disabled = !($("agree").checked && gpu.ok)));
$("btn-live").addEventListener("click", () => enterMode("live"));
$("btn-demo").addEventListener("click", () => enterMode("demo"));
$("btn-change").addEventListener("click", () => {
  if (busy) return;
  if (mode === "live" && engine) {
    // Free the graphics memory. Downloaded files stay cached until "Delete downloaded models".
    engine.unload().catch(() => {});
    engine = null; llmReady = null; extractor = null;
  }
  openConsent();
});
$("btn-delete").addEventListener("click", deleteModels);
$("consent").addEventListener("cancel", (e) => { if (!mode) e.preventDefault(); }); // must choose before continuing

$("answer").addEventListener("mouseover", (e) => e.target.classList.contains("cite") && highlight(Number(e.target.dataset.rank)));
$("answer").addEventListener("mouseout", (e) => e.target.classList.contains("cite") && highlight(null));

async function run(question) {
  if (busy || !mode) return;
  busy = true;
  const typing = !$("q").disabled;
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
    setInputs({ typing: mode === "live" && typing, examples: true });
  }
}

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q.length >= 2) run(q);
});

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

  gpu = await checkGpu();
  llmChoice = gpu.f16 ? LLM_F16 : LLM_F32;
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

  // Returning visitors skip the question only if nothing new needs downloading.
  // If the browser cleared its cache, we ask again rather than silently re-download.
  const saved = store.get();
  if (saved === "live" && gpu.ok && (await modelIsCached())) enterMode("live");
  else if (saved === "demo") enterMode("demo");
  else openConsent();
}

async function modelIsCached() {
  try {
    const { hasModelInCache } = await import("./vendor/web-llm.js"); // our own copy, no model download
    return await hasModelInCache(llmChoice.id);
  } catch {
    return false;
  }
}

init().catch((err) => {
  console.error(err);
  $("kbinfo").textContent = "Could not load the site's data. Please refresh the page.";
});

"use strict";

const EXAMPLES = [
  "What is a token?",
  "How does attention work in a transformer?",
  "Why do AI models hallucinate?",
  "What does temperature do?",
  "How is a model trained?",
];
const PALETTE = ["#4f46e5", "#d9622b", "#0f9d76", "#c2367a", "#2b7bd9", "#b58b00", "#7a4fd1", "#5f7a2b"];
const SVG_NS = "http://www.w3.org/2000/svg";
const W = 400, H = 300, PAD = 20;

const $ = (id) => document.getElementById(id);
let kb = null;          // knowledge base metadata from /api/knowledge
let project = null;     // maps PCA coords -> svg coords
let currentTop = [];    // chunk ids retrieved for the current question
let queryPoint = null;  // 2D position of the current question

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

/* ---------- initial load ---------- */

async function loadKnowledge() {
  const res = await fetch("/api/knowledge");
  kb = await res.json();
  $("dims").textContent = kb.dimensions;
  document.querySelectorAll(".dimsref").forEach((n) => (n.textContent = kb.dimensions));
  $("llm").textContent = kb.llm_model;
  const pct = Math.round((kb.pca_explained[0] + kb.pca_explained[1]) * 100);
  $("kbinfo").textContent =
    `Knowledge base: ${kb.docs.length} documents split into ${kb.chunks.length} passages. ` +
    `The 2D map keeps about ${pct}% of the variation in the original ${kb.dimensions} dimensions.` +
    (kb.llm_ready ? "" : " No API key is set, so steps 1 to 3 will work but step 4 is off.");

  const xs = kb.chunks.map((c) => c.x), ys = kb.chunks.map((c) => c.y);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  project = (x, y) => [
    clamp(PAD + ((x - minX) / (maxX - minX || 1)) * (W - 2 * PAD), 6, W - 6),
    clamp(H - PAD - ((y - minY) / (maxY - minY || 1)) * (H - 2 * PAD), 6, H - 6),
  ];
  drawMap();
  drawVector(new Array(kb.dimensions).fill(0));

  $("legend").replaceChildren(
    ...kb.docs.map((d, i) => {
      const s = el("span");
      s.append(el("i", { style: `background:${PALETTE[i % PALETTE.length]}` }), d.split(":")[0]);
      return s;
    })
  );
}

function drawMap(queryPoint, topIds = []) {
  const svg = $("map");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.replaceChildren();
  const top = new Set(topIds);
  let qx, qy;
  if (queryPoint) {
    [qx, qy] = project(queryPoint.x, queryPoint.y);
    for (const id of topIds) {
      const [x, y] = project(kb.chunks[id].x, kb.chunks[id].y);
      svg.append(el("svg:line", { class: "link", x1: qx, y1: qy, x2: x, y2: y }));
    }
  }
  for (const c of kb.chunks) {
    const [x, y] = project(c.x, c.y);
    const color = PALETTE[kb.docs.indexOf(c.doc) % PALETTE.length];
    const dot = el("svg:circle", {
      cx: x, cy: y, r: top.has(c.id) ? 7 : 4.5, fill: color,
      opacity: topIds.length && !top.has(c.id) ? 0.3 : 0.9, "data-id": c.id,
    });
    dot.append(el("svg:title", {}, `${c.doc} > ${c.section}`));
    svg.append(dot);
  }
  if (queryPoint) {
    const r1 = 11, r2 = 4.5, pts = [];
    for (let i = 0; i < 10; i++) {
      const r = i % 2 ? r2 : r1, a = (Math.PI / 5) * i - Math.PI / 2;
      pts.push(`${qx + r * Math.cos(a)},${qy + r * Math.sin(a)}`);
    }
    const star = el("svg:polygon", { class: "star", points: pts.join(" ") });
    star.append(el("svg:title", {}, "Your question"));
    svg.append(star);
  }
}

function drawVector(vec) {
  const max = Math.max(...vec.map(Math.abs)) || 1;
  const styles = getComputedStyle(document.documentElement);
  const pos = styles.getPropertyValue("--pos").trim(), neg = styles.getPropertyValue("--neg").trim();
  $("vector").replaceChildren(
    ...vec.map((v, i) => {
      const s = el("span", { title: `dimension ${i}: ${v.toFixed(3)}` });
      if (v !== 0) {
        s.style.background = v > 0 ? pos : neg;
        s.style.opacity = (0.15 + 0.85 * Math.abs(v) / max).toFixed(2);
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

/* ---------- citations in the streamed answer ---------- */

function renderAnswer(text, streaming) {
  const box = $("answer");
  box.replaceChildren();
  const parts = text.split(/(\[\d+\])/g);
  for (const part of parts) {
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
$("answer").addEventListener("mouseover", (e) => e.target.classList.contains("cite") && highlight(Number(e.target.dataset.rank)));
$("answer").addEventListener("mouseout", (e) => e.target.classList.contains("cite") && highlight(null));

/* ---------- asking ---------- */

async function ask(question) {
  $("go").disabled = true;
  ["embed", "retrieve", "prompt", "generate"].forEach((s) => setStep(s, null));
  ["t-embed", "t-prompt", "t-gen", "stats"].forEach((id) => ($(id).textContent = ""));
  $("answer").classList.remove("error");
  renderAnswer("", false);
  setStep("embed", "active");

  let answer = "";
  let promptChars = 0;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop();
      for (const frame of frames) {
        if (!frame.startsWith("data: ")) continue;
        const ev = JSON.parse(frame.slice(6));
        switch (ev.stage) {
          case "embed":
            drawVector(ev.vector);
            $("t-embed").textContent = `${ev.ms} ms`;
            drawMap(ev.point);
            setStep("embed", "done");
            setStep("retrieve", "active");
            queryPoint = ev.point;
            break;
          case "retrieve":
            currentTop = ev.top_ids;
            drawMap(queryPoint, ev.top_ids);
            drawScores(ev.scores, ev.top_ids);
            setStep("retrieve", "done");
            setStep("prompt", "active");
            break;
          case "prompt":
            $("system").textContent = ev.system;
            $("prompt").textContent = ev.prompt;
            promptChars = ev.system.length + ev.prompt.length;
            $("t-prompt").textContent = `${promptChars.toLocaleString()} characters ≈ ${Math.round(promptChars / 4).toLocaleString()} tokens`;
            setStep("prompt", "done");
            setStep("generate", "active");
            break;
          case "token":
            answer += ev.text;
            renderAnswer(answer, true);
            break;
          case "done":
            renderAnswer(answer, false);
            $("t-gen").textContent = `first token after ${ev.first_token_ms} ms`;
            $("stats").textContent =
              `${ev.model} read ${ev.input_tokens.toLocaleString()} tokens and wrote ${ev.output_tokens.toLocaleString()} tokens in ${(ev.total_ms / 1000).toFixed(1)} s.`;
            setStep("generate", "done");
            break;
          case "error":
            $("answer").classList.add("error");
            $("answer").classList.remove("caret");
            $("answer").textContent = ev.message;
            setStep("generate", "done");
            break;
        }
      }
    }
  } catch (err) {
    $("answer").classList.add("error");
    $("answer").textContent = `Something went wrong: ${err.message}`;
  } finally {
    $("go").disabled = false;
  }
}

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q.length >= 2) ask(q);
});
$("examples").replaceChildren(
  ...EXAMPLES.map((q) => {
    const b = el("button", { type: "button" }, q);
    b.addEventListener("click", () => { $("q").value = q; ask(q); });
    return b;
  })
);

loadKnowledge().catch(() => ($("kbinfo").textContent = "Could not load the knowledge base. Is the server running?"));

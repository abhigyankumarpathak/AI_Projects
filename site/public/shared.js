// Used by both the browser (app.js) and the build script (build.mjs).

export const EMBED_MODEL = "Xenova/bge-small-en-v1.5";
// bge models retrieve better when questions carry this instruction.
export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
export const LLM_MODEL_LABEL = "Llama 3.2 3B";

export const SYSTEM_PROMPT = `You are a friendly tutor on a website that teaches students how AI works.
Answer the student's question using ONLY the numbered sources provided in their message.
Cite sources inline like [1] or [2] right after the sentence they support.
If the sources don't contain the answer, say so plainly instead of guessing.
Write for a curious 14-18 year old: clear, concrete, at most about 180 words, plain paragraphs (no headings).`;

/** The "augmentation" step: paste retrieved passages into the prompt. */
export function buildPrompt(question, sources) {
  const blocks = sources.map((s, i) => `[${i + 1}] ${s.doc} > ${s.section}\n${s.text}`);
  return "Sources:\n\n" + blocks.join("\n\n") + `\n\nStudent's question: ${question}`;
}

/** Cosine similarity of one vector against many. */
export function cosineScores(query, vectors) {
  const norm = (v) => Math.hypot(...v);
  const qn = norm(query);
  return vectors.map((v) => v.reduce((a, x, i) => a + x * query[i], 0) / (qn * norm(v)));
}

/** Project a vector onto the 2D PCA map. */
export function project(vec, pca) {
  return pca.components.map((comp) => comp.reduce((a, c, j) => a + c * (vec[j] - pca.mean[j]), 0));
}

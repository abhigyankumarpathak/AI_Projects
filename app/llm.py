"""The generation half of RAG: build the prompt and stream Claude's answer."""

from __future__ import annotations

import os
from collections.abc import Iterator

import anthropic

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("CLAUDE_EFFORT", "low")  # short Q&A answers don't need deep thinking

SYSTEM_PROMPT = """You are a friendly tutor on a website that teaches students how AI works.
Answer the student's question using ONLY the numbered sources provided in their message.
Cite sources inline like [1] or [2] right after the sentence they support.
If the sources don't contain the answer, say so plainly instead of guessing.
Write for a curious 14-18 year old: clear, concrete, at most about 180 words, plain paragraphs (no headings)."""


def build_prompt(question: str, sources: list[dict]) -> str:
    """The 'augmentation' step: paste retrieved chunks into the prompt."""
    blocks = [
        f"[{n}] {s['doc']} > {s['section']}\n{s['text']}" for n, s in enumerate(sources, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nStudent's question: {question}"


def has_credentials() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def stream_answer(prompt: str) -> Iterator[dict]:
    """Yield {"type": "text", "text": ...} events, then one {"type": "usage", ...} event."""
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        output_config={"effort": EFFORT},
        # If a safety classifier declines, the API retries on a suitable fallback model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield {"type": "text", "text": text}
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield {"type": "text", "text": "\n\n(The model declined to answer this question.)"}
    yield {
        "type": "usage",
        "model": final.model,
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
        "stop_reason": final.stop_reason,
    }

"""The generation half of RAG: build the prompt and stream an answer.

Two providers:
  * "ollama"    - free, runs a model on your own computer (https://ollama.com)
  * "anthropic" - Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
LLM_PROVIDER=auto (the default) uses Claude if a key is set, otherwise Ollama.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import anthropic
import httpx

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")  # short Q&A answers don't need deep thinking
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

SYSTEM_PROMPT = """You are a friendly tutor on a website that teaches students how AI works.
Answer the student's question using ONLY the numbered sources provided in their message.
Cite sources inline like [1] or [2] right after the sentence they support.
If the sources don't contain the answer, say so plainly instead of guessing.
Write for a curious 14-18 year old: clear, concrete, at most about 180 words, plain paragraphs (no headings)."""


class LLMUnavailable(Exception):
    """Raised with a student-friendly message when no model can be reached."""


def build_prompt(question: str, sources: list[dict]) -> str:
    """The 'augmentation' step: paste retrieved chunks into the prompt."""
    blocks = [
        f"[{n}] {s['doc']} > {s['section']}\n{s['text']}" for n, s in enumerate(sources, start=1)
    ]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nStudent's question: {question}"


def has_anthropic_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def provider() -> str:
    choice = os.getenv("LLM_PROVIDER", "auto").lower()
    if choice in ("anthropic", "ollama"):
        return choice
    return "anthropic" if has_anthropic_key() else "ollama"


def model_name() -> str:
    return CLAUDE_MODEL if provider() == "anthropic" else f"{OLLAMA_MODEL} (local, via Ollama)"


def stream_answer(prompt: str) -> Iterator[dict]:
    """Yield {"type": "text", "text": ...} events, then one {"type": "usage", ...} event."""
    if provider() == "anthropic":
        if not has_anthropic_key():
            raise LLMUnavailable("LLM_PROVIDER is anthropic but no ANTHROPIC_API_KEY is set in .env.")
        yield from _stream_claude(prompt)
    else:
        yield from _stream_ollama(prompt)


def _stream_claude(prompt: str) -> Iterator[dict]:
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        output_config={"effort": CLAUDE_EFFORT},
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


def _stream_ollama(prompt: str) -> Iterator[dict]:
    body = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "options": {"temperature": 0.2},  # low temperature = stick closely to the sources
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=body, timeout=httpx.Timeout(10, read=300)) as res:
            if res.status_code == 404:
                raise LLMUnavailable(f"Ollama doesn't have the model '{OLLAMA_MODEL}'. Run: ollama pull {OLLAMA_MODEL}")
            res.raise_for_status()
            for line in res.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMUnavailable(f"Ollama error: {chunk['error']}")
                text = chunk.get("message", {}).get("content", "")
                if text:
                    yield {"type": "text", "text": text}
                if chunk.get("done"):
                    yield {
                        "type": "usage",
                        "model": f"{OLLAMA_MODEL} (local)",
                        "input_tokens": chunk.get("prompt_eval_count", 0),
                        "output_tokens": chunk.get("eval_count", 0),
                        "stop_reason": chunk.get("done_reason", "stop"),
                    }
    except httpx.ConnectError:
        raise LLMUnavailable(
            "Can't reach Ollama. Open the Ollama app (or run `ollama serve`), "
            "or add an ANTHROPIC_API_KEY to .env to use Claude instead."
        ) from None

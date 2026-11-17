"""RAGScope web server. Run with:  uvicorn app.main:app --reload"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from app import llm  # noqa: E402  (needs .env loaded first)
from app.rag import Embedder, KnowledgeBase  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
state: dict = {}


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    k: int = Field(default=4, ge=1, le=8)


def create_app(embedder=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["kb"] = KnowledgeBase(ROOT / "knowledge", embedder or Embedder())
        yield

    app = FastAPI(title="RAGScope", lifespan=lifespan)

    @app.get("/api/knowledge")
    def knowledge():
        kb: KnowledgeBase = state["kb"]
        return {
            "embed_model": getattr(kb.embedder, "model_name", "custom"),
            "dimensions": int(kb.vectors.shape[1]),
            "pca_explained": kb.pca.explained,
            "docs": kb.docs,
            "chunks": [c.__dict__ for c in kb.chunks],
            "llm_model": llm.MODEL,
            "llm_ready": llm.has_credentials(),
        }

    @app.post("/api/ask")
    def ask(body: Question):
        kb: KnowledgeBase = state["kb"]
        question = body.question.strip()
        if len(question) < 2:
            raise HTTPException(422, "Question is too short")

        def events():
            def send(stage: str, **data) -> str:
                return f"data: {json.dumps({'stage': stage, **data})}\n\n"

            t0 = time.perf_counter()
            result = kb.search(question, k=body.k)
            ms = round((time.perf_counter() - t0) * 1000)
            yield send("embed", vector=result["query_vector"], point=result["query_point"], ms=ms)
            yield send("retrieve", scores=result["scores"], top_ids=result["top_ids"])

            sources = [kb.chunks[i].__dict__ for i in result["top_ids"]]
            prompt = llm.build_prompt(question, sources)
            yield send("prompt", system=llm.SYSTEM_PROMPT, prompt=prompt)

            if not llm.has_credentials():
                yield send("error", message="No ANTHROPIC_API_KEY set. Add it to .env to see the generation step.")
                return
            try:
                t1 = time.perf_counter()
                first_token_ms = None
                for event in llm.stream_answer(prompt):
                    if event["type"] == "text":
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - t1) * 1000)
                        yield send("token", text=event["text"])
                    else:
                        event.pop("type")
                        yield send(
                            "done",
                            first_token_ms=first_token_ms,
                            total_ms=round((time.perf_counter() - t1) * 1000),
                            **event,
                        )
            except anthropic.AuthenticationError:
                yield send("error", message="Your Anthropic API key was rejected. Check .env.")
            except anthropic.RateLimitError:
                yield send("error", message="Rate limited by the Anthropic API. Wait a moment and retry.")
            except anthropic.APIStatusError as e:
                yield send("error", message=f"Anthropic API error {e.status_code}.")
            except anthropic.APIConnectionError:
                yield send("error", message="Could not reach the Anthropic API.")

        return StreamingResponse(events(), media_type="text/event-stream")

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    return app


app = create_app()

"""Tests that run offline: a fake embedder replaces the real model and no API key is needed."""

import json

import numpy as np
from fastapi.testclient import TestClient

from app.rag import PCA2D, chunk_markdown, cosine_similarities

VOCAB = ["token", "embedding", "attention", "training", "hallucination", "temperature", "rag", "neuron"]


class FakeEmbedder:
    """Bag-of-keywords vectors: deterministic and instant."""

    model_name = "fake"

    def embed(self, texts):
        rows = [[t.lower().count(w) + 0.01 for w in VOCAB] for t in texts]
        return np.array(rows, dtype=np.float32)


def test_chunk_markdown_splits_sections_and_long_text():
    md = "# Title\n\n## A\nShort.\n\n## B\n" + "This is a sentence. " * 80
    title, chunks = chunk_markdown(md, size=300, overlap=50)
    assert title == "Title"
    assert chunks[0] == ("A", "Short.")
    assert len(chunks) > 3 and all(sec == "B" for sec, _ in chunks[1:])
    assert all(len(text) <= 300 + 50 for _, text in chunks)


def test_cosine_similarity_and_pca():
    m = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    s = cosine_similarities(np.array([1.0, 0.0]), m)
    assert s[0] == 1.0 and abs(s[1]) < 1e-9
    pca = PCA2D.fit(np.random.default_rng(0).normal(size=(20, 5)))
    assert pca.transform(np.zeros(5)).shape == (1, 2)


def test_api_pipeline_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    from app import llm
    from app.main import create_app

    monkeypatch.setattr(llm, "OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens here

    with TestClient(create_app(embedder=FakeEmbedder())) as client:
        kb = client.get("/api/knowledge").json()
        assert len(kb["docs"]) == 8 and kb["chunks"]

        res = client.post("/api/ask", json={"question": "How does attention work?"})
        events = [json.loads(line[6:]) for line in res.text.split("\n\n") if line.startswith("data: ")]
        stages = [e["stage"] for e in events]
        assert stages == ["embed", "retrieve", "prompt", "error"]
        retrieved = [kb["chunks"][i]["text"].lower() for i in events[1]["top_ids"]]
        assert any("attention" in text for text in retrieved)
        assert "[1]" in events[2]["prompt"]
        assert "Ollama" in events[3]["message"]

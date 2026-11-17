"""The retrieval half of RAG: chunking, embedding, similarity search and a 2D map.

Every function here returns plain data so the frontend can visualise each step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small (384-dim), runs locally on CPU, free
CHUNK_CHARS = 700
CHUNK_OVERLAP = 120


@dataclass
class Chunk:
    id: int
    doc: str  # document title, e.g. "Tokens: How AI Reads Text"
    section: str  # the "## heading" the chunk came from
    text: str
    x: float = 0.0  # 2D map coordinates (PCA)
    y: float = 0.0


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~size-char windows on sentence boundaries, with overlap."""
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            pieces.append(current.strip())
            current = current[-overlap:] + " " + sentence  # carry a little context forward
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_markdown(markdown: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> tuple[str, list[tuple[str, str]]]:
    """Return (doc title, [(section heading, chunk text), ...])."""
    title_match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Untitled"
    chunks: list[tuple[str, str]] = []
    for block in re.split(r"^##\s+", markdown, flags=re.MULTILINE)[1:]:
        heading, _, body = block.partition("\n")
        body = " ".join(body.split())
        for piece in _split_long(body, size, overlap):
            chunks.append((heading.strip(), piece))
    return title, chunks


def cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q = query / np.linalg.norm(query)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return m @ q


@dataclass
class PCA2D:
    """Tiny PCA so we can draw 384-dimensional vectors on a flat map."""

    mean: np.ndarray
    components: np.ndarray  # shape (2, dim)
    explained: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "PCA2D":
        mean = matrix.mean(axis=0)
        _, s, vt = np.linalg.svd(matrix - mean, full_matrices=False)
        variance = s**2
        explained = (variance[:2] / variance.sum()).tolist()
        return cls(mean=mean, components=vt[:2], explained=explained)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(vectors) - self.mean) @ self.components.T


class Embedder:
    """Wraps fastembed so tests can swap in a fake embedder."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from fastembed import TextEmbedding  # imported lazily: downloads ~70MB on first use

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Questions get the model's search-instruction prefix, which improves retrieval."""
        return np.array(list(self._model.query_embed(text))[0], dtype=np.float32)


class KnowledgeBase:
    def __init__(self, folder: Path, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        for path in sorted(folder.glob("*.md")):
            title, pieces = chunk_markdown(path.read_text(encoding="utf-8"))
            for section, text in pieces:
                self.chunks.append(Chunk(id=len(self.chunks), doc=title, section=section, text=text))
        if not self.chunks:
            raise ValueError(f"No markdown chunks found in {folder}")
        # Embed "section: text" so headings add a little extra meaning.
        self.vectors = embedder.embed([f"{c.section}: {c.text}" for c in self.chunks])
        self.pca = PCA2D.fit(self.vectors)
        for chunk, (x, y) in zip(self.chunks, self.pca.transform(self.vectors)):
            chunk.x, chunk.y = float(x), float(y)

    @property
    def docs(self) -> list[str]:
        return list(dict.fromkeys(c.doc for c in self.chunks))

    def search(self, question: str, k: int = 4) -> dict:
        embed_query = getattr(self.embedder, "embed_query", None)
        query_vec = embed_query(question) if embed_query else self.embedder.embed([question])[0]
        scores = cosine_similarities(query_vec, self.vectors)
        order = np.argsort(-scores)
        top = [int(i) for i in order[:k]]
        qx, qy = self.pca.transform(query_vec)[0]
        return {
            "query_vector": [round(float(v), 4) for v in query_vec],
            "query_point": {"x": float(qx), "y": float(qy)},
            "scores": [round(float(s), 4) for s in scores],
            "top_ids": top,
        }

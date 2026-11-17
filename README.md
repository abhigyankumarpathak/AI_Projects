# RAGScope

**Watch an AI answer your question, one step at a time.**

RAGScope is a small website that teaches students how AI works by showing what goes on inside a
Retrieval-Augmented Generation (RAG) system. Ask a question such as *"How does attention work?"* and the page walks through each step:

1. **Embed**: your question becomes a 384-number vector, drawn as a colour strip.
2. **Search**: every passage in the knowledge base is plotted on a 2D map. Your question appears as a star, and lines connect it to the closest passages, ranked by cosine similarity.
3. **Build the prompt**: you see the exact text sent to the language model, retrieved sources included.
4. **Generate**: Claude streams an answer token by token, with citations. Hover over a citation to highlight its source on the map.

The knowledge base ([knowledge/](knowledge/)) is a set of short explainers on tokens, embeddings, neural
networks, transformers, training, sampling, RAG and hallucinations. That makes the app self-referential:
it uses RAG to explain RAG.

## How it works

| Step | Tool | Where |
| --- | --- | --- |
| Chunking | Markdown split by `##` section, then ~700-character windows with overlap | [app/rag.py](app/rag.py) |
| Embeddings | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via `fastembed`. Runs locally on the CPU and costs nothing | [app/rag.py](app/rag.py) |
| Vector search | Cosine similarity with NumPy (the dataset is small enough that a vector database isn't needed) | [app/rag.py](app/rag.py) |
| 2D map | PCA, a few lines of NumPy SVD | [app/rag.py](app/rag.py) |
| Generation | Claude via the Anthropic Python SDK, streamed | [app/llm.py](app/llm.py) |
| Server | FastAPI, which streams each pipeline step to the browser with Server-Sent Events | [app/main.py](app/main.py) |
| Frontend | Plain HTML, CSS and JavaScript, with hand-drawn SVG and no build step | [static/](static/) |

## Run it locally

You need Python 3.10 or newer.

```bash
git clone https://github.com/abhigyankumarpathak/AI_Projects.git
cd AI_Projects
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then paste your Anthropic API key into .env
uvicorn app.main:app --reload
```

Open http://localhost:8000. The first start downloads the embedding model (about 70 MB).

Without an API key, steps 1–3 still work, so you can explore embeddings and retrieval for free.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake embedder and no API key, so they run offline in about a second.

## Add your own knowledge

Drop any `.md` file into `knowledge/`. Start it with a `# Title` and split it into `## Sections`, then restart the server. You could add your own class notes and ask questions about them.

## License

MIT. See [LICENSE](LICENSE).

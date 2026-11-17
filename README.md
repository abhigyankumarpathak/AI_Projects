# RAGScope

**Watch an AI answer your question, one step at a time.**

RAGScope is a small website that teaches students how AI works by showing what goes on inside a
Retrieval-Augmented Generation (RAG) system. Ask a question such as *"How does attention work?"* and the page walks through each step:

1. **Embed**: your question becomes a 384-number vector, drawn as a colour strip.
2. **Search**: every passage in the knowledge base is plotted on a 2D map. Your question appears as a star, and lines connect it to the closest passages, ranked by cosine similarity.
3. **Build the prompt**: you see the exact text sent to the language model, retrieved sources included.
4. **Generate**: a language model (Claude, or a free local model through Ollama) streams an answer token by token, with citations. Hover over a citation to highlight its source on the map.

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
| Generation | Claude through the Anthropic Python SDK, or any local model through [Ollama](https://ollama.com) | [app/llm.py](app/llm.py) |
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

cp .env.example .env
uvicorn app.main:app --reload
```

Open http://localhost:8000. The first start downloads the embedding model (about 70 MB).

### Choosing the AI that writes answers

- **Free (Ollama):** install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and leave `ANTHROPIC_API_KEY` empty. The model runs on your own computer. It's slower, and smaller models are less accurate.
- **Claude:** paste an API key from [platform.claude.com](https://platform.claude.com/) into `.env`.

With `LLM_PROVIDER=auto` (the default), the app uses Claude when a key is set and Ollama otherwise.

## Free public website (runs in the visitor's browser)

[site/](site/) is a static version that needs no server and no API key. It's free to host on Render.

- Before downloading anything, it asks the visitor's permission. The consent screen explains the download size (about 1.8 GB), where the files come from, that questions never leave the device, and how to delete the files afterwards.
- **If the visitor agrees:** the search model ([transformers.js](https://huggingface.co/docs/transformers.js)) and the language model, Llama 3.2 3B through [WebLLM](https://webllm.mlc.ai), run on the visitor's own graphics chip using WebGPU.
- **If the visitor declines:** a demo replays all four steps from data recorded earlier with the same models, and nothing is downloaded.
- **Security:** the site serves its own copies of the JavaScript libraries (no third-party script CDNs), and [render.yaml](render.yaml) sets a strict Content-Security-Policy.

```bash
cd site
npm install
npm run build     # embeds the knowledge base and copies the browser libraries
npm run serve     # http://localhost:8080
npm run record    # optional: re-record the demo answers (needs Ollama with llama3.2:3b)
```

To deploy on Render, choose **New → Blueprint** and pick this repo. Render reads [render.yaml](render.yaml). If you edit `knowledge/`, rerun `npm run build` and commit the files in `site/public/data/`.

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

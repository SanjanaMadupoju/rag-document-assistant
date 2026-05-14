# RAG Document Assistant

A fully local Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and ask questions about its contents. No API keys. No internet connection required. Everything runs on your machine via [Ollama](https://ollama.com).

---

## What It Does

1. Upload any PDF document through a browser interface.
2. Ask natural-language questions about the document.
3. Receive grounded answers sourced directly from the PDF — the model will say "I don't know" rather than hallucinate when the answer isn't present.

A sidebar displays the active model names, page count, and chunk count so you always know what the app is working with.

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | 1.57.0 |
| PDF parsing | [pypdf](https://pypdf.readthedocs.io) via `langchain-community` | 6.11.0 |
| Text splitting | `langchain-text-splitters` | 1.1.2 |
| Embeddings | `langchain-ollama` + `nomic-embed-text` | 1.1.0 |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) (CPU) | 1.13.2 |
| LLM | `langchain-ollama` + `llama3.2:1b` | 1.1.0 |
| Ollama client | `ollama` | 0.6.2 |
| HTTP client | `httpx` | 0.28.1 |
| Orchestration | `langchain-core` | 1.3.3 |

**Runtime requirements:** Python 3.10+, [Ollama](https://ollama.com) installed and running locally.

---

## Installation & Setup

### 1. Install Ollama

Download and install Ollama from [ollama.com](https://ollama.com), then pull the two required models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

Start the Ollama server (if it isn't already running):

```bash
ollama serve
```

### 2. Clone the Repository

```bash
git clone https://github.com/SanjanaMadupoju/rag-document-assistant.git
cd rag-document-assistant
```

### 3. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install streamlit==1.57.0 \
            langchain-community==0.4.1 \
            langchain-ollama==1.1.0 \
            langchain-text-splitters==1.1.2 \
            langchain-core==1.3.3 \
            faiss-cpu==1.13.2 \
            ollama==0.6.2 \
            httpx==0.28.1 \
            pypdf==6.11.0
```

### 5. Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How It Works

This app implements a **Retrieval-Augmented Generation (RAG)** pipeline — a technique that grounds an LLM's answers in a specific document rather than relying on its training data alone.

```
PDF Upload
    │
    ▼
Parse pages (PyPDF)
    │
    ▼
Split into overlapping chunks (500 chars, 50-char overlap)
    │
    ▼
Embed each chunk as a vector (nomic-embed-text via Ollama)
    │
    ▼
Store vectors in FAISS (in-memory)
    │
    ▼
User submits a question
    │
    ▼
Embed the question → find the 3 closest chunks in FAISS
    │
    ▼
Inject chunks as context into the prompt
    │
    ▼
LLM (llama3.2:1b) generates a grounded answer
```

**Why RAG instead of just asking the LLM directly?**
A local 1B-parameter model has limited knowledge and can hallucinate. RAG bypasses this by providing the relevant text explicitly — the model only needs to read and summarise, not recall. The prompt instructs the model to say "I don't know based on the document" if the answer isn't in the retrieved chunks.

---

## Known Limitations

**Performance**
- The FAISS index is rebuilt from scratch every time the page reruns. For large documents this can be slow.
- There is no index persistence — closing the app discards all processed data.

**Document Support**
- PDF only. Word documents, plain text, and HTML are not supported.
- Scanned or image-based PDFs (no text layer) will load silently with no content, causing every question to return "I don't know."

**Retrieval Quality**
- Only the 3 most similar chunks are passed to the LLM. Broad questions receive very little context.
- The 500-character chunk size can split sentences mid-thought.
- No conversation memory — each question is independent.

**Model Quality**
- `llama3.2:1b` is a very small model. Swap to a larger model in `LLM_MODEL` at the top of `app.py` for better results.

**Infrastructure**
- Requires Ollama to be running locally.
- No authentication. Do not expose this app publicly without adding an access control layer.

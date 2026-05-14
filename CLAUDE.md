# CLAUDE.md — RAG Document Assistant

## What This Project Is

A fully local Retrieval-Augmented Generation (RAG) app built with Streamlit and Ollama. Users upload a PDF, then ask natural-language questions about it. The app retrieves the most relevant text chunks and passes them as context to a local LLM. No external APIs. No internet required.

The entire application is a single file: `app.py`.

### Project Files

| File | Purpose |
|---|---|
| `app.py` | Entire application — all logic, UI, and RAG pipeline |
| `CLAUDE.md` | This file — coding rules and project context for Claude |
| `README.md` | Public-facing setup and usage documentation |
| `.gitignore` | Excludes `venv/`, `__pycache__/`, `.env`, logs, `.claude/` |
| `.claude/settings.json` | Claude Code hooks: PostToolUse syntax check and activity log |

---

## Tech Stack

### Runtime Dependencies

| Package | Version | Role |
|---|---|---|
| `streamlit` | 1.57.0 | Browser UI, state management, sidebar |
| `langchain-core` | 1.3.3 | LCEL chain composition (`\|` operator), prompt templates, output parsers |
| `langchain-community` | 0.4.1 | `PyPDFLoader` for PDF parsing |
| `langchain-ollama` | 1.1.0 | `OllamaEmbeddings`, `OllamaLLM` wrappers |
| `langchain-text-splitters` | 1.1.2 | `RecursiveCharacterTextSplitter` |
| `faiss-cpu` | 1.13.2 | In-memory vector store and similarity search |
| `ollama` | 0.6.2 | Ollama Python client; `ResponseError` used for model-not-found errors |
| `httpx` | 0.28.1 | Health check ping to `http://localhost:11434` |
| `pypdf` | 6.11.0 | Underlying PDF parser (used via LangChain) |

### Local Models (via Ollama)

- **`nomic-embed-text`** — embedding model, converts text chunks to vectors
- **`llama3.2:1b`** — LLM for answer generation

Model names are defined as module-level constants at the top of `app.py`:

```python
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"
```

Always reference these constants — never hardcode the model name strings elsewhere.

### Infrastructure

- Python 3.10+
- Ollama must be running locally (`ollama serve`) on its default port `11434`
- No database, no external services, no authentication

---

## Coding Rules

### General

- **Single-file project.** All code lives in `app.py`. Do not create additional Python modules unless the file grows beyond ~300 lines and the user explicitly requests a split.
- **No new dependencies** without checking whether an existing package already covers it. `httpx` and `ollama` are already available.
- **No hardcoded model names.** Use `EMBEDDING_MODEL` and `LLM_MODEL` constants defined at the top of `app.py`.

### Streamlit Patterns

- Use `@st.cache_resource` for anything expensive to initialise (model loaders). Do **not** cache the FAISS index or retriever — these are document-specific and must rebuild per upload.
- Use `st.session_state` if per-session state needs to persist across reruns (e.g., conversation history). Do not use module-level variables for this.
- **Clear button / resettable inputs**: Do NOT set `st.session_state[widget_key] = ""` after the widget renders — Streamlit raises a `StreamlitAPIException`. Instead, use a counter key:
  - `st.session_state["input_version"]` (int, initialised to 0) drives the widget key: `key=f"query_input_{st.session_state['input_version']}"`
  - To clear: increment `input_version`, pop `answer`, call `st.rerun()`
  - `st.session_state["answer"]` holds the last LLM response so it persists after the input clears
- The sidebar is populated in two `with st.sidebar:` blocks — one unconditional (models), one inside `if uploaded_file:` (document stats). New sidebar content should follow this pattern.

### Error Handling

- All Ollama-touching operations must be wrapped in try/except. The established pattern is:
  ```python
  except ConnectionError:        # Ollama server down
  except OllamaResponseError:    # model not pulled
  except Exception as e:         # catch-all fallback
  ```
  Every except branch ends with `st.error(...)` then `st.stop()`.
- The `OllamaResponseError` import alias is `from ollama import ResponseError as OllamaResponseError`.
- A health check (`check_ollama()`) runs at page load before the file uploader renders. If Ollama is down, the app stops there.
- Do not add `try/except` around `loader.load()` — it reads a local temp file and Ollama is not involved.
- Do not add `try/except` around `load_embeddings()` or `load_llm()` — these return config objects and make no network calls on construction.

### LangChain

- Use LCEL (the `|` pipe operator) for chain composition. Do not use legacy `LLMChain` or `RetrievalQA`.
- The RAG chain structure is: `{context: retriever | format_docs, input: RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
- `format_docs` is a plain function that joins `doc.page_content` with `"\n\n"`. Keep it simple.

### Style

- No comments unless the reason is non-obvious. Well-named variables are self-documenting.
- No type annotations required except on standalone helper functions (e.g., `check_ollama() -> bool`).
- Keep `warnings.filterwarnings("ignore")` — LangChain emits noisy deprecation warnings that are not actionable here.

---

## Known Limitations

These are documented deficiencies — do not work around them silently. If fixing one, update this file.

1. **FAISS index rebuilt on every rerun.** Every Streamlit interaction re-reads the PDF, re-splits, and re-embeds all chunks. No caching of the vector index exists. Fix: cache the vectorstore in `st.session_state` keyed by file hash.

2. **No conversation memory.** Each question is independent. The LLM does not see prior Q&A. Fix: accumulate `(question, answer)` pairs in `st.session_state` and include them in the prompt.

3. **Scanned PDFs silently fail.** `PyPDFLoader` extracts the text layer only. Image-based PDFs load with empty content; every question returns "I don't know based on the document." Fix: add OCR via `pytesseract` or detect empty pages and warn the user.

4. **Chunk size is small for dense documents.** `chunk_size=500` chars (~80–100 words) frequently splits paragraphs mid-thought. `chunk_overlap=50` chars (~8 words) is too small to bridge boundaries reliably. Better defaults: `chunk_size=1000`, `chunk_overlap=200`.

5. **k=3 retrieval is hardcoded.** Three chunks × 500 chars = ~375 tokens of context per query. This is thin for broad questions. Fix: expose k as a sidebar slider.

6. **No index persistence.** The FAISS index is in-memory only. Restarting the app loses all processed data.

7. **PDF only.** No support for `.txt`, `.docx`, `.md`, or other formats.

8. **No authentication.** Do not deploy publicly without adding an access control layer.

9. **Temp file leak on error.** If an exception is raised before `os.unlink(pdf_path)` at the bottom of `app.py`, the temp file is not cleaned up. Fix: wrap the `if uploaded_file:` block in `try/finally`.

10. **Model names hardcoded in error messages.** The `OllamaResponseError` catch blocks in both the embedding and inference sections hardcode `nomic-embed-text` and `llama3.2:1b` in error strings instead of using `EMBEDDING_MODEL` and `LLM_MODEL` constants. Fix: use f-strings with the constants, e.g. `f"Run: \`ollama pull {EMBEDDING_MODEL}\`"`.

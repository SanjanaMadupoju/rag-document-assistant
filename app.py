# RAG Assistant v1.0
# hook terminal test v7
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM   # ✅ both from Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import httpx
from ollama import ResponseError as OllamaResponseError
import tempfile
import os
import warnings
warnings.filterwarnings("ignore")

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"


def check_ollama() -> bool:
    try:
        response = httpx.get("http://localhost:11434", timeout=3.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError):
        return False

st.title("📄 RAG Document Assistant")
st.caption("100% local — powered by Ollama")

if not check_ollama():
    st.error(
        "Ollama is not running. Start it with `ollama serve` and then refresh this page."
    )
    st.stop()

st.success("App loaded successfully")

with st.sidebar:
    st.header("Session Info")
    st.subheader("Models")
    st.write(f"**Embedding:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{LLM_MODEL}`")

@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

@st.cache_resource
def load_llm():
    return OllamaLLM(model=LLM_MODEL)

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    st.success(f"✅ Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    st.info(f"📦 Split into {len(split_docs)} chunks")

    with st.sidebar:
        st.subheader("Document")
        st.write(f"**Pages:** {len(docs)}")
        st.write(f"**Chunks:** {len(split_docs)}")

    embeddings = load_embeddings()
    llm = load_llm()

    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    except ConnectionError:
        st.error(
            "Lost connection to Ollama while building the index. "
            "Make sure Ollama is still running (`ollama serve`) and re-upload your PDF."
        )
        st.stop()
    except OllamaResponseError as e:
        if "not found" in str(e).lower():
            st.error(
                "The embedding model `nomic-embed-text` is not pulled. "
                "Run: `ollama pull nomic-embed-text` then refresh."
            )
        else:
            st.error(f"Ollama returned an error while embedding: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while processing the document: {e}")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {input}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    if "input_version" not in st.session_state:
        st.session_state["input_version"] = 0

    query = st.text_input(
        "💬 Ask a question about your PDF",
        key=f"query_input_{st.session_state['input_version']}"
    )

    if query:
        with st.spinner("Thinking..."):
            try:
                result = rag_chain.invoke(query)
            except ConnectionError:
                st.error(
                    "Lost connection to Ollama while generating the answer. "
                    "Make sure Ollama is still running and try again."
                )
                st.stop()
            except OllamaResponseError as e:
                if "not found" in str(e).lower():
                    st.error(
                        "The LLM model `llama3.2:1b` is not pulled. "
                        "Run: `ollama pull llama3.2:1b` then refresh."
                    )
                else:
                    st.error(f"Ollama returned an error while generating: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error while generating the answer: {e}")
                st.stop()
        st.session_state["answer"] = result

    if st.session_state.get("answer"):
        st.subheader("Answer")
        st.write(st.session_state["answer"])
        if st.button("Clear"):
            st.session_state["input_version"] += 1
            st.session_state.pop("answer", None)
            st.rerun()

    os.unlink(pdf_path)

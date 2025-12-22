import sys
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.llms import Ollama   # ❌ remove this
# from langchain_classic.chains.retrieval_qa.base import RetrievalQA  # ❌ drop chain for now

import ollama  # ✅ use the lightweight Ollama client directly


# LangChain Document class
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document



# =====================
# Config
# =====================
BASE_DIR = Path(__file__).resolve().parent
SPEECH_DIR = BASE_DIR / "speeches"     # folder with multiple .txt files
PERSIST_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
DEBUG = True


# =====================
# Chunking
# =====================
def simple_chunk_documents(text: str, chunk_size: int, chunk_overlap: int):
    """Minimal character-based chunking."""
    docs = []
    start = 0
    n = len(text)

    if chunk_overlap >= chunk_size:
        chunk_overlap = 0

    while start < n:
        end = start + chunk_size
        chunk_text = text[start:end]
        if not chunk_text.strip():
            break

        docs.append(
            Document(
                page_content=chunk_text,
                metadata={}
            )
        )
        start += chunk_size - chunk_overlap

    return docs


# =====================
# Vector Store
# =====================
def build_vectorstore(
    recreate: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    persist_suffix: str = "default",
):
    if not SPEECH_DIR.exists():
        print(f"Error: speech directory not found at {SPEECH_DIR}")
        sys.exit(1)

    # 👉 isolate vectorstores per experiment
    persist_dir = BASE_DIR / f"chroma_db_{persist_suffix}"
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 👉 clean ONLY this experiment’s DB
    if recreate:
        for p in persist_dir.iterdir():
            if p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p)

    all_docs = []

    for file_path in SPEECH_DIR.glob("*.txt"):
        with open(file_path, encoding="utf8") as f:
            text = f.read()

        chunks = simple_chunk_documents(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for c in chunks:
            c.metadata["source"] = file_path.name

        all_docs.extend(chunks)

    print("Creating HuggingFace embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    chroma = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="ambedkar_speeches",
    )

    try:
        collection_size = chroma._collection.count()
    except Exception:
        collection_size = 0

    if collection_size == 0:
        print("Adding documents to Chroma vectorstore...")
        chroma.add_documents(all_docs)
        print("Index built and persisted.")
    else:
        print(f"Loaded existing collection with {collection_size} embeddings.")

    return chroma



# =====================
# Retriever
# =====================
def build_retriever(chroma):
    return chroma.as_retriever(search_kwargs={"k": TOP_K})


# =====================
# LLM Call
# =====================
def answer_with_ollama(context: str, question: str) -> str:
    prompt = f"""
You are a helpful assistant answering questions about Dr. B.R. Ambedkar's writings.

Answer the question ONLY using the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:
"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            stream=False,   # 🔑 THIS FIXES EVERYTHING
        )
    except Exception as e:
        return f"Error calling Ollama: {e}"

    return response["message"]["content"].strip()



# =====================
# Interactive Loop
# =====================
def interactive_loop(retriever):
    print("\nAmbedkarGPT — multi-document Q&A prototype")
    print("Type 'exit' or 'quit' to stop. Type 'rebuild' to rebuild embeddings.")

    while True:
        query = input("\nYour question: ").strip()
        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if query.lower() == "rebuild":
            print("Rebuilding vectorstore...")
            chroma = build_vectorstore(recreate=True)
            retriever = build_retriever(chroma)
            print("Rebuild complete.")
            continue

        try:
            docs = retriever.get_relevant_documents(query)
        except AttributeError:
            docs = retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)
        answer = answer_with_ollama(context, query)

        print("\n=== Answer ===")
        print(answer)

        if DEBUG:
            sources = sorted(set(doc.metadata.get("source", "") for doc in docs))
            print("\nSources:", ", ".join(sources))


# =====================
# Main
# =====================
def main():
    chroma = build_vectorstore(recreate=False)
    retriever = build_retriever(chroma)
    interactive_loop(retriever)


if __name__ == "__main__":
    main()

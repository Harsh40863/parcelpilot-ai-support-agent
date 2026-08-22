"""
Step 3 — Document ingestion: PDFs → FAISS vector store with metadata

Extracts text from each PDF, splits into token-sized chunks, embeds with a
local HuggingFace model, and saves a FAISS index with per-chunk metadata.

Design decisions:
  - PyMuPDF (fitz) for PDF extraction — fast, no Java/Poppler dependency.
  - RecursiveCharacterTextSplitter with tiktoken-based token counting
    (chunk_size=500 tokens, overlap=50 tokens) so chunks respect
    paragraph/sentence boundaries while staying within embedding-model limits.
  - sentence-transformers/all-MiniLM-L6-v2 via langchain_huggingface — runs
    locally on CPU, no API key needed, 384-dim embeddings.
  - Each chunk carries metadata:
      source   – PDF filename
      version  – e.g. "v3_current", "v2_deprecated", "v4"
      scope    – access-control tag: "northstar", "lumenworks", or "all"
    The chatbot layer will filter on scope so a customer never sees another
    customer's documents.
  - FAISS index saved with save_local() to data/processed/faiss_index/.
  - A test query with northstar-scope filtering runs at the end to verify.
"""

from pathlib import Path

import pymupdf  # PyMuPDF
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FAISS_DIR = PROJECT_ROOT / "data" / "processed" / "faiss_index"

# ── Document metadata registry ──────────────────────────────────────────
# Maps each PDF filename to its version tag and access-control scope.
#   scope="all"        → visible to every customer
#   scope="northstar"  → visible only to Northstar Logistics
#   scope="lumenworks" → visible only to LumenWorks
DOC_METADATA = {
    "01_Support_Policy_v3_CURRENT.pdf": {
        "version": "v3_current",
        "scope": "all",
    },
    "02_Support_Policy_v2_DEPRECATED.pdf": {
        "version": "v2_deprecated",
        "scope": "all",
    },
    "03_Cancellation_and_Service_Credit_SOP_v4.pdf": {
        "version": "v4",
        "scope": "all",
    },
    "04_Product_Operations_Guide_and_Known_Issues.pdf": {
        "version": "v1",
        "scope": "all",
    },
    "05_Northstar_Logistics_Enterprise_Agreement.pdf": {
        "version": "v1",
        "scope": "northstar",
    },
    "06_LumenWorks_Service_Agreement.pdf": {
        "version": "v1",
        "scope": "lumenworks",
    },
}

# ── Embedding model ─────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Helpers ──────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def build_token_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter that measures length in tokens (cl100k_base)
    rather than raw characters, so chunk boundaries are more meaningful
    for policy-style text.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    def token_length(text: str) -> int:
        return len(enc.encode(text))

    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=token_length,
    )


def main():
    # ── 1. Extract & chunk all PDFs ──────────────────────────────────────
    splitter = build_token_splitter()
    all_docs: list[Document] = []

    for filename, meta in DOC_METADATA.items():
        pdf_path = RAW_DIR / filename
        if not pdf_path.exists():
            print(f"  ⚠ Skipping {filename} — file not found")
            continue

        raw_text = extract_pdf_text(pdf_path)
        chunks = splitter.split_text(raw_text)

        for i, chunk in enumerate(chunks):
            all_docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "version": meta["version"],
                        "scope": meta["scope"],
                        "chunk_index": i,
                    },
                )
            )
        print(f"  ✓ {filename}  →  {len(chunks)} chunks  (scope={meta['scope']})")

    print(f"\n  Total chunks: {len(all_docs)}")

    # ── 2. Embed & build FAISS index ─────────────────────────────────────
    print(f"\n  Loading embedding model: {EMBED_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    print("  Building FAISS index ...")
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # ── 3. Save locally ──────────────────────────────────────────────────
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"\n✅ FAISS index saved to {FAISS_DIR}")

    # ── 4. Test: northstar-scoped similarity search ──────────────────────
    print("\n── Test: northstar-scoped similarity search ──")
    test_query = "What are the cancellation terms?"

    # Filter: only docs visible to Northstar (scope="all" OR scope="northstar")
    def northstar_filter(meta: dict) -> bool:
        return meta["scope"] in ("all", "northstar")

    results = vectorstore.similarity_search(
        test_query, k=3, filter=northstar_filter
    )

    for i, doc in enumerate(results, 1):
        src = doc.metadata["source"]
        scope = doc.metadata["scope"]
        ver = doc.metadata["version"]
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{i}] scope={scope}  ver={ver}  src={src}")
        print(f"      {preview}...")
        print()

    # Verify no lumenworks-only docs leaked into northstar results
    leaked = [d for d in results if d.metadata["scope"] == "lumenworks"]
    if leaked:
        print("  ❌ FAIL — LumenWorks-scoped doc appeared in Northstar results!")
    else:
        print("  ✅ PASS — No cross-customer data leakage in filtered search")


if __name__ == "__main__":
    main()

"""Load Apple's 10-K PDF into Pinecone, one chunk per page.

Run once (safe to re-run; pages are upserted by stable IDs):
    uv run ingest.py
"""

import os
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

load_dotenv()

PDF_PATH = "_10-K-2025-As-Filed.pdf"
INDEX_NAME = "apple-10k"
EMBED_MODEL = "llama-text-embed-v2"
EMBED_DIM = 1024


def load_pages(path: str) -> list[Document]:
    """Chunk by page: each non-empty PDF page becomes one Document."""
    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(path), "page": i + 1},
                )
            )
    return docs


def main() -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    docs = load_pages(PDF_PATH)
    print(f"Loaded {len(docs)} pages from {PDF_PATH}")

    vectorstore = PineconeVectorStore(
        index=pc.Index(INDEX_NAME),
        embedding=PineconeEmbeddings(model=EMBED_MODEL),
    )
    ids = [f"page-{d.metadata['page']}" for d in docs]
    vectorstore.add_documents(docs, ids=ids)
    print(f"Upserted {len(docs)} pages into Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    main()

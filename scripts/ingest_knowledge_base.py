#!/usr/bin/env python3
"""
Run this script ONCE (or whenever knowledge base documents change).
It loads nutrition PDFs from knowledge_base/documents/, chunks them,
embeds them, and stores them in ChromaDB.

Usage:
    python scripts/ingest_knowledge_base.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.config import settings
from app.retriever.ingestion import extract_text_from_pdf, chunk_text

DOCUMENTS_DIR = Path("autism_rag_sys/knowledge_base/documents")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def main():
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    logger.info(
        f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
    )
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Using collection: {settings.CHROMA_COLLECTION_NAME}")

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {DOCUMENTS_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s) to ingest")

    all_chunks = []
    for pdf_path in pdf_files:
        logger.info(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, source_name=pdf_path.name)
        all_chunks.extend(chunks)
        logger.info(f"  -> {len(chunks)} chunks created")

    logger.info(f"Total chunks to embed: {len(all_chunks)}")

    BATCH_SIZE = 64
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch
        ]

        embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()

        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    final_count = collection.count()
    logger.info(f"Collection now has {final_count} chunks.")


if __name__ == "__main__":
    main()

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List
from dataclasses import dataclass
import logging

from app.config import settings
from app.models.genetic_profile import GeneticProfile

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class NutritionChunk:
    text: str
    source: str
    chunk_index: int
    similarity_score: float


class NutritionRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("NutritionRetriever initialized.")

    def retrieve(self, profile: GeneticProfile, top_k: int = 6) -> List[NutritionChunk]:
        query = profile.to_retrieval_query()
        logger.info(f"Retrieval query: {query[:200]}")

        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        count = self.collection.count()
        if count == 0:
            logger.warning("ChromaDB collection is empty. Returning 0 retrieved chunks.")
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = round(1 - (dist / 2), 4)
                chunks.append(
                    NutritionChunk(
                        text=doc,
                        source=meta.get("source", "unknown"),
                        chunk_index=meta.get("chunk_index", 0),
                        similarity_score=similarity,
                    )
                )

        logger.info(
            f"Retrieved {len(chunks)} chunks. "
            f"Top score: {chunks[0].similarity_score if chunks else 'N/A'}"
        )
        return chunks

    def get_collection_size(self) -> int:
        return self.collection.count()

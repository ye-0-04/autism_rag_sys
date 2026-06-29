from sentence_transformers import CrossEncoder
from typing import List
from app.retriever.retriever import NutritionChunk
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, chunks: List[NutritionChunk], top_k: int = 4
    ) -> List[NutritionChunk]:
        if not chunks:
            return chunks

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.similarity_score = float(score)

        reranked = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
        logger.info(
            f"Re-ranked {len(chunks)} chunks → keeping top {top_k}. "
            f"Top score: {reranked[0].similarity_score:.4f}"
        )
        return reranked[:top_k]

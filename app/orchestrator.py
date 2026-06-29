import logging
import tempfile
import os
from app.ocr.preprocessor import ImagePreprocessor
from app.ocr.extractor import OCRExtractor
from app.ocr.parser import GeneticDataParser
from app.retriever.retriever import NutritionRetriever
from app.formatter import OutputFormatter
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.models.nutrition_plan import NutritionPlan
from app.config import get_llm_provider

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.extractor = OCRExtractor()
        self.parser = GeneticDataParser()
        self.retriever = NutritionRetriever()
        self.llm = get_llm_provider()
        self.formatter = OutputFormatter(self.llm, SYSTEM_PROMPT)
        logger.info("RAGOrchestrator initialized.")

    async def process(
        self, file_bytes: bytes, filename: str, patient_id: str
    ) -> NutritionPlan:
        logger.info(f"Processing file '{filename}' for patient '{patient_id}'")

        suffix = os.path.splitext(filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            logger.info("Step 1/5: Preprocessing image(s)")
            pages = self.preprocessor.preprocess_file(tmp_path)

            logger.info("Step 2/5: Extracting text via OCR")
            raw_text = self.extractor.extract_from_pages(pages)

            logger.info("Step 3/5: Parsing genetic data")
            profile = self.parser.parse(raw_text, patient_id=patient_id)

            logger.info("Step 4/5: Retrieving knowledge base chunks")
            chunks = self.retriever.retrieve(profile, top_k=6)

            logger.info("Step 5/5: Generating nutrition plan via LLM")
            user_prompt = build_user_prompt(
                patient_id=patient_id,
                markers=profile.markers,
                flagged_values=profile.flagged_values,
                raw_text_summary=profile.raw_text_summary,
                chunks=chunks,
            )

            llm_response = await self.llm.generate(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            plan = await self.formatter.parse_with_retry(llm_response, profile, chunks)

        finally:
            os.unlink(tmp_path)

        logger.info(
            f"Pipeline complete for patient {patient_id}. "
            f"Confidence: {plan.confidence_score}. "
            f"Requires review: {plan.requires_doctor_review}"
        )
        return plan

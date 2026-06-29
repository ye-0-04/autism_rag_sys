from paddleocr import PaddleOCR
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class OCRExtractor:
    def __init__(self):
        logger.info("Loading PaddleOCR models...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
        logger.info("PaddleOCR models loaded.")

    def extract_text(self, image: np.ndarray) -> str:
        result = self.ocr.ocr(image, cls=True)

        if not result or not result[0]:
            logger.warning("OCR returned no results for this page.")
            return ""

        lines = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]

            if confidence >= 0.6:
                lines.append(text)
            else:
                logger.debug(
                    f"Skipping low-confidence OCR result: '{text}' ({confidence:.2f})"
                )

        extracted = "\n".join(lines)
        logger.info(f"Extracted {len(lines)} text lines from page")
        return extracted

    def extract_from_pages(self, pages: List[np.ndarray]) -> str:
        all_text = []
        for i, page in enumerate(pages):
            page_text = self.extract_text(page)
            all_text.append(f"--- PAGE {i + 1} ---\n{page_text}")

        return "\n\n".join(all_text)

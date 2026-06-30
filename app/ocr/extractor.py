import os
import platform
import pytesseract
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

# Point pytesseract to system Tesseract (portable: Windows msys2 vs Linux default)
if platform.system() == "Windows":
    os.environ.setdefault("TESSDATA_PREFIX", r"C:\msys64\ucrt64\share\tessdata")
    pytesseract.pytesseract.tesseract_cmd = r"C:\msys64\ucrt64\bin\tesseract.exe"
else:
    os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")


class OCRExtractor:
    def __init__(self):
        logger.info("Initializing Tesseract OCR extractor...")

    def extract_text(self, image: np.ndarray) -> str:
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/ +-"
        text = pytesseract.image_to_string(image, config=config)

        if not text or not text.strip():
            logger.warning("OCR returned no results for this page.")
            return ""

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        extracted = "\n".join(lines)
        logger.info(f"Extracted {len(lines)} text lines from page")
        return extracted

    def extract_from_pages(self, pages: List[np.ndarray]) -> str:
        all_text = []
        for i, page in enumerate(pages):
            page_text = self.extract_text(page)
            all_text.append(f"--- PAGE {i + 1} ---\n{page_text}")

        return "\n\n".join(all_text)

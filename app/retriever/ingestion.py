import fitz
import hashlib
import logging
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n\n".join(text_parts)


def chunk_text(text: str, source_name: str) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)

    return [
        {
            "text": chunk,
            "source": source_name,
            "chunk_index": i,
            "id": hashlib.md5(f"{source_name}:{i}:{chunk[:50]}".encode()).hexdigest(),
        }
        for i, chunk in enumerate(chunks)
        if len(chunk.strip()) > 50
    ]

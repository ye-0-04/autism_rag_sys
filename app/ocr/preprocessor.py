import cv2
import numpy as np
from pathlib import Path
import fitz
import logging
from typing import List

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            images.append(img)
        doc.close()
        logger.info(f"Extracted {len(images)} pages from PDF")
        return images

    def load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        return img

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = self._to_grayscale(img)
        img = self._denoise(img)
        img = self._deskew(img)
        img = self._binarize(img)
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (3, 3), 0)

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            return img

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if abs(angle) < 10:
                angles.append(angle)

        if not angles:
            return img

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return img

        logger.debug(f"Deskewing by {median_angle:.2f} degrees")
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        return cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )

    def preprocess_file(self, file_path: str) -> List[np.ndarray]:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            raw_images = self.pdf_to_images(file_path)
        elif path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
            raw_images = [self.load_image(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        preprocessed = [self.preprocess(img) for img in raw_images]
        logger.info(f"Preprocessed {len(preprocessed)} page(s) from {path.name}")
        return preprocessed

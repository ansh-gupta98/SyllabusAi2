"""
Document Parser Service
=======================
Extracts raw text from:
  - PDF  (base64) → via PyMuPDF (fitz)
  - Image (base64) → via pytesseract OCR
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from typing import Tuple

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentParserService:

    # ── PDF ──────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_text_from_pdf_base64(pdf_base64: str) -> Tuple[str, int]:
        """
        Decode base64 PDF and extract all text.
        Returns (extracted_text, page_count).
        Raises RuntimeError on failure.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError(
                "PyMuPDF not installed. Run: pip install PyMuPDF"
            )

        try:
            pdf_bytes = base64.b64decode(pdf_base64)
        except Exception:
            raise ValueError("Invalid base64 PDF data.")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = doc.page_count
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text("text"))
            doc.close()
            full_text = "\n\n".join(text_parts).strip()
            if not full_text:
                raise ValueError("PDF appears to be empty or image-only (no extractable text).")
            logger.info(f"PDF parsed: {pages} pages, {len(full_text)} chars extracted.")
            return full_text, pages
        except fitz.FileDataError:
            raise ValueError("Corrupted or invalid PDF file.")
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise RuntimeError(f"PDF parsing failed: {str(e)}")

    # ── Image ─────────────────────────────────────────────────────────────────

    @staticmethod
    def extract_text_from_image_base64(image_base64: str, mime_type: str = "image/jpeg") -> str:
        """
        Decode base64 image and extract text via OCR (pytesseract).
        Returns extracted text string.
        """
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            raise RuntimeError(
                "pytesseract/Pillow not installed. Run: pip install pytesseract pillow"
            )

        try:
            img_bytes = base64.b64decode(image_base64)
        except Exception:
            raise ValueError("Invalid base64 image data.")

        try:
            image = Image.open(io.BytesIO(img_bytes))
            # Preprocess: convert to greyscale for better OCR accuracy
            image = image.convert("L")
            text = pytesseract.image_to_string(image, config="--psm 3")
            text = text.strip()
            if not text:
                raise ValueError("No text could be extracted from the image. Ensure the image is clear and contains printed text.")
            logger.info(f"Image OCR complete: {len(text)} chars extracted.")
            return text
        except Exception as e:
            if "No text" in str(e):
                raise ValueError(str(e))
            logger.error(f"Image OCR error: {e}")
            raise RuntimeError(f"Image OCR failed: {str(e)}")


# Singleton
doc_parser = DocumentParserService()
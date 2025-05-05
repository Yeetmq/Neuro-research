from typing import Optional
from core.utils import logger
from pdfminer.high_level import extract_text
from io import BytesIO
import requests


class PDFHandler:
    @staticmethod
    def extract_text_from_url(pdf_url: str):
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            return PDFHandler._extract_from_bytes(response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"PDF download failed: {str(e)}")
            return None

    @staticmethod
    def _extract_from_bytes(content: bytes):
        try:
            with BytesIO(content) as pdf_file:
                text = extract_text(pdf_file)
                return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            return None
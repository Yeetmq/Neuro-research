from typing import List
from core.utils import logger
from xml.etree import ElementTree as ET
import requests
from typing import List, Optional
from io import BytesIO
from pdfminer.high_level import extract_text
from dataclasses import dataclass


@dataclass
class Article:
    title: str
    summary: str
    authors: List[str]
    published: str
    link: str
    pdf_url: str

    
class ArxivAPI:
    BASE_URL = "http://export.arxiv.org/api/query"
    NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
    
    def __init__(self):
        self.headers = {
            "User-Agent": "AcademicResearchTool/1.0 (https://github.com/your-repo)"
        }

    def search_articles(
        self,
        query: str,
        max_results: int = 5,
        sort_by: str = "submittedDate",
        sort_order: str = "descending"
    ) -> List[Article]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return self._parse_response(response.content)
        except requests.exceptions.RequestException as e:
            logger.error(f"Arxiv API request failed: {str(e)}")
            return []

    def _parse_response(self, content: bytes) -> List[Article]:
        try:
            root = ET.fromstring(content)
            articles = []
            
            for entry in root.findall("atom:entry", self.NAMESPACE):
                article = Article(
                    title=self._get_element_text(entry, "atom:title"),
                    summary=self._get_element_text(entry, "atom:summary"),
                    authors=[
                        author.find("atom:name", self.NAMESPACE).text
                        for author in entry.findall("atom:author", self.NAMESPACE)
                    ],
                    published=self._get_element_text(entry, "atom:published"),
                    link=self._get_element_text(entry, "atom:id"),
                    pdf_url=self._get_pdf_url(entry)
                )
                articles.append(article)
            return articles
        except ET.ParseError as e:
            logger.error(f"Failed to parse Arxiv response: {str(e)}")
            return []

    def _get_element_text(self, element, path: str) -> str:
        elem = element.find(path, self.NAMESPACE)
        return elem.text.strip() if elem is not None else ""

    def _get_pdf_url(self, entry) -> str:
        link = entry.find("atom:link[@title='pdf']", self.NAMESPACE)
        return link.attrib.get("href", "") if link is not None else ""

        
class TranslationClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = f"{base_url}/translate"

    def translate_text(
        self,
        text: str,
        source_lang: str = "ru",
        target_lang: str = "en"
    ) -> Optional[str]:
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }

        try:
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("translatedText")
        except requests.exceptions.RequestException as e:
            logger.error(f"Translation failed: {str(e)}")
            return None
import requests
from xml.etree import ElementTree as ET
from typing import List, Dict, Optional
from io import BytesIO
from pdfminer.high_level import extract_text
import logging
from dataclasses import dataclass
import re
from bs4 import BeautifulSoup
import certifi
from requests.exceptions import SSLError
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArxivArticle:
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
    ) -> List[ArxivArticle]:
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

    def _parse_response(self, content: bytes) -> List[ArxivArticle]:
        try:
            root = ET.fromstring(content)
            articles = []
            
            for entry in root.findall("atom:entry", self.NAMESPACE):
                article = ArxivArticle(
                    title=self._get_element_text(entry, "atom:title"),
                    summary=self._get_element_text(entry, "atom:summary"),
                    authors=[author.find("atom:name", self.NAMESPACE).text for author in entry.findall("atom:author", self.NAMESPACE)],
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

class PDFProcessor:
    @staticmethod
    def extract_text_from_url(pdf_url: str):
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            return PDFProcessor._extract_from_bytes(response.content)
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

# Работа с веб-страницами
def load_sites(filename="utils/sites.txt"):
    try:
        with open(filename, "r") as file:
            return [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        return [] 

def classify_links(links):
    video_links = []
    webpage_links = []

    for link in links:
        if re.search(r"(youtube\.com|rutube\.ru|vimeo\.com)", link, re.IGNORECASE):
            video_links.append(link)  
        else:
            webpage_links.append(link)  
    return video_links, webpage_links

def is_valid_page(content):
    if "Вы не робот?" in content or "Access forbidden" in content:
        return False
    return True

def filter_advertisements(soup):
    ads_classes = ["advertisement", "ad", "promo", "sponsored", "banner", "sidebar"]
    for ad_class in ads_classes:
        for ad_tag in soup.find_all(class_=ad_class):
            ad_tag.decompose() 

    ad_ids = ["ad-banner", "adblock", "ads", "popup"]
    for ad_id in ad_ids:
        ad_tag = soup.find(id=ad_id)
        if ad_tag:
            ad_tag.decompose()
    return soup

def extract_main_content(soup):
    content = ""
    main_content = soup.find(['article', 'main', 'section', 'div'], class_=re.compile(r'.*(content|body|main).*', re.IGNORECASE))
    if main_content:
        content = main_content.get_text(strip=True)
    else:
        for tag in ["p", "div", "span", "section"]:
            elements = soup.find_all(tag)
            for element in elements:
                content += element.get_text(strip=True) + "\n"
    return content

def parse_webpage(session, url, results_file):
    try:
        response = session.get(url, verify=certifi.where())
        
        if response.status_code != 200:
            logger.error(f"Ошибка доступа к {url}: статус код {response.status_code}")
            return

        content = response.text
        
        if not is_valid_page(content):
            logger.error(f"Страница с {url} содержит ошибку или проверку на робота.")
            return
        
        soup = BeautifulSoup(response.content, "html.parser")
        soup = filter_advertisements(soup) 
        
        title = soup.title.string if soup.title else "Без заголовка"
        logger.info(f"Заголовок страницы: {title}")
        
        content = extract_main_content(soup) 
        
        if content:
            with open(results_file, "a", encoding="utf-8") as file:
                file.write(f"\n\n{'='*50}\n")
                file.write(f"Заголовок страницы: {title}\n")
                file.write(f"URL: {url}\n")
                file.write(f"{'='*50}\n\n")
                file.write(content)
                file.write("\n\n" + "="*50 + "\n\n")
            
            logger.info(f"Содержимое страницы {url} записано в файл.")
        else:
            logger.info(f"Контент не найден на странице {url}.")
    except SSLError as e:
        logger.error(f"SSL ошибка при попытке соединения с {url}: {e}")
    except Exception as e:
        logger.error(f"Ошибка при обработке {url}: {e}")

def parse_all_pages(sites, results_file):
    session = requests.Session()  
    with ThreadPoolExecutor(max_workers=10) as executor:  
        future_to_url = {executor.submit(parse_webpage, session, url, results_file): url for url in sites}
        for future in as_completed(future_to_url):
            future.result()

# Основной процесс
def main():
    arxiv_client = ArxivAPI()
    translator = TranslationClient()
    pdf_processor = PDFProcessor()

    query = "transformers"
    translated_query = translator.translate_text(query)
    
    if not translated_query:
        logger.error("Translation failed")
        return
    
    logger.info(f"Translated query: {translated_query}")
    
    articles = arxiv_client.search_articles(translated_query, max_results=2)
    logger.info(f"Found {len(articles)} articles")
    
    results_file = "utils/results.txt"
    
    with open(results_file, "w", encoding="utf-8") as file:
        file.write("Результаты поиска по запросу: " + query + "\n")
        file.write("="*50 + "\n\n")

    results = []
    for article in articles:
        logger.info(f"Processing article: {article.title}")
        text = pdf_processor.extract_text_from_url(article.pdf_url)
        if text:
            results.append(text)
            with open(results_file, "a", encoding="utf-8") as file:
                file.write(f"Заголовок статьи: {article.title}\n")
                file.write(f"Ссылка: {article.link}\n")
                file.write(f"PDF URL: {article.pdf_url}\n")
                file.write(f"Текст статьи:\n{text[:1000]}\n\n{'='*50}\n")

    if results:
        logger.info("First article text snippet:")
        print(results[0][:1000])

    # Обрабатываем веб-сайты
    sites = load_sites('utils/sites.txt')
    _, webpage_links = classify_links(sites)
    parse_all_pages(webpage_links, results_file)

if __name__ == "__main__":
    main()

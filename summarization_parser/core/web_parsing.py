from typing import List, Tuple
from core.utils import logger
from web_parsing.links_search import get_yandex_results, get_google_results
from web_parsing.parser import classify_links, parse_all_pages, wiki_search, parse_all_links

class WebHandler:
    def find_sites(self, query: str, links_num: int) -> List[str]:
        logger.info(f"Searching sites for query: {query}")
        return get_google_results(query, links_num)
        
    def classify_links(self, path: str) -> Tuple[List[str], List[str]]:
        return classify_links(path)
        
    def parse_pages(self, result_path, links: List[str]) -> List[str]:
        logger.info(f"Parsing {len(links)} web pages")
        return parse_all_pages(links, result_path)
    
    def wiki_search(self, query: str):
        logger.info(f"Searching wiki articles")
        return wiki_search(query)
    
    def parse_all_links(self, result_path, links: List[str]) -> List[str]:
        logger.info(f"Parsing {len(links)} web pages")
        return parse_all_links(links, result_path)
from typing import List, Tuple
from core.utils import logger
from googlesearch import search
import wikipedia
from core.utils import logger
from threading import Thread
import trafilatura
import re

class WebHandler:
    def find_sites(self, query: str, links_num: int) -> List[str]:
        '''
        Находит ссылки в количестве links_num по запросу query
        '''
        logger.info(f"Searching sites for query: {query}")
        results = list(
            search(
                query, 
                num_results=links_num,
                lang="en",
                advanced=False
                ))
        return results
    
    def classify_links(self, path: str) -> Tuple[List[str], List[str]]:
        '''
        Сортирует ссылки на видео и веб ссылки
        '''
        video_links = []
        webpage_links = []
        wiki_links = []

        links = []

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    links.append(stripped_line)
            
        for link in links:
            if re.search(r"(youtube\.com|rutube\.ru|vimeo\.com)", link, re.IGNORECASE):
                video_links.append(link)
            elif re.search(r"(wiki|wikipedia)", link, re.IGNORECASE):
                wiki_links.append(link)
            else:
                webpage_links.append(link)  
        return video_links, webpage_links

    def wiki_search(self, query: str):
        logger.info(f"Searching wiki articles")
        try:
            wikipedia.set_lang("en")
            content = wikipedia.summary(query)
            logger.info(f"wiki добавлена")
            return content
        except wikipedia.exceptions.WikipediaException as e:
            logger.error(f"Ошибка при поиске в Wikipedia для запроса '{query}': {e}")
            return None
        except Exception as e:
            logger.exception(f"Неожиданная ошибка при поиске в Wikipedia: {e}")
            return None
    
    def _get_page_content(self, link, result_path: str,  timeout=5):
        '''
        trafilatura парсинг определенной ссылки
        '''
        result = None
        def task():
            nonlocal result
            try:
                html = trafilatura.fetch_url(link)
                result = trafilatura.extract(html)
            except Exception:
                pass
                
        thread = Thread(target=task)
        thread.start()
        thread.join(timeout)

        if result:
            with open(result_path, "a", encoding="utf-8") as file:
                
                file.write(result)
                
                file.write("\n\n" + "="*50 + "\n\n")
            
            print(f"Содержимое страницы {link} записано в файл.")

        return result

    def parse_all_links(self, result_path, links: List[str]) -> List[str]:
        '''
        Абстрактный метод парсинга содержимого всех найденных ранее ссылок
        '''
        logger.info(f"Parsing {len(links)} web pages")
        for link in links:
            self._get_page_content(link, result_path)


    

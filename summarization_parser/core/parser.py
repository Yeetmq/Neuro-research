import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from core.web_parsing_handler import WebHandler
from core.pdf_handler import PDFHandler
from core.translation import TranslationClient
from core.utils import save_to_file, logger, save_wiki_to_file, clear_and_save_to_file, clear_file


class SummarizationParser:
    def __init__(self, config):
        self.config = config
        self.web_handler = WebHandler()
        self.pdf_handler = PDFHandler()
        self.translate = config.get('translate', False)
        # self.translate = None
        self.translation_client = TranslationClient() if self.translate else None
        
    def run(self):
        original_query = self.config.get('query', '')
        translated_query = original_query
        
        if self.translate and self.translation_client:
            logger.info("Translating query from Russian to English...")
            translated_query = self.translation_client.translate(
                original_query, 
                source_lang='ru', 
                target_lang='en'
            )
            if not translated_query:
                logger.warning("Query translation failed, using original query")
                translated_query = original_query
            else:
                logger.info(f"Translated query: {translated_query}")
        
        self.config['query'] = translated_query
        logger.info(self.config)
        clear_file(self.config['result_path'])
        clear_file(self.config['links_path'])
        site_links = self.web_handler.find_sites(self.config['query'], self.config['links_num'])
        clear_and_save_to_file(site_links, self.config['links_path'])
        
        video_links, webpage_links = self.web_handler.classify_links(self.config['links_path'])
        self.web_handler.parse_all_links(links=webpage_links, result_path=self.config['result_path'])
        
        wiki = self.web_handler.wiki_search(self.config['query'])

        if wiki is not None:
            save_wiki_to_file(wiki, self.config['result_path'])
        else:
            logger.warning("Вики-статья не найдена для запроса: %s", self.config['query'])

    def _process_arxiv_articles(self):
        articles = self.arxiv_api.search_articles(self.config['query'], self.config['max_results'])
        results = []
        
        for article in articles:
            try:
                text = self.pdf_handler.extract_text_from_url(article.pdf_url)
                if text:
                    results.append(text)
            except Exception as e:
                logger.error(f"Error processing {article.title}: {e}")
                
        return results

    def _translate_file(self, file_path: str) -> None:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                logger.info(f"File {file_path} is empty, skipping translation")
                return

            logger.info(f"Translating content of {file_path} to English...")
            translated_text = self.translation_client.translate_text(
                content, 
                source_lang='auto',
                target_lang='en'
            )
            
            if translated_text:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)
                logger.info(f"Translated and saved to {file_path}")
            else:
                logger.error("Translation failed, leaving file unchanged")
                
        except Exception as e:
            logger.exception(f"Error translating file {file_path}: {e}")

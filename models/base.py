import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from models.bart import BartSummarizer
from models.llm import ReportGenerator
from summarization_parser.core.translation import TranslationClient

def load_config(path: str) -> Dict:
    """Загрузка конфигурации из YAML"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_text(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Текст загружен из {file_path}")
        return text
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        raise

def save_text(text: str, file_path: str) -> None:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Текст сохранен в {file_path}")
    except Exception as e:
        logger.error(f"Ошибка сохранения файла: {e}")
        raise

class SummarizationPipeline:
    
    def __init__(self, 
                 bart_model_path,
                 llm_model_path,
                 query: str):
        self.summarizer = bart_model_path
        self.generator = llm_model_path
        self.query = query
        self.translator = TranslationClient()
    
    def run(self, input_text: str) -> Dict[str, Any]:
        """Выполняет полный пайплайн обработки текста"""
        results = {
            "original_length": len(input_text.split()),
            "summaries": [],
            "structured_report": "",
            "original_report": ""
        }
        
        logger.info("Запуск рекурсивной суммаризации")
        summaries = self.summarizer.summarize(input_text)
        results["summaries"] = summaries
        
        logger.info("Генерация структурированного отчета")
        structured_report = self.generator.generate(summaries, self.query)
        results["original_report"] = structured_report
        results["structured_report"] = self.translator.translate(structured_report, source_lang="en", target_lang="ru")
        
        return results

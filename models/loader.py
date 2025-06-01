# models/loader.py
from models.base import SummarizationPipeline
import logging
from models.bart import BartSummarizer
from models.llm import ReportGenerator

logger = logging.getLogger(__name__)

def load_pipeline(config):
    """Загружает пайплайн один раз при старте"""
    try:
        logger.info("Загрузка модели при старте...")
        summarizer, generator = load_models(
            config['path_to_bart_model'], 
            config['llm_model_name'])

        return SummarizationPipeline(
            bart_model_path=summarizer,
            llm_model_path=generator,
            query=config.get('query', 'initial_query')
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise

def load_models(bart_model_path: str, llm_model_path: str):
    summarizer = BartSummarizer(model_path=bart_model_path)
    generator = ReportGenerator(model_name=llm_model_path)
    return summarizer, generator
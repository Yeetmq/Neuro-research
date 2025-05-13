# pipeline.py

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

from bart import BartSummarizer
from llm import ReportGenerator

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
                 bart_model_path: str,
                 llama_model_path: str):
        self.summarizer = BartSummarizer(model_path=bart_model_path)
        self.generator = ReportGenerator(model_name=llama_model_path)
    
    def run(self, input_text: str) -> Dict[str, Any]:
        """Выполняет полный пайплайн обработки текста"""
        results = {
            "original_length": len(input_text.split()),
            "summaries": [],
            "structured_report": ""
        }
        
        cleaned_text = input_text
        
        logger.info("Запуск рекурсивной суммаризации")
        summaries = self.summarizer.summarize(cleaned_text)
        results["summaries"] = summaries
        
        logger.info("Генерация структурированного отчета")
        structured_report = self.generator.generate(summaries)
        results["structured_report"] = structured_report
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Пайплайн суммаризации и генерации отчета')
    parser.add_argument('--input-file', required=True, help='Путь к входному файлу')
    parser.add_argument('--output-summary', default='summary.txt', help='Путь к файлу суммаризации')
    parser.add_argument('--output-report', default='report.txt', help='Путь к файлу отчета')
    parser.add_argument('--config', help='Путь до кфг моделей')
    
    sample_agrs_list = [
        '--input-file', '/home/debian/develop/denis/Neuro-research/test/input.txt',
        '--output-summary', '/home/debian/develop/denis/Neuro-research/test/output-summary.txt',
        '--config', '/home/debian/develop/denis/Neuro-research/config/model_cfg.yaml',
        '--output-report', '/home/debian/develop/denis/Neuro-research/test/output-report.txt',
        # '--recursive', 'true'
    ]

    args = parser.parse_args(sample_agrs_list)
    config = load_config(args.config)
    
    try:
        input_text = load_text(args.input_file)

        logger.info("Создание пайплайна")

        pipeline = SummarizationPipeline(
            bart_model_path=config['path_to_bart_model'],
            llama_model_path=config['llm_model_name'])

        logger.info("запуск пайпланйна")

        results = pipeline.run(input_text)
        
        if results["summaries"]:
            summaries_text = "\n\n".join(results["summaries"])
            save_text(summaries_text, args.output_summary)
        else:
            logger.warning("Суммаризация не вернула результатов")
            
        if results["structured_report"]:
            save_text(results["structured_report"], args.output_report)
        else:
            logger.warning("Генерация отчета не вернула результатов")
            
        print("\n" + "="*80)
        print("Структурированный отчет:")
        print("-"*80)
        print(results["structured_report"] or "Не сгенерирован")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения пайплайна: {e}")
        print(f"Произошла ошибка: {e}")
    except KeyboardInterrupt:
        logger.info("Пайплайн прерван пользователем")
        print("\nПроцесс прерван пользователем")

if __name__ == "__main__":
    main()
# src/pipeline_executor.py

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any
import argparse


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from summarization_parser.cli.main import main as run_parser
from models.base import SummarizationPipeline, load_config, load_text, save_text
from summarization_parser.cli.main import main as parser_main
from summarization_parser.core.parse import SummarizationParser


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def execute_pipeline(summarizer, generator, query: str, config_path: str, data_dir: str, max_retries: int = 3, delay: int = 5, links_num: int = 5):
    """
    Полный пайплайн: парсинг -> суммаризация -> генерация отчета
    
    Args:
        query: Поисковый запрос
        config_path: Путь к файлу конфигурации
        data_dir: Директория для данных
        max_retries: Максимум попыток ожидания готовности файла
        delay: Задержка между попытками (в секундах)
    
    Returns:
        Dict[str, Any]: Результаты обработки
    """
    logger.info("Запуск парсера")
    
    parser_args = argparse.ArgumentParser(description='Data parser for BART summarization')
    parser_args.add_argument('--query', required=True, help='Search query')
    parser_args.add_argument('--config', default='config/settings.yaml', help='Config file path')
    parser_args.add_argument('--translate', default='false', help='Need translation?')
    
    sample_agrs_list = [
        '--query', query,
        '--config', os.getenv("CONFIG_PATH", "/app/cfg.yaml"),
        '--translate', 'false'
    ]
    
    args = parser_args.parse_args(sample_agrs_list)

    logger.info(f"Загрузка кфг")
    config = load_config(args.config)
    logger.info(f"кфг загрузили")
    config['links_num'] = links_num
    config['query'] = args.query
    config['translate'] = args.translate
    logger.info(f"парсер")
    parser = SummarizationParser(config)
    parser.run()
    logger.info(f"парсер закончил")
    input_file = Path(data_dir) / "page_content.txt"
    
    logger.info(f"Ожидание сохранения данных в {input_file}")
    
    # Ожидание появления файла
    for attempt in range(max_retries):
        if input_file.exists() and input_file.stat().st_size > 0:
            logger.info(f"Файл найден после {attempt+1} попыток")
            break
        elif attempt < max_retries - 1:
            logger.info(f"Попытка {attempt+1}/{max_retries}. Ожидание {delay} секунд...")
            time.sleep(delay)
        else:
            logger.error("Файл с данными не найден")
            raise FileNotFoundError(f"Файл {input_file} не найден или пуст")
    
    logger.info(f"Загрузка текста из {input_file}")
    input_text = load_text(str(input_file))
    
    logger.info("Создание пайплайна")
    

    # model_config = load_config(config["model_config_path"])
    
    pipeline = SummarizationPipeline(
        bart_model_path=summarizer,
        llm_model_path=generator,
        query=config['query']
    )
    
    logger.info("Запуск пайплайна суммаризации")
    results = pipeline.run(input_text)
    
    summary_file = Path(data_dir) / "summary.txt"
    report_file = Path(data_dir) / "report.txt"
    
    if results["summaries"]:
        logger.info(f"Сохранение суммаризации в {summary_file}")
        save_text("\n\n".join(results["summaries"]), str(summary_file))
    else:
        logger.warning("Суммаризация не вернула результатов")

    if results["structured_report"]:
        logger.info(results["structured_report"])
        logger.info(f"Сохранение отчета в {report_file}")
        save_text(results["structured_report"], str(report_file))
    else:
        logger.warning("Генерация отчета не вернула результатов")
    
    print("\n" + "="*80)
    print("Финальная суммаризация:")
    print("-"*80)
    
    if results["summaries"]:
        for i, summary in enumerate(results["summaries"], 1):
            print(f"{i}. {summary}")
    else:
        print("Не выполнено")
    
    print("\n" + "="*80)
    print("Структурированный отчет на английском:")
    print("-"*80)
    print(results["original_report" or "Не сгенерирован"])
    print("\n" + "="*80)
    print("Структурированный отчет:")
    print("-"*80)
    print(results["structured_report"] or "Не сгенерирован")
    
    return {
        "original_length": results["original_length"],
        "summaries": results["summaries"],
        "original_report": results["original_report"],
        "structured_report": results["structured_report"],
    }

if __name__ == "__main__":
    QUERY = "Algorithms for beginers"
    CONFIG_PATH = '/home/debian/develop/denis/Neuro-research/summarization_parser/config/settings.yaml'
    MODEL_CONFIG_PATH = "/models/config/model_cfg.yaml"
    DATA_DIR = "/home/debian/develop/denis/Neuro-research/summarization_parser/data"
    
    try:
        logger.info("Запуск полного пайплайна: парсинг -> суммаризация -> отчет")
        
        # Добавляем путь к конфигу модели в config
        # model_config = load_config(MODEL_CONFIG_PATH)
        # parser_config = load_config(CONFIG_PATH)
        # parser_config["model_config_path"] = MODEL_CONFIG_PATH
        
        # Сохраняем обновленный конфиг
        # import yaml
        # with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        #     yaml.safe_dump(parser_config, f)
        
        # Запуск пайплайна
        full_results = execute_pipeline(
            query=QUERY,
            config_path=CONFIG_PATH,
            data_dir=DATA_DIR
        )
        logger.info("Запуск произошел")


        logger.info(f"Исходная длина: {full_results['original_length']} слов")
        logger.info(f"Найдено источников: {full_results['source_count']}")
        logger.info(f"Длина отчета: {full_results['summary_length']} слов")
        logger.info("Пайплайн завершен успешно")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения пайплайна: {e}")
        print(f"Произошла ошибка: {e}")
    except KeyboardInterrupt:
        logger.info("Пайплайн прерван пользователем")
        print("\nПроцесс прерван пользователем")
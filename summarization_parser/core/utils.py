import logging
import yaml
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_to_file(data: List[str], path: str) -> None:
    """Сохранение данных в файл"""
    with open(path, "a", encoding="utf-8") as f:
        for item in data:
            f.write(item + "\n\n")

def save_wiki_to_file(data: List[str], path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(data + "\n\n")

def load_config(path: str) -> Dict:
    """Загрузка конфигурации из YAML"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def clear_and_save_to_file(data: List[str], path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(item + "\n\n")
        logger.info(f"Данные успешно сохранены в {path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {e}")
        raise

def clear_file( path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
                f.write("")
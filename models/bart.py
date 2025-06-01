from transformers import BartForConditionalGeneration, AutoTokenizer
from typing import List, Optional
import torch
import logging
import gc
import re
from torch.cuda.amp import autocast

logger = logging.getLogger(__name__)

class BartSummarizer:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.max_input_length = 1024
        self.max_new_tokens = 128
        self.chunk_size = 512
        self.overlap = 64

    def chunk_text(self, text: str, chunk_size: int = None) -> List[str]:
        chunk_size = chunk_size or self.chunk_size
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
            
        logger.info(f"Текст разбит на {len(chunks)} чанков")
        return chunks

    def _generate_summary(self, text: str) -> str:
        try:
            inputs = self.tokenizer(
                text,
                max_length=self.max_input_length,
                truncation=True,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad(), autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.5,
                    length_penalty=1.0
                )
                
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка генерации суммаризации: {e}")
            return text

    def _get_word_count(self, text: str | List[str]) -> int:
        if isinstance(text, str):
            return len(text.split())
        return sum(len(t.split()) for t in text)

    def run(self, input_data: str | List[str], max_iterations: int = 5) -> List[str]:
        """Рекурсивная суммаризация длинных текстов"""
        current_text = input_data
        current_length = self._get_word_count(current_text)
        iterations = 0
        
        logger.info(f"Исходная длина: {current_length} слов")

        logger.info(f"Исходный текст до суммаризации: {input_data}")
        
        while (current_length > 3000 and iterations < max_iterations) or iterations == 0 :
            logger.info(f"Итерация {iterations+1}: Текст слишком длинный ({current_length} слов). Выполняем повторную суммаризацию...")
            
            if isinstance(current_text, str):
                chunks = self.chunk_text(current_text)
            elif isinstance(current_text, list):
                text = ''
                for item in current_text:
                    text += item
                chunks = self.chunk_text(text)
            else:
                raise ValueError("Неподдерживаемый формат входных данных")
                
            summaries = [self._generate_summary(chunk) for chunk in chunks]
            
        
            current_text = ' '.join(summaries)

            current_length = self._get_word_count(current_text)
            iterations += 1

            logger.info(f"Исходный текст до суммаризации: {current_text} после {iterations} итераций")

            if iterations >= max_iterations:
                logger.warning(f"Достигнут лимит итераций ({max_iterations}). Текст: {current_length} слов")
                
        logger.info(f"Суммаризация завершена. Финальная длина: {current_length} слов")
        
        return summaries

    def summarize(self, text: str) -> str:
        try:
            summaries = self.run(text)

            return summaries
            
        except Exception as e:
            logger.error(f"Ошибка в суммаризации: {e}")
            return text

    def unload(self):
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        logger.info("Модель выгружена из памяти")
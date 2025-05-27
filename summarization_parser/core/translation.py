import transformers
from transformers import MarianTokenizer, MarianMTModel
from typing import List, Optional
import torch
import logging
import gc
import re


class TranslationClient:
    def __init__(self, base_url: str = "0.0.0.0:5000"):
        self.base_url = f"{base_url}/translate"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _load_model(self, source_lang: str, target_lang: str):
        """Загружает модель в зависимости от языков"""
        model_map = {
            ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
            ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru"
        }
        model_name = model_map.get((source_lang, target_lang))
        
        if not model_name:
            raise ValueError(f"Модель для перевода с {source_lang} на {target_lang} не найдена")
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
    
    def translate(self, text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
        """Переводит текст между русским и английским"""
        if source_lang == "auto":
            source_lang = "ru" if any("\u0400" <= c <= "\u04FF" for c in text[:100]) else "en"
        
        # Загрузка модели в зависимости от направления
        self._load_model(source_lang, target_lang)
        
        # Разделение на чанки для длинных текстов
        chunks = self._split_into_chunks(text)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"Перевод части {i+1}/{len(chunks)}")
            try:
                tokenized_text = self.tokenizer.prepare_seq2seq_batch([chunk], return_tensors='pt').to(self.device)
                translated = self.model.generate(**tokenized_text)
                result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_chunks.append(result)
            except Exception as e:
                print(f"Ошибка при переводе части {i+1}: {e}")
        
        return " ".join(translated_chunks)

    def _split_into_chunks(self, text: str, max_length: int = 512) -> list:
        """Разбивает текст на чанки с учётом слов"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) > max_length:
                chunks.append(" ".join(current_chunk[:-1]))
                current_chunk = [word]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def translate_file_in_parts(self, file_path: str, source_lang: str = "auto", target_lang: str = "en"):
        with open(file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()
        
        print(f"Исходный текст: {len(input_text)} символов")
        
        print(f"После очистки: {len(input_text)} символов")
        
        translated_text = self.translate(input_text, source_lang, target_lang)
        print(f"После перевода: {len(translated_text)} символов")
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(translated_text)
        
        print("Перевод завершён.")

if __name__=='__main__':
    translator = TranslationClient()
    result = translator.translate("Привет, как дела?", source_lang="ru", target_lang="en")
    print(result)

    print(translator.translate("Hello, how are you?", source_lang="en", target_lang="ru"))
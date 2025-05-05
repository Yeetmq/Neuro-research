import requests
from typing import Optional
from core.utils import logger
from transformers import MarianMTModel, MarianTokenizer

import torch
import re


class TranslationClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = f"{base_url}/translate"
        model_name = "Helsinki-NLP/opus-mt-ru-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def translate_text(
        self,
        text: str,
        source_lang: str = "auto",  # Автоматическое определение
        target_lang: str = "en"
    ) -> Optional[str]:
        payload = {
            "q": text,
            "source": source_lang,
            "target": target_lang,
            "format": "text"
        }

        try:
            response = requests.post(
                self.base_url,
                data=payload,  # Используем data вместо json
                timeout=120
            )
            response.raise_for_status()
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response content: {response.text}")
            return response.json().get("translatedText")
        except requests.exceptions.RequestException as e:
            logger.error(f"Translation failed: {str(e)}")
            return None

    def translate(self, text: str) -> str:
        """Переводит текст с русского на английский"""
        tokenized_text = self.tokenizer.prepare_seq2seq_batch([text], return_tensors='pt').to(self.device)
        translated = self.model.generate(**tokenized_text)
        result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return result

    def translate_file(self, file_path: str):
        """Читает файл, переводит текст и сохраняет результат обратно"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                input_text = file.read()

            print("Файл успешно прочитан.")

            translated_text = self.translate(input_text)

            print("Перевод завершён.")

            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(translated_text)

            print(f"Перевод сохранён в файл: {file_path}")

        except Exception as e:
            print(f"Произошла ошибка: {e}")

    def clean_text(self, text):
        text = re.sub(r"<[^>]+>", "", text)
        
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        
        lines = text.split("\n")
        seen = set()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
        
        text = re.sub(r"(?i)(?:etext|from|q)=\S+", "", text)

        return text

    def translate_file_in_parts(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()
        
        print(f"Исходный текст: {len(input_text)} символов")
        
        cleaned_text = self.clean_text(input_text)
        print(f"После очистки: {len(cleaned_text)} символов")
        
        translated_text = self.translate_in_chunks(cleaned_text)
        print(f"После перевода: {len(translated_text)} символов")
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(translated_text)
        
        print("Перевод завершён.")

    def translate_in_chunks(self, text, chunk_size=1000):
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        translated = []

        for i, chunk in enumerate(chunks):
            print(f"Перевод части {i+1}/{len(chunks)}")
            try:
                result = self.translate(chunk)
                if result:
                    translated.append(result)
                else:
                    print(f"Часть {i+1} не переведена")
            except Exception as e:
                print(f"Ошибка при переводе части {i+1}: {e}")

        return "\n".join(translated)
    

# if __name__ == '__main__':
    # translate_file_in_parts()
    
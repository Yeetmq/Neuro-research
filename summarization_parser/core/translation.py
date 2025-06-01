from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class TranslationClient:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-1.3B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model(model_name)
        
        # Языковые коды для NLLB
        self.language_codes = {
            "ru": "rus_Cyrl",  # Русский
            "en": "eng_Latn",  # Английский
            "russian": "rus_Cyrl",
            "english": "eng_Latn"
        }

    def _load_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="sdpa"
        )
        return model, tokenizer

    def _preprocess_text(self, text: str) -> str:
        """Очистка и подготовка технического текста"""
        return " ".join(text.strip().split())

    def _detect_language(self, text: str) -> str:
        """Простое определение языка по символам"""
        russian_chars = sum(1 for char in text if 'а' <= char <= 'я' or 'А' <= char <= 'Я')
        english_chars = sum(1 for char in text if 'a' <= char <= 'z' or 'A' <= char <= 'Z')
        return "ru" if russian_chars > english_chars else "en"

    def translate(
        self, 
        texts: str,  # Принимаем как строку, так и список строк
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        batch_size: int = 4
    ) -> str:  # Возвращаем соответственно строку или список
        try:
            # Получение кодов языков
            src_code = self.language_codes.get(source_lang.lower(), source_lang)
            tgt_code = self.language_codes.get(target_lang.lower(), target_lang)
            
            input_texts = [texts]

            logger.info(f'Orig_text_____________{input_texts}')
            
            # Предварительная обработка текстов
            processed_texts = [self._preprocess_text(text) for text in input_texts]
            
            translations = []
            tokens = self.tokenizer.encode(texts, add_special_tokens=False)
            # Пакетная обработка для экономии памяти
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i+batch_size]
                
                # Токенизация с указанием исходного языка
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    max_length=1024
                ).to(self.device)
                
                # Получение ID целевого языка
                lang_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
                
                # Генерация переводов
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=lang_id,
                    max_length=2048,  # ← Увеличьте лимит
                    min_length=int(len(tokens)-0.1*len(tokens)),
                    num_beams=4,
                    # repetition_penalty=1.2,
                    # no_repeat_ngram_size=3,
                    early_stopping=False,  # ← Не останавливайте генерацию преждевременно
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Декодирование результатов
                batch_translations = self.tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                translations.extend(batch_translations)
            
            # Возвращаем результат в соответствии с входным типом
            return translations[0]
        
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return "" if isinstance(texts, str) else [""] * len(texts)

# Примеры использования
if __name__ == "__main__":
    translator = TranslationClient()
    
    # Пример 3: Перевод en->ru
    text = 'Quantum computing harnesses quantum mechanics to unlock unprecedented computational abilities beyond the reach of classical computers. Quantum computers utilize qubits, which exist in a superposition of states, allowing them to represent a vast array of possibilities simultaneously. This quantum superposition empowers quantum computers to tackle complex problems with exponential speed, opening new avenues for scientific exploration and technological advancement. While still nascent, quantum computing holds the potential to revolutionize various fields, including drug discovery, machine learning, and supply chain optimization. However, the practical implementation of quantum computers faces challenges such as decoherence and the need for specialized hardware. Nonetheless, researchers are making strides towards overcoming these obstacles, paving the way for future quantum breakthroughs.'

    print("Явный перевод en->ru:")
    print(translator.translate(text, source_lang="en", target_lang="ru"))
    print("\n" + "="*80 + "\n")

    # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3b")
    # tokens = tokenizer.encode(text, add_special_tokens=False)
    # print(f"Количество токенов: {len(tokens)}")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class TranslationClient:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-1.3B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tokenizer = self._load_model(model_name)
        
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
        texts: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        batch_size: int = 4
    ) -> str:
        try:
            src_code = self.language_codes.get(source_lang.lower(), source_lang)
            tgt_code = self.language_codes.get(target_lang.lower(), target_lang)
            
            input_texts = [texts]

            processed_texts = [self._preprocess_text(text) for text in input_texts]
            
            translations = []
            tokens = self.tokenizer.encode(texts, add_special_tokens=False)
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    max_length=1024
                ).to(self.device)
                
                lang_id = self.tokenizer.convert_tokens_to_ids(tgt_code)
                
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=lang_id,
                    max_length=2048,
                    min_length=int(len(tokens)-0.1*len(tokens)),
                    num_beams=4,
                    # repetition_penalty=1.2,
                    # no_repeat_ngram_size=3,
                    early_stopping=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                batch_translations = self.tokenizer.batch_decode(
                    generated_tokens, 
                    skip_special_tokens=True
                )
                
                translations.extend(batch_translations)
            
            return translations[0]
        
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return "" if isinstance(texts, str) else [""] * len(texts)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
import logging
from typing import List
import re
import os

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, model_name: str = "google/gemma-7b-it", use_4bit: bool = True):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        login(token=os.getenv("HF_TOKEN"))
        self.model, self.tokenizer = self._load_model(model_name, use_4bit)
        self._setup_tokenizer()

    def _load_model(self, model_name: str, use_4bit: bool):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if use_4bit else None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            attn_implementation="sdpa"
        )
        return model, tokenizer

    def _setup_tokenizer(self):
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, query: str, summaries: List[str]) -> str:
        return f"""<bos><start_of_turn>user
    Generate a SINGLE-PARAGRAPH academic summary addressing the research query. 

    Query: {query}

    Key points to integrate:
    {"; ".join(summaries)}

    STRICT REQUIREMENTS:
    1. Exactly one continuous paragraph with 5-10 complex sentences
    2. Academic technical style with precise terminology
    3. No markdown, bullet points, or section headings
    4. Complete grammatical sentences with proper punctuation
    5. Directly address the query without digressions
    6. Avoid informal language and filler words<end_of_turn>
    <start_of_turn>model
    Academic Summary:
    """


    def _clean_response(self, text: str) -> str:
        # Удаление markdown и лишних переносов
        text = re.sub(r'\*{1,3}|\-{1,3}', '', text)
        text = re.sub(r'\n{2,}', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Обеспечение одного абзаца
        if len(text.split('\n')) > 1:
            return ' '.join(text.split('\n'))
        
        return text.strip()

    def generate(self, summaries: List[str], query: str) -> str:
        try:
            prompt = self._build_prompt(query, summaries)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False  # Спецтокены уже в промпте
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.2,  # Более детерминированный
                top_k=40,
                top_p=0.92,
                repetition_penalty=1.25,
                do_sample=True,
                num_beams=4,  # Лучшее покрытие вариантов
                early_stopping=True,
                no_repeat_ngram_size=4,
                length_penalty=0.8,  # Предотвращает многословие
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Обрезаем промпт из вывода
            response_start = inputs.input_ids.shape[-1]
            generated_ids = outputs[0, response_start:]
            
            return self.tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True
            ).strip()
        
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return ""

if __name__ == "__main__":
    generator = ReportGenerator()
    
    report = generator.generate(
        summaries=[
            "Neural architecture search (NAS) reduces model design time by 60%",
            "Evolutionary algorithms show better exploration than reinforcement learning in NAS",
            "Hardware-aware NAS improves inference speed by 3x on target devices"
        ],
        query="Compare NAS approaches for edge device optimization"
    )
    
    print("Generated Report:")
    print(report)
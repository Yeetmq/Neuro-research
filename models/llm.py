from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(self.device)

    def generate(self, summaries: List[str]) -> str:
        prompt = """<|user|>: Write a single continuous paragraph that seamlessly integrates these key points:
        {}
        
        Guidelines:
        1. Connect ideas using transitional phrases
        2. Maintain logical flow
        3. Avoid section headings or bullet points
        4. Use academic linking
        5. Keep technical terminology
        
        Output must be one cohesive paragraph (3-5 complex sentences). <|assistant|>:""".format("\n".join(f"- {s}" for s in summaries))

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.4,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            return response.split("<|assistant|>:")[-1].strip()
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return ""
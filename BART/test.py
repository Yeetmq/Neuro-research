from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk, concatenate_datasets, Dataset

from evaluate import load

import os
import torch
import gc
import numpy as np

from datasets import load_dataset, concatenate_datasets
import aiohttp
from pathlib import Path
from datasets import Dataset
import numpy as np

model_path = "/home/debian/develop/denis/Neuro-research/BART/bart-finetuned/final_model"
CHUNKS_DIR = "/home/debian/develop/denis/Neuro-research/BART/data"
BATCH_SIZE = 4
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
SEED = 42

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


arxiv = load_dataset("scientific_papers", "arxiv", 
                     split="train", 
                     trust_remote_code=True, 
                     storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
                     cache_dir='/home/debian/.cache/huggingface/datasets')


arxiv = arxiv.remove_columns(['section_names'])
arxiv = arxiv.rename_column('abstract', 'summary')

def load_filtered_dataset(data_root="/home/debian/develop/denis/Neuro-research/BART/converted_data/data/test"):
    data_path = Path(data_root)
    return Dataset.from_json([
        str(p) for p in data_path.rglob("*.json")
    ])

patent_dataset = load_filtered_dataset()


patent_dataset = patent_dataset.remove_columns(['publication_number', 'application_number'])
patent_dataset = patent_dataset.rename_column('abstract', 'summary')
patent_dataset = patent_dataset.rename_column('description', 'article')



def split_and_combine_datasets(arxiv_ds: Dataset, 
                              patent_ds: Dataset, 
                              seed: int = 42,
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1) -> tuple[Dataset, Dataset, Dataset]:
    """
    Разделяет каждый датасет на train/val/test и объединяет соответствующие части
    
    Параметры:
    arxiv_ds: Датасет arXiv
    patent_ds: Датасент патентов
    seed: Сид для воспроизводимости
    train_ratio: Доля тренировочных данных (0.0-1.0)
    val_ratio: Доля валидационных данных (0.0-1.0)
    
    Возвращает:
    (train, val, test) - объединенные датасеты
    """
    
    assert np.isclose(train_ratio + val_ratio + (1 - train_ratio - val_ratio), 1.0), "Пропорции должны суммироваться к 1"
    
    def split_single(ds: Dataset) -> tuple[Dataset, Dataset, Dataset]:
        train_test = ds.train_test_split(
            test_size=1-train_ratio, 
            seed=seed,
            shuffle=True
        )
        
        val_test = train_test['test'].train_test_split(
            test_size=val_ratio/(val_ratio + (1 - train_ratio - val_ratio)), 
            seed=seed,
            shuffle=True
        )
        
        return train_test['train'], val_test['train'], val_test['test']
    
    arxiv_train, arxiv_val, arxiv_test = split_single(arxiv_ds)
    patent_train, patent_val, patent_test = split_single(patent_ds)
    
    combined_train = concatenate_datasets([arxiv_train, patent_train])
    combined_val = concatenate_datasets([arxiv_val, patent_val])
    combined_test = concatenate_datasets([arxiv_test, patent_test])
    
    return combined_train, combined_val, combined_test

train_ds, val_ds, test_ds = split_and_combine_datasets(
    arxiv_ds=arxiv,
    patent_ds=patent_dataset,
    seed=42,
    train_ratio=0.8,
    val_ratio=0.1
)

print(f"Размеры финальных датасетов:")
print(f"Train: {len(train_ds)} samples")
print(f"Val: {len(val_ds)} samples")
print(f"Test: {len(test_ds)} samples")

def load_chunks_optimized(chunk_dir):
    chunk_dirs = [
        os.path.join(chunk_dir, d)
        for d in sorted(os.listdir(chunk_dir))
        if d.startswith("chunk") and os.path.isdir(os.path.join(chunk_dir, d))
    ]

    chunk_files = []
    for d in chunk_dirs:
        filename = "data-00000-of-00001.arrow"
        file_path = os.path.join(d, filename)

        if os.path.exists(file_path):
            chunk_files.append(file_path)
        else:
            raise FileNotFoundError(f"File {file_path} not found in directory {d}")

    if not chunk_files:
        raise ValueError(f"No valid chunk files found in {chunk_dir}")

    print(f"Loading {len(chunk_files)} chunks from {chunk_dir}")
    return concatenate_datasets([
        Dataset.from_file(f) for f in chunk_files
    ])

import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions = []
references = []

for example in tqdm(test_ds, desc="Generating predictions"):
    inputs = tokenizer(
        example["article"], 
        return_tensors="pt", 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            early_stopping=True
        )
    
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(pred_text)
    references.append(example["summary"])

# %%
from evaluate import load

rouge = load("rouge")
bleu = load("bleu")

rouge_results = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True
)

bleu_results = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

print("ROUGE Scores:")
print(f"ROUGE-1: {rouge_results['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_results['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")

print("\nBLEU Score:")
print(f"BLEU: {bleu_results['bleu']:.4f}")


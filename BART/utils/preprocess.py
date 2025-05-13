from datasets import load_dataset, concatenate_datasets
import aiohttp
from pathlib import Path
from datasets import Dataset
import numpy as np
from datasets import Features, Value

from convert_data import *
from chunkinise import *

def get_dataset_from_hf(dataset_name: str, dataset_type: str, split, cache_dir: str=None) -> Dataset:
    
    dataset = load_dataset(
        dataset_type, dataset_name, 
        split=split, 
        trust_remote_code=True,
        storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}},
        cache_dir=cache_dir)
    
    return dataset

def get_dataset_from_json(path_to_data: str, target_path: str, mid_target: str) -> Dataset:

    # process_gz_files(path_to_data, mid_target)
    # filter_and_save_records(mid_target, target_path)

    print('---------------------------')

    print('filter_and_save_records')
    print(target_path)
    
    print('---------------------------')

    # features = Features({
    #     'article': Value('string'),
    #     'summary': Value('string'),
    # })
    
    json_files = []
    data_path = Path(target_path)
    for p in data_path.rglob("*.json"):
        if p.stat().st_size == 0:
            print(f"Empty file skipped: {p}")
            continue
        json_files.append(str(p))
    
    if not json_files:
        raise ValueError("No valid JSON files found")
    
    return Dataset.from_json(
        json_files
    )

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
    
    arxiv_ds = arxiv_ds.remove_columns(['section_names'])
    arxiv_ds = arxiv_ds.rename_column('abstract', 'summary')

    patent_ds = patent_ds.remove_columns(['publication_number', 'application_number'])
    patent_ds = patent_ds.rename_column('abstract', 'summary')
    patent_ds = patent_ds.rename_column('description', 'article')

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


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")


    arxiv = get_dataset_from_hf('arxiv', 'scientific_papers', 
                                split='train',
                                cache_dir=None)
    
    patent_dataset = get_dataset_from_json(
        path_to_data=r'D:\ethd\ml\Neuro-research\BART\data',
        target_path='patent_data',
        mid_target='mid_target'
    )

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
    print(test_ds)

    # tokenize_in_chunks(train_ds, save_dir='processed_chunks_train', chunk_size=2000, tokenizer=tokenizer)
    # tokenize_in_chunks(val_ds, save_dir='processed_chunks_val', chunk_size=2000, tokenizer=tokenizer)
    # tokenize_in_chunks(test_ds, save_dir='processed_chunks_test', chunk_size=2000, tokenizer=tokenizer)

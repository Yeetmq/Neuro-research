from datasets import Dataset
import gc
import os
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import re
import gc
import os

def tokenize_in_chunks(dataset: Dataset, tokenizer, save_dir: str,  chunk_size=1000):
    os.makedirs(save_dir, exist_ok=True)
    
    total_samples = len(dataset)
    num_chunks = total_samples // chunk_size + 1
    
    for i in range(num_chunks):
        chunk = dataset.select(range(
            i * chunk_size,
            min((i + 1) * chunk_size, total_samples)
        ))
        
        tokenized_chunk = chunk.map(
            lambda examples: tokenizer(
                examples["article"],
                text_target=examples["summary"],
                max_length=1024,
                truncation=True,
                padding=False
            ),
            batched=True,
            batch_size=32,
            remove_columns=["article", "summary"],
            load_from_cache_file=False
        )
        
        tokenized_chunk.save_to_disk(
            os.path.join(save_dir, f"chunk_{i}"),
            max_shard_size="100MB"
        )
        
        # Очистка памяти
        del chunk
        del tokenized_chunk
        gc.collect()
        
        print(f"Processed chunk {i+1}/{num_chunks}")


# if __name__ == '__main__':
    # tokenize_in_chunks(patent_dataset, chunk_size=2000)

from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk, concatenate_datasets, Dataset
import os
import torch

MODEL_CHOICES = {
    "tiny": "sshleifer/distilbart-cnn-12-6",
    "base": "facebook/bart-base",
    "distilled": "sshleifer/distilbart-cnn-12-6",
    "custom": "patrickvonplaten/bart-tiny-random"
}

# Пример использования
MODEL_NAME = MODEL_CHOICES["tiny"]

CHUNKS_DIR = "processed"
OUTPUT_DIR = "bart-finetuned"
BATCH_SIZE = 4
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 256
SEED = 42

# 1. Инициализация модели с оптимизациями памяти
model = BartForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,        # Полуточность
    low_cpu_mem_usage=True,
    gradient_checkpointing=True       # Экономит до 60% памяти
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,                   # Используем быстрый токенизатор
    model_max_length=MAX_INPUT_LENGTH
)

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


def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# train_dataset = load_chunks_optimized(os.path.join(CHUNKS_DIR, 'train'))
# train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=8)


data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8,            # Улучшает производительность на Tensor Cores
    padding='longest',
    max_length=MAX_INPUT_LENGTH
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*4,  # Увеличил для валидации
    gradient_accumulation_steps=4,    # Эмулирует batch_size=32
    learning_rate=3e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    bf16=True,                        # Аппаратное ускорение
    seed=SEED,
    warmup_steps=500,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=8,         # Используем больше ядер CPU
    dataloader_pin_memory=True,       # Ускоряет передачу данных в GPU
    dataloader_prefetch_factor=2,     # Предзагрузка данных
    remove_unused_columns=True,       # Удаляем неиспользуемые столбцы
    optim="adamw_bnb_8bit",           # 8-битный оптимизатор
    report_to="none"                  # Отключаем логирование для ускорения
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=load_chunks_optimized(os.path.join(CHUNKS_DIR, 'train')),
    eval_dataset=load_chunks_optimized(os.path.join(CHUNKS_DIR, 'train')),
    data_collator=data_collator,
    tokenizer=tokenizer
)

try:
    trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving final model...")

trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
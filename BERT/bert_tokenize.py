# bert_tokenize.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizerFast
from tqdm import tqdm

from config import (
    DATA_DIR,
    LABEL_COL,
    FEATURES_DIR,
)

def load_raw_dataset(prefix):
    data_files = {
        "train": os.path.join(DATA_DIR, "processed", f"{prefix}_train.csv"),
        "validation": os.path.join(DATA_DIR, "processed", f"{prefix}_val.csv"),
        "test": os.path.join(DATA_DIR, "processed", f"{prefix}_test.csv"),
    }
    dataset = load_dataset("csv", data_files=data_files)
    return dataset

def tokenize_dataset(dataset, tokenizer, text_col, label_col, max_length):
    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    print("Running tokenization mapping across splits...")
    tokenized = dataset.map(tokenize, batched=True, desc="Tokenizing Dataset")

    if label_col in tokenized["train"].column_names:
        tokenized = tokenized.rename_column(label_col, "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized

def main():
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    prefix = "english"
    text_col = "text"
    max_len = 256

    print(f"\n{'='*50}")
    print(f"PROCESSING DATASET: {prefix.upper()} (BERT)")
    print(f"{'='*50}")
    
    print(f"Loading raw CSV datasets for {prefix}...")
    dataset = load_raw_dataset(prefix)

    print(f"Tokenizing {prefix} datasets (Max Length: {max_len})...")
    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=LABEL_COL,
        max_length=max_len,
    )

    tokens_path = os.path.join(FEATURES_DIR, f'bert_{prefix}_tokenized')
    os.makedirs(tokens_path, exist_ok=True)

    print(f"Saving tokenized {prefix} dataset to {tokens_path} ...")
    tokenized_dataset.save_to_disk(tokens_path)

    print(f"\n✅ Done. BERT tokenized dataset for '{prefix}' saved successfully to disk.")

if __name__ == "__main__":
    main()
# phobert_tokenize.py
import sys
import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from datasets import load_dataset
from transformers import AutoTokenizer
from pyvi import ViTokenizer

from config import (
    DATA_DIR,
    LABEL_COL,
    FEATURES_DIR,
    PHOBERT_MODEL_NAME, 
)

def load_raw_dataset(prefix):
    data_files = {
        "train": os.path.join(DATA_DIR, "processed", f"{prefix}_train.csv"),
        "validation": os.path.join(DATA_DIR, "processed", f"{prefix}_val.csv"),
        "test": os.path.join(DATA_DIR, "processed", f"{prefix}_test.csv"),
    }
    dataset = load_dataset("csv", data_files=data_files)
    return dataset

def tokenize_vietnamese(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    return ViTokenizer.tokenize(str(text))

def tokenize_dataset(dataset, tokenizer, text_col, label_col, max_length):
    print("Applying ViTokenizer (Word Segmentation)...")
    dataset = dataset.map(lambda x: {text_col: tokenize_vietnamese(x[text_col])})

    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    print(f"Tokenizing dataset (Max Length: {max_length})...")
    tokenized = dataset.map(tokenize, batched=True, desc="Tokenizing Dataset")

    if label_col in tokenized["train"].column_names:
        tokenized = tokenized.rename_column(label_col, "labels")

    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return tokenized

def main():
    print("Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)

    prefix = "vietnamese"
    text_col = "text"
    max_len = 256 

    print(f"\n{'='*50}")
    print(f"PROCESSING DATASET: {prefix.upper()} (PhoBERT)")
    print(f"{'='*50}")
    
    print(f"Loading raw CSV datasets for {prefix}...")
    dataset = load_raw_dataset(prefix)

    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=LABEL_COL,
        max_length=max_len,
    )

    output_path = os.path.join(FEATURES_DIR, f'phobert_{prefix}_tokenized')
    print(f"Saving tokenized dataset to: {output_path}")
    tokenized_dataset.save_to_disk(output_path)
    
    print("✅ DONE! Dataset is ready for training.")

if __name__ == "__main__":
    main()
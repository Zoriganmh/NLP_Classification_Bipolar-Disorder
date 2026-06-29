# -*- coding: utf-8 -*-
import sys
import os
import torch
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

current_dir = os.path.dirname(os.path.abspath(__file__)) 
scripts_dir = os.path.dirname(current_dir)               
sys.path.insert(0, scripts_dir)                          

from config import (
    FEATURES_DIR, MODEL_DIR, LOGGING_DIR, NUM_LABELS, 
    LEARNING_RATE, NUM_EPOCHS, PHOBERT_MODEL_NAME
)

def load_tokenized_dataset(prefix):
    tokens_path = os.path.join(FEATURES_DIR, f'phobert_{prefix}_tokenized')
    print(f"Loading tokenized dataset from {tokens_path}...")
    return load_from_disk(tokens_path)

def load_model_and_tokenizer():
    print("Loading PhoBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PHOBERT_MODEL_NAME,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.3,
        use_safetensors=True, 
    )
    return model, tokenizer

def get_training_args(prefix, batch_size):
    use_cuda = torch.cuda.is_available()
    col_model_dir = os.path.join(MODEL_DIR, f"phobert_{prefix}")
    log_dir = os.path.join(LOGGING_DIR, f"phobert_{prefix}")

    return TrainingArguments(
        output_dir=col_model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2, 
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        fp16=use_cuda,
        report_to="none", 
        logging_dir=log_dir
    )

def get_compute_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1)
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(f1_metric.compute(predictions=preds, references=labels, average="macro"))
        return results
        
    return compute_metrics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    prefix = "vietnamese"
    batch_size = 16 

    print(f"\n{'='*50}")
    print(f"TRAINING PHOBERT ON: {prefix.upper()}")
    print(f"{'='*50}")

    tokenized_dataset = load_tokenized_dataset(prefix)
    model, tokenizer = load_model_and_tokenizer()
    training_args = get_training_args(prefix, batch_size)
    compute_metrics = get_compute_metrics()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics,
    )

    print("\n=== STARTING TRAINING ===")
    trainer.train()

    final_dir = os.path.join(MODEL_DIR, f"phobert_{prefix}", "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"\nSaving best model to {final_dir}...")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("✅ TRAINING COMPLETE!")

if __name__ == "__main__":
    main()
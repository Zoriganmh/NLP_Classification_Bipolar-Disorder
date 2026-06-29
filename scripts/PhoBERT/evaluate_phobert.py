# -*- coding: utf-8 -*-
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__)) 
scripts_dir = os.path.dirname(current_dir)               
sys.path.insert(0, scripts_dir)                          

from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DATA_DIR, FEATURES_DIR, MODEL_DIR, OUTPUT_DIR, EVAL_DIR, NUM_LABELS
)

def load_tokenized_dataset(prefix):
    tokens_path = os.path.join(FEATURES_DIR, f'phobert_{prefix}_tokenized')
    return load_from_disk(tokens_path)

def load_model_and_tokenizer(prefix):
    model_dir = os.path.join(MODEL_DIR, f"phobert_{prefix}", "final")
    print(f"Loading trained model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=NUM_LABELS,
    )
    return model, tokenizer

def get_eval_args(prefix, batch_size):
    use_cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f'phobert_{prefix}_eval_tmp'),
        per_device_eval_batch_size=batch_size,
        fp16=use_cuda,
        report_to="none" 
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

def save_metrics(metrics, prefix, split):
    eval_output_dir = os.path.join(EVAL_DIR, f"phobert_{prefix}")
    os.makedirs(eval_output_dir, exist_ok=True)
    
    out_file = os.path.join(eval_output_dir, f"{split}_metrics.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved {split} metrics to {out_file}")

def detailed_evaluation(trainer, dataset, prefix):
    print("\nCreating detailed report and Confusion Matrix for test set...")
    predictions = trainer.predict(dataset["test"])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    target_names = ["ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"]
    eval_output_dir = os.path.join(EVAL_DIR, f"phobert_{prefix}")
    os.makedirs(eval_output_dir, exist_ok=True)

    # 1. Classification Report
    report = classification_report(labels, preds, target_names=target_names)
    print(report)
    
    report_path = os.path.join(eval_output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - PhoBERT ({prefix.capitalize()})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    cm_path = os.path.join(eval_output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    prefix = "vietnamese"
    batch_size = 16

    print(f"\n{'='*50}")
    print(f"EVALUATING PHOBERT ON: {prefix.upper()}")
    print(f"{'='*50}")

    dataset = load_tokenized_dataset(prefix)
    model, tokenizer = load_model_and_tokenizer(prefix)
    eval_args = get_eval_args(prefix, batch_size)
    compute_metrics = get_compute_metrics()

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer, 
        compute_metrics=compute_metrics,
    )

    for split in ["train", "validation", "test"]:
        print(f"\nEvaluating on {split.upper()}...")
        metrics = trainer.evaluate(eval_dataset=dataset[split])
        save_metrics(metrics, prefix, split)

    detailed_evaluation(trainer, dataset, prefix)
    print("✅ EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()
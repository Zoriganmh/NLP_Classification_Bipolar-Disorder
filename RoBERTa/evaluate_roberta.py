# evaluate_roberta.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sns as sns
import seaborn as sns
from datasets import load_from_disk
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    TrainingArguments,
    Trainer,
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DATA_DIR,
    FEATURES_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    EVAL_DIR,
    LOGGING_DIR,
    NUM_LABELS,
)

def load_tokenized_dataset(prefix):
    tokens_path = os.path.join(FEATURES_DIR, f'roberta_{prefix}_tokenized')
    return load_from_disk(tokens_path)

def load_model_and_tokenizer(prefix):
    model_dir = os.path.join(MODEL_DIR, f"roberta_{prefix}", "final")
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=NUM_LABELS,
    )
    return model, tokenizer

def get_eval_args(prefix, batch_size):
    use_cuda = torch.cuda.is_available()
    return TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f'roberta_{prefix}_eval_tmp'),
        per_device_eval_batch_size=batch_size,
        fp16=use_cuda,
        logging_dir=os.path.join(LOGGING_DIR, f"roberta_{prefix}_eval"),
        report_to="none" 
    )

def get_compute_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        results = {}
        results.update(acc_metric.compute(predictions=preds, references=labels))
        results.update(
            f1_metric.compute(
                predictions=preds, references=labels, average="macro"
            )
        )
        return results

    return compute_metrics

def save_metrics(metrics, prefix, split_name):
    roberta_eval_dir = os.path.join(EVAL_DIR, "roberta")
    os.makedirs(roberta_eval_dir, exist_ok=True)

    out_path = os.path.join(roberta_eval_dir, f"roberta_{prefix}_{split_name}_metrics.txt")
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved {prefix} {split_name} metrics to {out_path}")

def save_detailed_evaluation(predictions, labels, prefix):
    roberta_eval_dir = os.path.join(EVAL_DIR, "roberta")
    os.makedirs(roberta_eval_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(DATA_DIR, "processed", f"{prefix}_test.csv"))
    class_mapping = dict(zip(test_df['class_id'], test_df['class_name']))
    target_names = [class_mapping.get(i, f"Class_{i}") for i in range(NUM_LABELS)]

    # 1. Classification Report
    report = classification_report(labels, predictions, target_names=target_names)
    report_path = os.path.join(roberta_eval_dir, f"roberta_{prefix}_detailed_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"Saved detailed report to {report_path}")

    # 2. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - RoBERTa ({prefix.upper()})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    cm_path = os.path.join(roberta_eval_dir, f"roberta_{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    prefix = "english"
    batch_size = 16

    print(f"\n{'='*50}")
    print(f"EVALUATING RoBERTa ON: {prefix.upper()}")
    print(f"{'='*50}")

    print(f"Loading {prefix} tokenized dataset from disk...")
    dataset = load_tokenized_dataset(prefix)
    
    print("Loading RoBERTa model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(prefix)

    eval_args = get_eval_args(prefix, batch_size)
    compute_metrics = get_compute_metrics()

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    for split in ["train", "validation", "test"]:
        print(f"Evaluating on {split} set...")
        metrics = trainer.evaluate(eval_dataset=dataset[split])
        save_metrics(metrics, prefix, split)

    print(f"Running detailed predictions on test set for {prefix}...")
    predictions_output = trainer.predict(dataset["test"])
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    
    save_detailed_evaluation(preds, labels, prefix)
    print(f"\n✅ Evaluation complete for RoBERTa - {prefix}.")

if __name__ == "__main__":
    main()
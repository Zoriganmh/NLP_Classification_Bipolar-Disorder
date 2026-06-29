# evaluate_lstm.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)

import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from train_lstm import PaperLSTM
from config import DATA_DIR, FEATURES_DIR, NUM_LABELS, MODEL_DIR, EVAL_DIR

def load_test_data_and_vocab(prefix):
    data_dir = os.path.join(FEATURES_DIR, f"lstm_{prefix}_preprocessed")
    
    with open(os.path.join(data_dir, "vocab.pkl"), 'rb') as f:
        vocab = pickle.load(f)
        
    test_data = torch.load(os.path.join(data_dir, "test.pt"))
    return vocab, test_data

def get_test_dataloader(test_data, batch_size=32):
    test_dataset = TensorDataset(test_data[0], test_data[1])
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def load_trained_model(prefix, vocab, device):
    model = PaperLSTM(
        vocab_size=len(vocab), 
        num_labels=NUM_LABELS, 
        padding_idx=vocab['<PAD>']
    )
    model_path = os.path.join(MODEL_DIR, f"lstm_{prefix}", "best_lstm.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def save_detailed_evaluation(predictions, labels, prefix):
    lstm_eval_dir = os.path.join(EVAL_DIR, "lstm")
    os.makedirs(lstm_eval_dir, exist_ok=True)
    
    test_df = pd.read_csv(os.path.join(DATA_DIR, "processed", f"{prefix}_test.csv"))
    class_mapping = dict(zip(test_df['class_id'], test_df['class_name']))
    target_names = [class_mapping.get(i, f"Class_{i}") for i in range(NUM_LABELS)]

    report = classification_report(labels, predictions, target_names=target_names)
    report_path = os.path.join(lstm_eval_dir, f"lstm_{prefix}_detailed_report.txt")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(report)
    print(f"Saved detailed report to {report_path}")

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - LSTM ({prefix.upper()})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    cm_path = os.path.join(lstm_eval_dir, f"lstm_{prefix}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    prefix = "english"

    print(f"\n{'='*50}")
    print(f"EVALUATING LSTM ON: {prefix.upper()}")
    print(f"{'='*50}")

    vocab, test_data = load_test_data_and_vocab(prefix)
    test_loader = get_test_dataloader(test_data)
    
    print("Loading trained LSTM model...")
    model = load_trained_model(prefix, vocab, device)

    print(f"Running predictions on test set for {prefix}...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    save_detailed_evaluation(all_preds, all_labels, prefix)
    print(f"\n✅ Evaluation complete for {prefix}.")

if __name__ == "__main__":
    main()
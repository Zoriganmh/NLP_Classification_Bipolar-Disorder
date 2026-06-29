# train_lstm.py
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) 
sys.path.append(root_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm 

from config import FEATURES_DIR, NUM_LABELS, MODEL_DIR

class PaperLSTM(nn.Module):
    def __init__(self, vocab_size, num_labels, padding_idx, embedding_dim=100, hidden_dim=256, num_layers=2, dropout_prob=0.5):
        super(PaperLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_dim, 
            num_layers=num_layers, batch_first=True, dropout=dropout_prob
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1, :, :])
        return self.fc(out)

def load_preprocessed_data(prefix):
    data_dir = os.path.join(FEATURES_DIR, f"lstm_{prefix}_preprocessed")
    
    with open(os.path.join(data_dir, "vocab.pkl"), 'rb') as f:
        vocab = pickle.load(f)
        
    train_data = torch.load(os.path.join(data_dir, "train.pt"), weights_only=True)
    val_data = torch.load(os.path.join(data_dir, "val.pt"), weights_only=True)
    
    return vocab, train_data, val_data

def get_dataloaders(train_data, val_data, batch_size):
    train_dataset = TensorDataset(train_data[0], train_data[1])
    val_dataset = TensorDataset(val_data[0], val_data[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    prefix = "english"
    batch_size = 32

    print(f"\n{'='*50}")
    print(f"TRAINING LSTM ON: {prefix.upper()} (COMBINED TEXT)")
    print(f"{'='*50}")

    print(f"Loading {prefix} preprocessed dataset from disk...")
    vocab, train_data, val_data = load_preprocessed_data(prefix)
    train_loader, val_loader = get_dataloaders(train_data, val_data, batch_size)

    print(f"Initializing LSTM model with {NUM_LABELS} classes...")
    model = PaperLSTM(
        vocab_size=len(vocab), 
        num_labels=NUM_LABELS, 
        padding_idx=vocab['<PAD>']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    best_f1 = 0.0
    epochs = 25
    
    col_model_dir = os.path.join(MODEL_DIR, f"lstm_{prefix}")
    os.makedirs(col_model_dir, exist_ok=True)
    best_model_path = os.path.join(col_model_dir, "best_lstm.pth")

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", leave=False)
        
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        all_preds, all_labels = [], []
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Val]", leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"    --> New best model saved with F1: {best_f1:.4f}")

    print(f"\n✅ Training complete for {prefix}. Best model saved to {best_model_path}\n")

if __name__ == "__main__":
    main()
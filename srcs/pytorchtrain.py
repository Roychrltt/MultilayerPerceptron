import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("M1 GPU is ready!")
else:
    device = torch.device("cpu")


BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
PATIENCE = 10


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class MLP(nn.Module):
    """Dynamic Multi-Layer Perceptron.

    Args:
        input_size (int): Number of input features.
        hidden_layers (list): List of neurons per hidden layer (e.g., [64, 32, 16]).
    """
    def __init__(self, input_size, hidden_layers):
        super().__init__()
        layers = []
        last_dim = input_size

        for h_dim in hidden_layers:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


def load_and_tensorize(path):
    """ Safely loads CSV and converts to float32 tensors. """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{Color.RED}[ERROR] Missing data file: {path}{Color.END}")

    df = pd.read_csv(path, header=None)
    X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32)
    return X, y


def _plot_results(history):
    """ Generates side-by-side plots for Loss and Accuracy. """
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def train_model(hidden_config=[64, 64, 32]):
    X_train, y_train = load_and_tensorize("data/train.csv")
    X_val, y_val = load_and_tensorize("data/val.csv")

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    joblib.dump(scaler, 'data/scaler.joblib')
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = MLP(input_size=X_train.shape[1], hidden_layers=hidden_config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"{Color.GREEN}Training model with input_dim={X_train.shape[1]} and layers={hidden_config}{Color.END}")

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss, train_acc = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_acc += (preds == y).sum().item()

        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(1)
                val_loss += criterion(logits, y).item() * x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_acc += (preds == y).sum().item()

        epoch_train_loss = train_loss / len(X_train)
        epoch_val_loss = val_loss / len(X_val)
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(train_acc / len(X_train))
        history['val_acc'].append(val_acc / len(X_val))

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            if not os.path.exists("model"):
                os.makedirs("model")
            torch.save(model.state_dict(), "model/best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Accuracy: {train_acc / len(X_train):.4f} | Val Accuracy: {val_acc / len(X_val):.4f}")

    _plot_results(history)


def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == y).float().mean()


def main():
    train_model()


if __name__ == "__main__":
    main()

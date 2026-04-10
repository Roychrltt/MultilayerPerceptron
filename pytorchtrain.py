import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]
X_val = val_df.iloc[:, 1:]
y_val = val_df.iloc[:, 0]
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
model = MLP()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


def accuracy_from_logits(logits, y):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == y).float().mean()


def train():
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(100):
        model.train()
        epoch_loss, epoch_acc = 0, 0

        for x, y in train_loader:
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += accuracy_from_logits(logits, y).item()

        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(epoch_acc / len(train_loader))

        # ---- VALIDATION ----
        model.eval()
        val_loss, val_acc = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)

                val_loss += loss.item()
                val_acc += accuracy_from_logits(logits, y).item()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))

        print(
            f"Epoch {epoch+1:02d} | "
            f"Train loss: {train_losses[-1]:.4f}, acc: {train_accs[-1]:.4f} | "
            f"Val loss: {val_losses[-1]:.4f}, acc: {val_accs[-1]:.4f}"
        )
        scheduler.step()

    plt.figure(figsize=(12,5))

    epochs_range = range(100)
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label="training loss")
    plt.plot(epochs_range, val_losses, label="validation loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accs, label="training acc")
    plt.plot(epochs_range, val_accs, label="validation acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
    print(X_train_t.shape)
    print(X_val_t.shape)


def main():
    train()


if __name__ == "__main__":
    main()

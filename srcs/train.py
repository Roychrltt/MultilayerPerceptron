import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import argparse
from nn import *


def binary_cross_entropy(logit, y):
    x = logit
    max_part = x.relu()
    log_part = ((-x.abs()).exp() + Value(1)).log()
    loss_matrix = max_part - x * y + log_part
    return loss_matrix.mean()


def cross_entropy(probs, label):
    return -probs[label].log()


def softmax(logits):
    exps = [l.exp() for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP from scratch")
    parser.add_argument("train_set", type=str, help="Path to training CSV")
    parser.add_argument("val_set", type=str, help="Path to validation CSV")
    parser.add_argument("--layers", type=int, nargs="+", default=[64, 64, 32, 1], help="Layer sizes")
    parser.add_argument("--lr", type=float, default=0.007, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    return parser.parse_args()


def main():
    if len(sys.argv) < 3:
        print("Usage: python train.py <train_set.csv> <validation_set.csv> [options]")
        return

    args = parse_args()
    train_df = pd.read_csv(sys.argv[1], header=None)
    val_df = pd.read_csv(sys.argv[2], header=None)

    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    X_val = val_df.iloc[:, 1:].values
    y_val = val_df.iloc[:, 0].values

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_train, y_train = Value(X_train_std), Value(y_train.reshape(-1, 1))
    X_val, y_val = Value(X_val_std), Value(y_val.reshape(-1, 1))

    mlp = MLP(X_train.data.shape[1], args.layers)

    best_loss = float('inf')
    wait = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(args.epochs):
        mlp.zero_grad()
        logits = mlp(X_train)
        loss = binary_cross_entropy(logits, y_train)
        loss.backward()

        for p in mlp.parameters():
            p.data -= args.lr * p.grad

        train_preds = (logits.data > 0).astype(int)
        train_acc = np.mean(train_preds == y_train.data)

        val_logits = mlp(X_val)
        val_loss = binary_cross_entropy(val_logits, y_val)
        val_preds = (val_logits.data > 0).astype(int)
        val_acc = np.mean(val_preds == y_val.data)

        history['train_loss'].append(loss.data)
        history['val_loss'].append(val_loss.data)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}: train_loss={loss.data:.8f}, train_acc={train_acc:.8f}, val_loss={val_loss.data:.8f}, val_acc={val_acc:.8f}")

        if val_loss.data < best_loss:
            best_loss = val_loss.data
            wait = 0
            model_data = {
                "layers": args.layers,
                "weights": [p.data for p in mlp.parameters()],
                "mean": mean,
                "std": std
            }
            with open("model/model.pkl", "wb") as f:
                pickle.dump(model_data, f)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping as epoch {epoch + 1}")
                break

    print("\033[1;92m✔ Training phase finished. Model data saved to model/model.pkl\033[0m")
    epochs_range = range(len(history['train_loss']))
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, history['train_loss'], label="training loss")
    plt.plot(epochs_range, history['val_loss'], label="validation loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, history['train_acc'], label="training acc")
    plt.plot(epochs_range, history['val_acc'], label="validation acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

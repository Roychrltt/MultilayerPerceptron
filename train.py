from describe import read_data
from describe import clean_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import *
import sys
import pickle


def binary_cross_entropy(logit, y):
    x = logit
    max_part = x.relu()
    log_part = ((-x.abs()).exp() + Value(1)).log()
    return max_part - x*y + log_part


def cross_entropy(probs, label):
    return -probs[label].log()


def softmax(logits):
    exps = [l.exp() for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


def main():
    if len(sys.argv) != 3:
        print("Usage: python train.py <train_set.csv> <validation_set.csv>")
        sys.exit(1)
    
    train = sys.argv[1]
    val = sys.argv[2]

    train_df = pd.read_csv(train)
    val_df = pd.read_csv(val)

    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values

    X_val = val_df.iloc[:, 1:].values
    y_val = val_df.iloc[:, 0].values

    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)

    X_train_norm = (X_train - min_val) / (max_val - min_val)
    X_val_norm = (X_val - min_val) / (max_val - min_val)
    mlp = MLP(30, [16, 8, 2])

    lr = 0.001
    epochs = 100
    prev_loss = 1
    patience = 5
    wait = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for x, y in zip(X_train_norm, y_train):
            x = x.tolist()
            mlp.zero_grad()

            logit = mlp(x)
            probs = softmax(logit)
            loss = cross_entropy(probs, y)
            total_loss += loss.data
            loss.backward()

            for p in mlp.parameters():
                p.data -= lr * p.grad

            pred = 0 if probs[0].data > probs[1].data else 1
            correct += (pred == y)

        train_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        total_loss = 0.0
        correct = 0
        
        for x, y in zip(X_val_norm, y_val):
            logits = mlp(x.tolist())
            probs = softmax(logits)
            loss = cross_entropy(probs, y)
            total_loss += loss.data
            
            pred = 0 if probs[0].data > probs[1].data else 1
            correct += (pred == y)
        
        val_loss = total_loss / len(X_val)
        val_acc = correct / len(X_val)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.8f}, train_acc={train_acc:.8f}, val_loss={val_loss:.8f}, val_acc={val_acc:.8f}")

        if val_loss < prev_loss:
            prev_loss = val_loss
            wait = 0
            model_data = {
                "input": 30,
                "layers": [16, 8, 2],
                "weights": [p.data for p in mlp.parameters()]
            }
            with open("model.pkl", "wb") as f:
                pickle.dump(model_data, f)
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping as epoch {epoch + 1}")
            break

    print("\033[1;92m✔ Training phase finished. Model data saved to model.pkl\033[0m")
    epochs_range = range(len(train_losses))
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label="training loss")
    plt.plot(epochs_range, val_losses, label="validation loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_accuracies, label="training acc")
    plt.plot(epochs_range, val_accuracies, label="validation acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

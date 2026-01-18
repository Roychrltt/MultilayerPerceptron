from describe import read_data
from describe import clean_data
import numpy as np
import matplotlib.pyplot as plt
from nn import *


def data_split(df):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    np.random.seed(0)
    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


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
    filename = "data.csv"
    df = read_data(filename)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = data_split(df)
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)

    X_train_norm = (X_train - min_val) / (max_val - min_val)
    X_test_norm = (X_test - min_val) / (max_val - min_val)
    mlp = MLP(30, [16, 8, 2])

    lr = 0.005
    epochs = 50
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
        
        for x, y in zip(X_test_norm, y_test):
            logits = mlp(x.tolist())
            probs = softmax(logits)
            loss = cross_entropy(probs, y)
            total_loss += loss.data
            
            pred = 0 if probs[0].data > probs[1].data else 1
            correct += (pred == y)
        
        val_loss = total_loss / len(X_test)
        val_acc = correct / len(X_test)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.8f}, train_acc={train_acc:.8f}, val_loss={val_loss:.8f}, val_acc={val_acc:.8f}")
        
    epochs_range = range(epochs)
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

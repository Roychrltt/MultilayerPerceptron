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
    return max_part - x*y + log_part


def cross_entropy(probs, label):
    return -probs[label].log()


def softmax(logits):
    exps = [l.exp() for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def predict_class(probs):
    return max(range(len(probs)), key=lambda i: probs[i].data)


def main():
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    train_df = train_df.apply(pd.to_numeric)
    val_df = val_df.apply(pd.to_numeric)

    X_train = train_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y_train = train_df.iloc[:, 0].to_numpy(dtype=np.int64)

    X_val = val_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    y_val = val_df.iloc[:, 0].to_numpy(dtype=np.int64)

    X_train_norm = X_train / 255
    X_val_norm = X_val / 255
    numf = X_train.shape[1]
    mlp = MLP(numf, [int(numf / 2), int(numf / 4), int(numf / 8), 10])

    batch_size = 64
    lr = 0.005
    epochs = 10
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        print("epoch")
        total_loss = 0.0
        correct = 0
        indices = np.random.permutation(len(X_train_norm))
        X_train_norm = X_train_norm[indices]
        y_train = y_train[indices]

        for i in range(0, len(X_train_norm), batch_size):
            print(i)
            X_batch = X_train_norm[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            mlp.zero_grad()
            batch_loss = Value(0.0)

            for x, y in zip(X_batch, y_batch):
                x_vals = [Value(float(v)) for v in x]   # 🔥 THIS LINE
                logits = mlp(x_vals)
                probs = softmax(logits)
                loss = cross_entropy(probs, y)
                batch_loss += loss

                pred = predict_class(probs)
                correct += (pred == y)

            batch_loss = batch_loss * (1.0 / len(X_batch))
            total_loss += batch_loss.data
            batch_loss.backward()

            for p in mlp.parameters():
                p.data -= lr * p.grad

        train_loss = total_loss / (len(X_train_norm) / batch_size)
        train_acc = correct / len(X_train_norm)

        total_loss = 0.0
        correct = 0

        for x, y in zip(X_val_norm, y_val):
            x_vals = [Value(float(v)) for v in x]
            logits = mlp(x_vals)
            probs = softmax(logits)
            loss = cross_entropy(probs, y)
            total_loss += loss.data

            pred = predict_class(probs)
            correct += (pred == y)

        val_loss = total_loss / len(X_val_norm)
        val_acc = correct / len(X_val_norm)

               
        print(f"correct predicts: {correct}/{len(X_val)}")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.8f}, train_acc={train_acc:.8f}, val_loss={val_loss:.8f}, val_acc={val_acc:.8f}")

    model_data = {
                "input": numf,
                "layers": [int(numf / 2), int(numf / 4), int(numf / 8), 10],
                "weights": [p.data for p in mlp.parameters()]
                }
    with open("data/model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("\033[1;92m✔ Training phase finished. Model data saved to data/model.pkl\033[0m")
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

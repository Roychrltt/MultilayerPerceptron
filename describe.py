import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from nn import *


def data_split(df):
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    idx = np.random.permutation(len(X))
    split = int(0.8 * len(X))

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def data_describe(filename):
    cols = [""] + ["diagnosis"] + [f"No.{i}" for i in range (1, 31)]

    df = pd.read_csv(filename, header=None, names=cols)
    df = df.drop(df.columns[0], axis=1)

    pd.set_option("display.max_columns", None)
    print(df.describe())
    pd.reset_option("display.max_columns")

    print(df.iloc[:, 0].value_counts())
    df.iloc[:, 0] = df.iloc[:, 0].map({"M":1, "B":0})
    df.iloc[:, 1:].hist(figsize=(15,10), bins=100)
    plt.tight_layout()
    #plt.show()
    df.iloc[:, 1:].plot(kind="box", figsize=(15,5))
    plt.tight_layout()
    #plt.show()

    return df

def binary_cross_entropy(logit, y):
    x = logit
    max_part = x.relu()
    log_part = ((-x.abs()).exp() + Value(1)).log()
    return max_part - x*y + log_part



def main():
    filename = "data.csv"
    df = data_describe(filename)
    X_train, X_test, y_train, y_test = data_split(df)
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)

    X_train_norm = (X_train - min_val) / (max_val - min_val)
    X_test_norm = (X_test - min_val) / (max_val - min_val)
    mlp = MLP(30, [16, 8, 1])

    lr = 0.001
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in zip(X_train_norm, y_train):
            x = x.tolist()

            mlp.zero_grad()
            logit = mlp(x)
            loss = binary_cross_entropy(logit, y)
            total_loss += loss.data

            loss.backward()

            for p in mlp.parameters():
                p.data -= lr * p.grad

        print(f"epoch {epoch}, loss {total_loss / len(X_train)}")

    correct = 0

    for x, y in zip(X_test_norm, y_test):
        p = mlp(x.tolist()).relu().data
        pred = 1 if p > 0.5 else 0
        correct += (pred == y)

    accuracy = correct / len(X_test)
    print("test accuracy:", accuracy)


if __name__ == "__main__":
    main()

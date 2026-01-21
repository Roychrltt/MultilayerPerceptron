import pickle
import pandas
from nn import *
import pandas as pd
from train import softmax
import numpy as np

def main():
    mlp = MLP(16, [8, 4, 2])

    with open("data/model.pkl", "rb") as f:
        model_data = pickle.load(f)
    
    mlp = MLP(
        model_data["input"],
        model_data["layers"]
    )

    for p, w in zip(mlp.parameters(), model_data["weights"]):
        p.data = w

    test_df = pd.read_csv("data/val.csv")
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values
    min_val = X_test.min(axis=0)
    max_val = X_test.max(axis=0)

    X_test_norm = (X_test - min_val) / (max_val - min_val)

    correct = 0
    total_loss = 0.0
    e = 1e-8

    for x, y in zip(X_test_norm, y_test):
        logits = mlp(x.tolist())
        probs = softmax(logits)

        pred = 0 if probs[0].data > probs[1].data else 1
        correct += (pred == y)

        p1 = probs[1].data
        total_loss += -(y * np.log(p1 + e) + (1 - y) * np.log(1 - p1 + e))

    accuracy = correct / len(X_test_norm)
    loss = total_loss / len(X_test_norm)

    print("Test accuracy:", accuracy)
    print("Test BCE loss:", loss)


if __name__ == "__main__":
    main()

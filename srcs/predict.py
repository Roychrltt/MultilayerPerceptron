import pickle
import pandas as pd
import numpy as np
from nn import *
from train import softmax

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

def main():
    try:
        with open("model/model.pkl", "rb") as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print(f"{Color.RED}Error: model/model.pkl not found. Run train.py first. {Color.END}")
        return
    test_df = pd.read_csv("data/val.csv", header=None)
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, 0].values.reshape(-1, 1)

    X_test = (X_test - model_data["mean"]) / model_data["std"]
    mlp = MLP(X_test.shape[1], model_data["layers"])
   
    for p, w in zip(mlp.parameters(), model_data["weights"]):
        p.data = w

    logits = mlp(Value(X_test))

    predictions = (logits.data > 0).astype(int)
    accuracy = np.mean(predictions == y_test)

    x = logits.data
    loss_matrix = np.maximum(0, x) - x * y_test + np.log(1 + np.exp(-np.abs(x)))
    test_loss = np.mean(loss_matrix)

    print(f"{Color.GREEN}Test Results:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"   BCE Loss: {test_loss:.6f}{Color.END}")


if __name__ == "__main__":
    main()

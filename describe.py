import pandas as pd
import matplotlib.pyplot as plt
import sys


def main():
    filename = "data.csv"
    cols = [""] + ["diagnosis"] + [f"No.{i}" for i in range (1, 31)]
    df = pd.read_csv(filename, header=None, names=cols)
    df = df.drop(df.columns[0], axis=1)
    pd.set_option("display.max_columns", None)
    print(df.describe())
    pd.reset_option("display.max_columns")
    print(df.iloc[:, 0].value_counts())
    print(df.iloc[:, 0].unique())
    df.iloc[:, 0] = df.iloc[:, 0].map({"M":1, "B":0})
    df.iloc[:, 1:].hist(figsize=(15,10), bins=100)
    plt.tight_layout()
    plt.show()
    df.iloc[:, 1:].plot(kind="box", figsize=(15,5))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def read_data(filename):
    """Read data in csv with pandas and return the df"""
    df = pd.read_csv(filename, header=None)
    df = df.drop(df.columns[0], axis=1)
    cols = ["label"] + [f"No.{i}" for i in range (1, 31)]
    df.columns = cols
    df.iloc[:, 0] = df.iloc[:, 0].map({"M":1, "B":0})
    return df


def data_describe(df):
    """Describe data with pandas.describe() and visualize data with scatter plots"""
    pd.set_option("display.max_columns", None)
    print(df.describe())
    pd.reset_option("display.max_columns")
    print(df.iloc[:, 0].value_counts())

    mask_M = df["label"] == 1
    mask_B = df["label"] == 0
    fig, axes = plt.subplots(5, 6, figsize=(18, 10))
    axes = axes.flatten()
    features = df.columns[1:]
    for i, col in enumerate(features):
        ax = axes[i]

        ax.scatter(
            df.index[mask_M],
            df.loc[mask_M, col],
            label="M",
            alpha=0.6,
            s=8
        )

        ax.scatter(
            df.index[mask_B],
            df.loc[mask_B, col],
            label="B",
            alpha=0.6,
            s=8
        )

        ax.set_title(col)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    #plt.show()


def clean_data(df):
    useful = ["label"] + [f"No.{i}" for i in [1,3,4,6,7,8,11,13,14,21,22,23,24,26,27,28]]
    df = df[useful]
    return df


def data_split_and_save(df, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split = int(0.8 * len(df))

    train_idx = idx[:split]
    val_idx = idx[split:]

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    os.makedirs("./data", exist_ok=True)
    df_train.to_csv("./data/train.csv", index=False)
    df_val.to_csv("./data/val.csv", index=False)
    

def main():
    filename = "./data/data.csv"
    df = read_data(filename)
    data_describe(df)
    df = clean_data(df)
    data_split_and_save(df, 10)
    print("\033[1;92m✔ Data visualization and split done.\033[0m")
    print("\033[1;92mSplit data saved as data/train.csv ans data/val.csv\033[0m")
 

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def data_split_and_save(df, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split = int(0.8 * len(df))

    train_idx = idx[:split]
    val_idx = idx[split:]

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    df_train.to_csv("train.csv", index=False)
    df_val.to_csv("val.csv", index=False)
    

def main():
    filename = "data.csv"
    df = pd.read_csv(filename,header=None,dtype=np.float32,skiprows=1,low_memory=False)
    data_split_and_save(df, 10)
 

if __name__ == "__main__":
    main()

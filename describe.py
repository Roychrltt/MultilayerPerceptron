import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats


def read_clean_data(filename, handle_nan='mean'):
    """ Use pandas to read from data.csv, check for consistent types per column and handle NaNs.

        handle_nan: 'mean' to fill with average, 'drop' to remove rows.

        The first column (patient ID) is dropped.
        The lable column is converted from str to int to facilitate later training.

    """
    df = pd.read_csv(filename, header=None)
    #print(df.describe().T)

    df = df.drop(df.columns[0], axis=1)
    df.columns = range(df.shape[1])
    n_col = df.shape[1]
    valid_labels = ['M', 'B']
    initial_count = len(df)
    df = df[df[0].isin(valid_labels)].copy()

    if len(df) < initial_count:
        print(f"🗑️ Dropped {initial_count - len(df)} samples with invalid labels.")

    mapping = {'M': 1, 'B': 0}
    df[df.columns[0]] = df[df.columns[0]].map(mapping)

    for col in range(1, n_col):
        series = df[col]
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        if numeric_series.isna().sum() > series.isna().sum():
            print(f"⚠️ Column {col} had mixed types. Non-numeric values converted to NaN.")

        if numeric_series.isna().any():
            if handle_nan == 'mean':
                mean_val = numeric_series.mean()
                numeric_series = numeric_series.fillna(mean_val)
                print(f"🩹 Column {col}: Filled NaNs with mean ({mean_val:.4f})")
            elif handle_nan == 'drop':
                df = df.dropna(subset=[col])
                numeric_series = df[col]
                print(f"🗑️ Column {col}: Dropped rows containing NaN")
        
        df[col] = numeric_series

    return df


def drop_noisy_features(df, threshold=1e-21):
    """ Use Mann-Whitney U test to drop noisy features where two class do not differ significantly.
        If p_value is higher than threshold, it means the feature is noisy.

    """
    class1 = df[df.iloc[:, 0] == 0]
    class2 = df[df.iloc[:, 0] == 1]

    good_features = [df.columns[0]] # To keep the lable column
    n_features = df.shape[1]
    for i in range(1, n_features):
        col = df.columns[i]
        stat, p_value = stats.mannwhitneyu(class1[col], class2[col])
        if p_value < threshold:
            good_features.append(col)
    return df[good_features]


def data_split_and_save(df):
    """ Split the data into train, val, and test set, and save them into corresponding files.

    """
    idx = np.random.permutation(len(df))
    split1 = int(0.7 * len(df))
    split2 = int(0.85 * len(df))

    train_idx = idx[:split1]
    val_idx = idx[split1:split2]
    test_idx = idx[split2:]

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    
    df_train.to_csv("train.csv", index=False, header=False)
    df_val.to_csv("val.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)
    

def main():
    """ Main function to read from data.csv, clean the data and split it into train, val, and test set

    """
    if len(sys.argv) != 2:
        print("Usage: python3 describe.py <path_to_csv>")
        return
    filename = sys.argv[1]
    df = read_clean_data(filename)
    df = drop_noisy_features(df)
    data_split_and_save(df)
 

if __name__ == "__main__":
    main()

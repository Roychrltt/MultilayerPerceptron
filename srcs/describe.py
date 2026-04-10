import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler


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

def read_clean_data(filename, handle_nan='mean'):
    """ Use pandas to read from data.csv, check for consistent types per column and handle NaNs.

        handle_nan: 'mean' to fill with average, 'drop' to remove rows.

        The first column (patient ID) is dropped.
        The label column is converted from str to int to facilitate later training.

    """
    df = pd.read_csv(filename, header=None)
    print('='*45+' Describe Begin '+'='*45)
    print(df.describe().T)
    print('='*45+' Describe End '+'='*45)

    df = df.drop(df.columns[0], axis=1)
    cols = ["label"] + [f"No.{i}" for i in range(1, 31)]
    df.columns = cols
    n_col = df.shape[1]
    valid_labels = ['M', 'B']
    initial_count = len(df)
    df = df[df['label'].isin(valid_labels)].copy()

    if len(df) < initial_count:
        print(f"🗑️ Dropped {initial_count - len(df)} samples with invalid labels.")

    df["label"] = df["label"].map({'M': 1, 'B': 0})

    for col in df.columns[1:]:
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


def data_visualise(df):
    """ Visualise data distribution for different features.

    """
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
    plt.show()
    plt.close(fig)


def drop_noisy_features(df, threshold=1e-21):
    """ Use Mann-Whitney U test to drop noisy features where two class do not differ significantly.
        If p_value is higher than threshold, it means the feature is noisy.

    """
    class1 = df[df.iloc[:, 0] == 0]
    class2 = df[df.iloc[:, 0] == 1]

    good_features = [df.columns[0]] # To keep the label column
    n_features = df.shape[1]
    for i in range(1, n_features):
        col = df.columns[i]
        stat, p_value = stats.mannwhitneyu(class1[col], class2[col])
        if p_value < threshold:
            good_features.append(col)
        else:
            print(f'{Color.YELLOW}Feature {i} being dropped.{Color.END}')
    return df[good_features]


def data_split_and_save(df, output_dir='data'):
    """ Split the data into train, val, and test set, and save them into corresponding files.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    idx = np.random.permutation(len(df))
    split1 = int(0.7 * len(df))
    split2 = int(0.85 * len(df))

    train_idx = idx[:split1]
    val_idx = idx[split1:split2]
    test_idx = idx[split2:]

    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_test = df.iloc[test_idx]
    scaler = StandardScaler()
    scaler.fit(df_train.iloc[:, 1:])
    
    df_train.iloc[:, 1:] = scaler.transform(df_train.iloc[:, 1:])
    df_val.iloc[:, 1:] = scaler.transform(df_val.iloc[:, 1:])
    df_test.iloc[:, 1:] = scaler.transform(df_test.iloc[:, 1:])
    df_train.to_csv("train.csv", index=False, header=False)
    df_val.to_csv("val.csv", index=False, header=False)
    df_test.to_csv("test.csv", index=False, header=False)
    print(f'{Color.GREEN}Data preprocessing finished. Results saved to data/train.csv, data/val.csv, and data/test.csv.{Color.END}')
    

def main():
    """ Main function to read from data.csv, clean the data and split it into train, val, and test set. """
    if len(sys.argv) < 2:
        print("Usage: python3 describe.py <path_to_csv> [--visual]")
        return
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"{Color.RED}[ERROR]{Color.END} File '{filename}' not found.")
        return

    df = read_clean_data(filename)
    if df.empty:
        print(f"{Color.RED}[ERROR]{Color.END} No valid data remaining after cleaning.")
        return
    
    if len(sys.argv) > 2 and sys.argv[2] == "--visual":
        data_visualise(df)
    df = drop_noisy_features(df)
    data_split_and_save(df)
 

if __name__ == "__main__":
    main()

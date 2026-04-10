import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

model = models.Sequential([
    layers.Input(shape=(16,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
X_train = train_df.iloc[:, 1:]
y_train = train_df.iloc[:, 0]
X_val = val_df.iloc[:, 1:]
y_val = val_df.iloc[:, 0]

model.fit(X_train, y_train, batch_size=32, epochs=20)

proba = model.predict(X_val)

print(proba)

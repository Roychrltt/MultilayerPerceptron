import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler


BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3


def build_dynamic_model(input_shape, hidden_layers=[64, 32, 16]):
    """ Creates a sequential model with dynamic hidden layers. """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))

    for neurons in hidden_layers:
        model.add(layers.Dense(neurons, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def _plot_history(history):
    """ Generates Loss and Accuracy plots from the history object. """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_tensorflow_model(hidden_config=[64, 32, 16]):
    train_df = pd.read_csv("data/train.csv", header=None)
    val_df = pd.read_csv("data/val.csv", header=None)

    X_train, y_train = train_df.iloc[:, 1:].values, train_df.iloc[:, 0].values
    X_val, y_val = val_df.iloc[:, 1:].values, val_df.iloc[:, 0].values

    model = build_dynamic_model(input_shape=X_train.shape[1], hidden_layers=hidden_config)
    model.summary()

    my_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath='model/best_tf_model.h5', monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=my_callbacks,
        verbose=1
    )
    _plot_history(history)


if __name__ == "__main__":
    train_tensorflow_model(hidden_config=[64, 32, 16])

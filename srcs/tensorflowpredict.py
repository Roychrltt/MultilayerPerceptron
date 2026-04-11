import tensorflow as tf
import pandas as pd
import numpy as np
import os

def predict_with_tensorflow(model_path='model/best_tf_model.h5', test_path='data/test.csv'):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
    
    model = tf.keras.models.load_model(model_path) 
    
    test_df = pd.read_csv(test_path, header=None)
    y_test = test_df.iloc[:, 0].values
    X_test = test_df.iloc[:, 1:].values

    X_test_scaled = X_test

    probabilities = model.predict(X_test_scaled, verbose=0).flatten()

    predictions = (probabilities > 0.5).astype(int)

    accuracy = np.mean(predictions == y_test)
    
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions,
        'Confidence': probabilities
    })
    pd.set_option('display.max_rows', None)
    print(results)

    return predictions

if __name__ == "__main__":
    predict_with_tensorflow()

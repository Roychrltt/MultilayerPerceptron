import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from pytorchtrain import MLP, load_and_tensorize, Color

def evaluate_on_test(model_path="model/best_model.pth", test_path="data/test.csv", hidden_config=[64, 32, 16]):
    """ Loads model and tests the model's prediction accuracy. """
    X_test, y_test = load_and_tensorize(test_path)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = MLP(input_size=X_test.shape[1], hidden_layers=hidden_config)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    model.eval() 
    
    total_correct = 0
    all_preds = []
    
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            total_correct += (preds == y).sum().item()
            all_preds.extend(preds.numpy())

    accuracy = total_correct / len(y_test)
    print(f"✅ Evaluation Complete!")
    print(f"📊 Test Accuracy: {accuracy * 100:.2f}%")
    
    return all_preds

if __name__ == "__main__":
    evaluate_on_test()

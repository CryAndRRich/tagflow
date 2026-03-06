from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch

def run_inference(model: torch.nn.Module, 
                  loader: torch.utils.data.DataLoader, 
                  attribute_list: List[int], 
                  device: torch.device) -> Dict[str, List[int]]:
    """
    Chạy inference trên tập dữ liệu và thu thập dự đoán cho mỗi thuộc tính
    """
    model.eval()
    all_predictions = {f"attr_{i}": [] for i in attribute_list}
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                batch_x = batch[0]
            else:
                batch_x = batch
                
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            for i, j in enumerate(attribute_list):
                preds = torch.argmax(outputs[i], dim=1).cpu().numpy()
                all_predictions[f"attr_{j}"].extend(preds)

    return all_predictions


def evaluate_em(y_true: np.ndarray, 
                y_pred: np.ndarray) -> float:
    """
    Tính toán độ chính xác dựa trên số lượng mẫu có dự đoán chính xác hoàn toàn
    """
    y_true = y_true.astype(np.int64)
    y_pred = np.round(y_pred).astype(np.int64)

    exact_matches = np.all(y_true == y_pred, axis=1)
    
    accuracy = np.mean(exact_matches)
    return accuracy


def get_stats(val_predictions: Dict[str, List[int]], 
              y_true: pd.DataFrame, 
              attribute_list: List[int],
              attribute_cols: List[str]) -> None:
    """
    Tính toán và in ra điểm F1 macro cho từng thuộc tính và độ chính xác hoàn toàn
    """
    y_pred_val = np.column_stack([val_predictions[f"attr_{i}"] for i in attribute_list])
    y_true = y_true[attribute_cols].values

    macro_f1_scores = []

    for i, j in enumerate(attribute_list):
        f1_i = f1_score(y_true[:, i], y_pred_val[:, i], average="macro")
        macro_f1_scores.append(f1_i)
        print(f"Attribute {j} Val Macro F1: {f1_i:.4f}")

    exact_match_acc = evaluate_em(y_true, y_pred_val)
    print(f"Exact-Match Val Accuracy: {exact_match_acc:.4f}")
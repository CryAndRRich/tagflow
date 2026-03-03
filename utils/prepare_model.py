from typing import Any, List, Dict, Tuple
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.optim as optim

from ..model import get_model
from ..preprocess import DataManager

def get_loss_functions(y_df: torch.Tensor, 
                       attr_cols: List[str],
                       num_classes_list: List[int], 
                       label_smoothing: float,
                       device=torch.device) -> List[torch.nn.Module]:
    """
    Khởi tạo hàm mất mát có trọng số cho mỗi thuộc tính
    """
    loss_fns = []
    for i, col in enumerate(attr_cols):
        y_true = y_df[col].values
        classes = np.unique(y_true)
        raw_weights = compute_class_weight("balanced", classes=classes, y=y_true)
        
        smoothed_weights = np.sqrt(raw_weights)
        clipped_weights = np.clip(smoothed_weights, a_min=None, a_max=10.0)
        
        weight_tensor = np.ones(num_classes_list[i], dtype=np.float32)
        for cls, weight in zip(classes, clipped_weights):
            weight_tensor[cls] = weight
            
        weight_tensor = torch.tensor(weight_tensor, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(
            weight=weight_tensor, 
            label_smoothing=label_smoothing, 
            reduction="none"
        )
        
        loss_fns.append(criterion)
        
    return loss_fns


def update_model_kwargs(data: DataManager,
                        model_kwargs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if "vocab_size" in model_kwargs.keys():
        model_kwargs["vocab_size"] = data.VOCAB_SIZE

    if "num_classes_list" in model_kwargs.keys():
        model_kwargs["num_classes_list"] = data.NUM_CLASSES_LIST

    if "w2v_tensor" in model_kwargs.keys():
        model_kwargs["w2v_tensor"] = data.W2V_TENSOR

    if "seq_length" in model_kwargs.keys():
        model_kwargs["seq_length"] = data.SEQ_LENGTH

    if "embedding_dim" in model_kwargs.keys():
        model_kwargs["embedding_dim"] = data.EMBEDDING_DIM

    return model_kwargs
    

def get_model_optim_schedule(model_name: str,
                             data: DataManager, 
                             model_kwargs: Dict[str, Any], 
                             optim_kwargs: Dict[str, Any],
                             scheduler_kwargs: Dict[str, Any],
                             device: torch.device) -> Tuple[torch.nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Chuẩn bị mô hình, bộ tối ưu hóa và bộ điều chỉnh tốc độ học
    """
    model_kwargs = update_model_kwargs(data, model_kwargs)
    model = get_model(
        name=model_name,
        **model_kwargs  
    ).to(device)
    
    optimizer = optim.AdamW(
        params=model.parameters(), 
        **optim_kwargs
    ) 
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        **scheduler_kwargs
    )
    
    return (model, optimizer, scheduler)
from typing import Dict
from collections import defaultdict

import torch
import torch.nn as nn
from torch.amp import autocast
from captum.attr import LayerIntegratedGradients

# Wrapper Class
class TAGNetTaskWrapper(nn.Module):
    def __init__(self, 
                 model: torch.nn.Module, 
                 task_idx: int) -> None:
        super().__init__()
        self.model = model
        self.task_idx = task_idx 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs[self.task_idx] 

# Hàm trích xuất Global Integrated Gradients
def extract_global_ig(model: torch.nn.Module, 
                      dataloader: torch.utils.data.DataLoader, 
                      task_idx: int, 
                      id_to_idx: Dict[int, int], 
                      device: torch.device,
                      n_steps: int = 20,
                      max_batches: int = None) -> Dict[int, float]:
    """
    Phân tích tầm quan trọng của từng hành động đối với một Thuộc tính cụ thể bằng Integrated Gradients
    """
    # Khởi tạo Wrapper và Layer Integrated Gradients
    wrapped_model = TAGNetTaskWrapper(model, task_idx).to(device)
    wrapped_model.eval()
    lig = LayerIntegratedGradients(wrapped_model, model.embedding)

    idx_to_id = {idx: original_id for original_id, idx in id_to_idx.items()}
    
    action_impact_sum = defaultdict(float)
    action_count = defaultdict(int)

    for batch_idx, (batch_x, _) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
            
        batch_x = batch_x.to(device)
        
        # Lấy predicted_class làm Target giải thích
        with torch.no_grad(), autocast("cuda"):
            logits = wrapped_model(batch_x)
            predicted_class = torch.argmax(logits, dim=1)

        # Chạy IG với internal_batch_size để chống tràn VRAM GPU
        with autocast("cuda"):
            attributions = lig.attribute(
                inputs=batch_x,
                target=predicted_class,
                n_steps=n_steps,
                internal_batch_size=32 
            )
        
        token_importance = attributions.sum(dim=-1).cpu().detach().numpy()
        batch_actions = batch_x.cpu().numpy()
        
        # Đo cường độ tác động
        for user_actions, user_impacts in zip(batch_actions, token_importance):
            for action_idx, impact in zip(user_actions, user_impacts):
                if action_idx != 0 and action_idx in idx_to_id:
                    original_id = idx_to_id[action_idx]
                    # Lấy trị tuyệt đối vì quan tâm "độ mạnh", không quan trọng âm dương
                    action_impact_sum[original_id] += abs(impact)
                    action_count[original_id] += 1

    # Tính trung bình và sắp xếp
    avg_impact_dict = {}
    for original_id in action_impact_sum:
        avg_impact_dict[original_id] = action_impact_sum[original_id] / action_count[original_id]
        
    sorted_impact = dict(sorted(avg_impact_dict.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_impact
from typing import Dict
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.amp import autocast

def extract_global_attention(model: torch.nn.Module, 
                             dataloader: torch.utils.data.DataLoader, 
                             id_to_idx: Dict[int, int], 
                             device: torch.device) -> Dict[int, float]:
    """
    Trích xuất tổng trọng số Attention cho từng hành động gốc trên toàn bộ DataLoader
    """
    # Dọn dẹp hook cũ còn sót lại trong RAM (nếu có) để tránh xung đột
    model.attn_pool.attention_weights._forward_hooks.clear()
    
    captured_data = {}

    # Định nghĩa hàm Hook
    def hook_fn(module, input, output):
        captured_data["raw_scores"] = output.detach()

    # Gắn Hook mới
    hook_handle = model.attn_pool.attention_weights.register_forward_hook(hook_fn)
    
    idx_to_id = {idx: original_id for original_id, idx in id_to_idx.items()}
    action_attention_sum = defaultdict(float)
    action_count = defaultdict(int)

    model.eval()
    
    try: # Dùng try-finally để đảm bảo chắc chắn sẽ remove hook
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                mask = (batch_x != 0)
                
                # Chạy forward
                with autocast("cuda"):
                    _ = model(batch_x)
                
                # Tính toán attention
                raw_scores = captured_data["raw_scores"] 
                fill_value = torch.finfo(raw_scores.dtype).min
                masked_scores = raw_scores.masked_fill(~mask.unsqueeze(-1), fill_value)
                attn_weights = F.softmax(masked_scores, dim=1).squeeze(-1) 
                
                batch_actions = batch_x.cpu().numpy()
                batch_weights = attn_weights.cpu().numpy()
                
                for user_actions, user_weights in zip(batch_actions, batch_weights):
                    for action_idx, weight in zip(user_actions, user_weights):
                        if action_idx != 0 and action_idx in idx_to_id: 
                            original_id = idx_to_id[action_idx]
                            action_attention_sum[original_id] += weight
                            action_count[original_id] += 1
    
    finally:
        hook_handle.remove()
        
    # Tính trung bình và sắp xếp
    avg_attention_dict = {}
    for original_id in action_attention_sum:
        avg_weight = (action_attention_sum[original_id] / action_count[original_id]) * 100
        avg_attention_dict[original_id] = avg_weight
        
    sorted_attention = dict(sorted(avg_attention_dict.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_attention
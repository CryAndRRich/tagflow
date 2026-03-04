import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling1D(nn.Module):
    def __init__(self, in_features: int) -> None:
        """
        Khởi tạo lớp pooling sử dụng cơ chế attention để tổng hợp thông tin từ các bước thời gian khác nhau trong một chuỗi đầu vào."""
        super(AttentionPooling1D, self).__init__()
        self.attention_weights = nn.Linear(in_features=in_features, 
                                           out_features=1)

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        scores = self.attention_weights(x) 
        if mask is not None:
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask.unsqueeze(-1), fill_value)

        attn_weights = F.softmax(scores, dim=1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)
        return weighted_sum
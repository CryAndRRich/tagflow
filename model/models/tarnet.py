from typing import List
import torch
import torch.nn as nn

from .attention import AttentionPooling1D
from ..models import register_model

@register_model("tarnet")
class TARNet(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 embedding_dim: int = 256, 
                 hidden_dim: int = 256, 
                 num_layers: int = 2, 
                 dropout_rate: float = 0.3) -> None:
        """
        Khởi tạo mô hình TARNet (Temporal Actions Recurrent Network)
        
        Tham số:
            vocab_size: Kích thước từ điển
            num_classes_list: Danh sách số lượng lớp cho mỗi nhiệm vụ phân loại
            w2v_tensor: Tensor chứa vector nhúng từ điển đã được khởi tạo trước (pre-trained)
            embedding_dim: Số chiều của vector nhúng từ điển
            hidden_dim: Số chiều của hidden state
            num_layers: Số lớp RNN
            dropout_rate: Tỷ lệ dropout cho các lớp fully connected
        """
        super(TARNet, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout_rate if num_layers > 1 else 0)
        
        self.proj = nn.Sequential(
            nn.Linear(in_features=hidden_dim * 2, 
                      out_features=embedding_dim),
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.attn_pool = AttentionPooling1D(in_features=embedding_dim)
        
        self.classifier_heads = nn.ModuleList([
            nn.Linear(in_features=embedding_dim, 
                      out_features=num_classes) 
            for num_classes in num_classes_list
        ])

        self.apply(self._init_weights)
        self.embedding.weight.data.copy_(w2v_tensor)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mask = (x != 0) 

        embedded = self.embedding(x) 
        rnn_out, _ = self.rnn(embedded) 
        
        h = self.proj(rnn_out)
        pooled = self.attn_pool(h, mask=mask) 
        
        outputs = [head(pooled) for head in self.classifier_heads]
        return outputs
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List
import torch
import torch.nn as nn

from model.models.attention import AttentionPooling1D
from model.models import register_model

@register_model("taanet")
class FinetuneBehaviorModel(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int],
                 w2v_tensor: torch.Tensor, 
                 embedding_dim: int = 256, 
                 num_heads: int = 4, 
                 num_layers: int = 2, 
                 max_seq_len: int = 500, 
                 pad_token: int = 0) -> None:
        """
        Khởi tạo mô hình TAANet (Temporal Actions Attention Network)
        
        Tham số:
            vocab_size: Kích thước từ điển
            num_classes_list: Danh sách số lượng lớp cho mỗi nhiệm vụ phân loại
            w2v_tensor: Tensor chứa vector nhúng từ điển đã được khởi tạo trước (pre-trained)
            embedding_dim: Số chiều của vector nhúng từ điển
            num_heads: Số lượng head trong multi-head attention
            num_layers: Số lớp Transformer Encoder
            max_seq_len: Độ dài tối đa của chuỗi đầu vào
            pad_token: Giá trị token dùng để padding trong embedding
        """
        super().__init__()
        self.pad_token = pad_token
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=pad_token)
        self.pos_embedding = nn.Embedding(num_embeddings=max_seq_len, 
                                          embedding_dim=embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=embedding_dim * 4, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                                 num_layers=num_layers)
        self.attn_pool = AttentionPooling1D(in_features=embedding_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.3) for _ in range(5)])
        hidden_dim = embedding_dim // 2 
        
        self.classifier_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features=embedding_dim, 
                          out_features=hidden_dim),
                nn.BatchNorm1d(num_features=hidden_dim), 
                nn.GELU(),                 
                nn.Dropout(p=0.1),           
                nn.Linear(in_features=hidden_dim, 
                          out_features=num_classes)
            ) 
            for num_classes in num_classes_list
        ])

        self.apply(self._init_weights)
        self.embedding.weight.data.copy_(w2v_tensor) 

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.1, 0.1)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)
        elif isinstance(m, nn.TransformerEncoderLayer):
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        x_emb = self.embedding(x) + self.pos_embedding(positions)
        padding_mask = (x == self.pad_token)
        trans_out = self.transformer(x_emb, src_key_padding_mask=padding_mask)
        
        pooled_features = self.attn_pool(trans_out, padding_mask)

        outputs = []
        for classifier_head in self.classifier_heads:
            x_drop = torch.mean(
                torch.stack([classifier_head(drop(pooled_features)) for drop in self.dropouts], dim=0), 
                dim=0
            )
            outputs.append(x_drop)
            
        return outputs
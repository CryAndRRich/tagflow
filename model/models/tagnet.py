import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.attention import AttentionPooling1D
from model.models import register_model

class DenseGAT(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 heads: int = 4, 
                 dropout: float = 0.2) -> None:
        """
        Graph Attention Network (GAT) cho đồ thị dày đặc
        
        Tham số:
            in_channels: Số chiều của đặc trưng đầu vào cho mỗi nút
            out_channels: Số chiều của đặc trưng đầu ra cho mỗi nút sau khi qua lớp GAT
            heads: Số lượng attention heads để sử dụng trong lớp GAT
            dropout: Tỷ lệ dropout áp dụng cho các trọng số attention sau khi tính toán
        """
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.head_dim = out_channels // heads
        
        self.lin = nn.Linear(in_features=in_channels, 
                             out_features=out_channels, 
                             bias=False)
        self.att_src = nn.Parameter(data=torch.Tensor(1, heads, 1, self.head_dim))
        self.att_dst = nn.Parameter(data=torch.Tensor(1, heads, 1, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=dropout)
        
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, 
                x: torch.Tensor, 
                adj_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        h = self.lin(x)
        h = h.view(B, L, self.heads, self.head_dim).transpose(1, 2) 
        
        alpha_src = (h * self.att_src).sum(dim=-1, keepdim=True)
        alpha_dst = (h * self.att_dst).sum(dim=-1, keepdim=True)
        e = alpha_src + alpha_dst.transpose(2, 3)
        e = self.leaky_relu(e)
        
        adj_mask = adj_mask.unsqueeze(1) 
        e = e.masked_fill(~adj_mask, -1e9)
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        
        out = torch.matmul(alpha, h) 
        out = out.transpose(1, 2).contiguous().view(B, L, self.out_channels) 
        return out
    

class GatedDirectedGAT(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 heads: int = 4, 
                 dropout: float = 0.2) -> None:
        """
        Lớp GAT có cổng (gated) cho đồ thị có hướng, kết hợp thông tin từ cả hai hướng
        """
        super().__init__()
        self.gat_past = DenseGAT(in_channels=in_channels, 
                                 out_channels=out_channels, 
                                 heads=heads, 
                                 dropout=dropout)
        self.gat_future = DenseGAT(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   heads=heads, 
                                   dropout=dropout)
        
        self.msg_proj = nn.Linear(in_features=out_channels * 2, 
                                  out_features=out_channels)
        
        self.gru_cell = nn.GRUCell(input_size=out_channels, 
                                   hidden_size=out_channels)
        self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, 
                x: torch.Tensor, 
                adj_past: torch.Tensor, 
                adj_future: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        
        m_past = self.gat_past(x, adj_past)
        m_future = self.gat_future(x, adj_future)
        
        m_combined = torch.cat([m_past, m_future], dim=-1)
        m_combined = F.relu(self.msg_proj(m_combined))  
        
        x_flat = x.view(-1, C)
        m_flat = m_combined.view(-1, C)
        
        h_new_flat = self.gru_cell(m_flat, x_flat)
        h_new = h_new_flat.view(B, L, C)
        
        h_new = self.norm(h_new)
        h_new = h_new * mask.unsqueeze(-1).float()
        
        return h_new


@register_model("tagnet")
class TAGNet(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int],
                 w2v_tensor: torch.Tensor, 
                 seq_length: int = 37, 
                 embedding_dim: int = 256, 
                 window: int = 3, 
                 dilations: List[int] = [1, 2, 4], 
                 heads: int = 4, 
                 dropout_rate: float = 0.3) -> None:
        """
        Khởi tạo mô hình TAGNet (Temporal Actions Graph Network)
        
        Tham số:
            vocab_size: Kích thước từ điển
            num_classes_list: Danh sách số lượng lớp cho mỗi nhiệm vụ phân loại
            w2v_tensor: Tensor chứa vector nhúng từ điển đã được khởi tạo trước (pre-trained)
            seq_length: Độ dài tối đa của chuỗi đầu vào
            embedding_dim: Số chiều của vector nhúng từ điển
            window: Kích thước cửa sổ để xác định các nút lân cận trong đồ thị
            dilations: Danh sách các độ giãn cách để áp dụng trong các lớp GAT
            heads: Số lượng attention heads trong mỗi lớp GAT
            dropout_rate: Tỷ lệ dropout để áp dụng trong mô hình
        """
        super(TAGNet, self).__init__()
        
        self.window = window 
        self.dilations = dilations 
        self.num_layers = len(dilations)
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        self.pos_embedding = nn.Embedding(num_embeddings=seq_length, 
                                          embedding_dim=embedding_dim)
        
        self.layers = nn.ModuleList([
            GatedDirectedGAT(in_channels=embedding_dim, 
                             out_channels=embedding_dim, 
                             heads=heads, 
                             dropout=dropout_rate)
            for _ in range(self.num_layers)
        ])
        
        self.attn_pool = AttentionPooling1D(in_features=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        
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
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.1)
            if m.padding_idx is not None:
                nn.init.constant_(m.weight[m.padding_idx], 0)

    def _get_directed_adj_masks(self, 
                                logical_pos: torch.Tensor, 
                                valid_nodes: torch.Tensor, 
                                window: int, 
                                dilation: int, 
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tạo các mặt nạ adjacency cho đồ thị có hướng dựa trên vị trí logic của các nút 
        và các tham số cửa sổ và độ giãn cách.
        
        Tham số:
            logical_pos: Tensor chứa vị trí logic của các nút trong chuỗi
            valid_nodes: Tensor boolean chỉ ra các nút hợp lệ (không phải padding)
            window: Kích thước cửa sổ để xác định các nút lân cận
            dilation: Độ giãn cách để áp dụng khi xác định các nút lân cận
            device: Thiết bị (CPU hoặc GPU) để tạo các tensor mặt nạ
        
        Trả về:
            Tuple[torch.Tensor, torch.Tensor]: Hai tensor boolean tương ứng với mặt nạ adjacency cho các cạnh 
                                               hướng về quá khứ và tương lai
        """
        diff_mat = logical_pos.unsqueeze(2) - logical_pos.unsqueeze(1)
        abs_diff = torch.abs(diff_mat)
        
        in_window = (abs_diff <= window * dilation) & (abs_diff % dilation == 0)
        
        past_window = in_window & (diff_mat < 0)
        future_window = in_window & (diff_mat > 0)
        
        adj_past = valid_nodes & past_window
        adj_future = valid_nodes & future_window
        
        eye = torch.eye(n=logical_pos.size(1), 
                        dtype=torch.bool, 
                        device=device).unsqueeze(0)
        adj_past = adj_past | eye 
        adj_future = adj_future | eye 
        
        return (adj_past, adj_future)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len = x.size()
        mask = (x != 0) 

        positions = torch.arange(start=0, 
                                 end=seq_len, 
                                 device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions) * mask.unsqueeze(-1).float() 
        
        embedded = self.embedding(x) + pos_emb
        embedded = embedded * mask.unsqueeze(-1).float()
        
        logical_pos = (torch.cumsum(mask.long(), dim=1) - 1).clamp(min=0)
        valid_nodes = mask.unsqueeze(1) & mask.unsqueeze(2) 
        
        h = embedded
        for i, layer in enumerate(self.layers):
            current_dilation = self.dilations[i]
            adj_past, adj_future = self._get_directed_adj_masks(
                logical_pos=logical_pos, 
                valid_nodes=valid_nodes, 
                window=self.window, 
                dilation=current_dilation, 
                device=x.device
            )
            h = layer(h, adj_past, adj_future, mask)
        
        pooled_features = self.attn_pool(h, mask=mask)
        x_drop = self.dropout(pooled_features)
        
        outputs = [head(x_drop) for head in self.classifier_heads]
        return outputs
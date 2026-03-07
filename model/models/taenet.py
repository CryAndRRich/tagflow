import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from typing import List, Tuple
import torch
import torch.nn as nn

from model.models.attention import AttentionPooling1D
from model.models.tacnet import ModernTCNBlock
from model.models.tagnet import GatedDirectedGAT
from model.models import register_model

@register_model("taenet")
class TAENet(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 num_classes_list: List[int], 
                 w2v_tensor: torch.Tensor, 
                 seq_length: int = 37, 
                 embedding_dim: int = 256, 
                 rnn_hidden_dim: int = 256, 
                 rnn_num_layers: int = 2, 
                 cnn_kernel_sizes: List[int] = [3, 5, 7], 
                 cnn_expansion_factor: int = 2, 
                 gnn_window: int = 3, 
                 gnn_dilations: List[int] = [1, 2, 4], 
                 gnn_heads: int = 4, 
                 dropout_rate: float = 0.3) -> None:
        """
        Khởi tạo mô hình TAENet (Temporal Actions Ensemble Network)
        Kết hợp 3 backbone: RNN, CNN và GNN
        """
        super(TAENet, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        self.pos_embedding = nn.Embedding(num_embeddings=seq_length, 
                                          embedding_dim=embedding_dim)
        
        # Nhánh RNN (TARNet)
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=rnn_hidden_dim,
                          num_layers=rnn_num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout_rate if rnn_num_layers > 1 else 0)
        self.rnn_proj = nn.Sequential(
            nn.Linear(in_features=rnn_hidden_dim * 2, out_features=embedding_dim),
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.attn_pool_rnn = AttentionPooling1D(in_features=embedding_dim)
        
        # Nhánh CNN (TACNet)
        cnn_blocks = []
        for ks in cnn_kernel_sizes:
            cnn_blocks.append(
                ModernTCNBlock(d_model=embedding_dim, 
                               kernel_size=ks, 
                               expansion_factor=cnn_expansion_factor, 
                               dropout=dropout_rate)
            )
        self.cnn_backbone = nn.Sequential(*cnn_blocks)
        self.attn_pool_cnn = AttentionPooling1D(in_features=embedding_dim)
        
        # Nhánh GNN (TAGNet)
        self.gnn_window = gnn_window
        self.gnn_dilations = gnn_dilations
        self.gnn_layers = nn.ModuleList([
            GatedDirectedGAT(in_channels=embedding_dim, 
                             out_channels=embedding_dim, 
                             heads=gnn_heads, 
                             dropout=dropout_rate)
            for _ in range(len(gnn_dilations))
        ])
        self.attn_pool_gnn = AttentionPooling1D(in_features=embedding_dim)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Input của Classifier là tổng hợp từ 3 backbone
        concat_dim = embedding_dim * 3
        self.classifier_heads = nn.ModuleList([
            nn.Linear(in_features=concat_dim, out_features=num_classes) 
            for num_classes in num_classes_list
        ])

        # Khởi tạo trọng số
        self.apply(self._init_weights)
        self.embedding.weight.data.copy_(w2v_tensor)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
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
        diff_mat = logical_pos.unsqueeze(2) - logical_pos.unsqueeze(1)
        abs_diff = torch.abs(diff_mat)
        in_window = (abs_diff <= window * dilation) & (abs_diff % dilation == 0)
        past_window = in_window & (diff_mat < 0)
        future_window = in_window & (diff_mat > 0)
        
        adj_past = valid_nodes & past_window
        adj_future = valid_nodes & future_window
        
        eye = torch.eye(n=logical_pos.size(1), dtype=torch.bool, device=device).unsqueeze(0)
        return (adj_past | eye, adj_future | eye)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len = x.size()
        mask = (x != 0) 
        
        embedded = self.embedding(x)
        embedded = embedded * mask.unsqueeze(-1).float()
        
        # Positional Embedding (Dùng cho CNN và GNN)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pos_embedding(positions) * mask.unsqueeze(-1).float()
        embedded_with_pos = embedded + pos_emb
        
        # Nhánh RNN 
        rnn_out, _ = self.rnn(embedded) 
        h_rnn = self.rnn_proj(rnn_out)
        pooled_rnn = self.attn_pool_rnn(h_rnn, mask=mask)
        
        # Nhánh CNN 
        cnn_in = embedded_with_pos.permute(0, 2, 1) 
        h_cnn = self.cnn_backbone(cnn_in).permute(0, 2, 1) 
        pooled_cnn = self.attn_pool_cnn(h_cnn, mask=mask)
        
        # Nhánh GNN 
        logical_pos = (torch.cumsum(mask.long(), dim=1) - 1).clamp(min=0)
        valid_nodes = mask.unsqueeze(1) & mask.unsqueeze(2) 
        
        h_gnn = embedded_with_pos
        for i, layer in enumerate(self.gnn_layers):
            current_dilation = self.gnn_dilations[i]
            adj_past, adj_future = self._get_directed_adj_masks(
                logical_pos, valid_nodes, self.gnn_window, current_dilation, x.device
            )
            h_gnn = layer(h_gnn, adj_past, adj_future, mask)
            
        pooled_gnn = self.attn_pool_gnn(h_gnn, mask=mask)
        
        # Gộp 3 vector lại: (Batch, 256) x 3 -> (Batch, 768)
        combined_features = torch.cat([pooled_rnn, pooled_cnn, pooled_gnn], dim=-1)
        x_drop = self.dropout(combined_features)
        
        outputs = [head(x_drop) for head in self.classifier_heads]
        return outputs
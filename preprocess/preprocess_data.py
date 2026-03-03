from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
from gensim.models import Word2Vec

def format_dtypes(df: pd.DataFrame, 
                  cols: List[str]) -> pd.DataFrame:
    """Điền 0 và ép kiểu int64 cho các cột chỉ định"""
    df_clean = df.copy()
    df_clean[cols] = df_clean[cols].fillna(0).astype(np.int64)
    return df_clean


def drop_duplicates_and_leaks(x_train: pd.DataFrame, 
                              y_train: pd.DataFrame, 
                              x_val: pd.DataFrame, 
                              y_val: pd.DataFrame, 
                              feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loại bỏ trùng lặp và data leakage giữa Train và Val"""
    # Drop duplicates trong tập train
    train_keep_idx = x_train.drop_duplicates(subset=feature_cols, keep="first").index
    x_train = x_train.loc[train_keep_idx].reset_index(drop=True)
    y_train = y_train.loc[train_keep_idx].reset_index(drop=True)

    # Drop duplicates trong tập val
    val_keep_idx = x_val.drop_duplicates(subset=feature_cols, keep="first").index
    x_val = x_val.loc[val_keep_idx].reset_index(drop=True)
    y_val = y_val.loc[val_keep_idx].reset_index(drop=True)

    # Chống rò rỉ
    train_features_unique = x_train[feature_cols].copy()
    train_features_unique["is_in_train"] = True 
    x_val["old_idx"] = x_val.index
    val_merged = x_val.merge(train_features_unique, on=feature_cols, how="left")
    clean_val_idx = val_merged[val_merged["is_in_train"].isna()]["old_idx"]
    
    x_val = x_val.loc[clean_val_idx].drop(columns=["old_idx"]).reset_index(drop=True)
    y_val = y_val.loc[clean_val_idx].reset_index(drop=True)
    
    return x_train, y_train, x_val, y_val


def build_vocab_mapping(dfs: List[pd.DataFrame], 
                        feature_cols: List[str]) -> Tuple[Dict[int, int], int]:
    """Chỉ quét qua dữ liệu để tạo bộ từ điển id_to_idx"""
    x_data = pd.concat([df[feature_cols] for df in dfs], axis=0)
    unique_ids = pd.unique(x_data.values.ravel())
    unique_ids = unique_ids[~pd.isna(unique_ids)] 
    unique_ids = [uid for uid in unique_ids if uid != 0]
    
    id_to_idx = {id: idx for idx, id in enumerate(sorted(unique_ids), start=1)}
    id_to_idx[0] = 0 
    vocab_size = max(id_to_idx.values()) + 1
    return id_to_idx, vocab_size


def apply_vocab_mapping(df: pd.DataFrame, 
                        id_to_idx: Dict[int, int], 
                        feature_cols: List[str]) -> pd.DataFrame:
    """Áp dụng bộ từ điển vào DataFrame"""
    df_mapped = df.copy()
    for col in feature_cols:
        df_mapped[col] = df_mapped[col].map(id_to_idx).fillna(0).astype(np.int64)
    return df_mapped


def train_w2v_model(x_train_mapped: pd.DataFrame, 
                    feature_cols: List[str], 
                    vocab_size: int, 
                    embedding_dim: int, 
                    random_seed: int) -> torch.FloatTensor:
    """Huấn luyện Word2Vec dựa trên tập Train đã map ID"""
    sentences = []
    vals = x_train_mapped[feature_cols].values
    for row in vals:
        seq = [str(val) for val in row if val != 0]
        if len(seq) > 0: 
            sentences.append(seq)

    w2v_model = Word2Vec(sentences=sentences, 
                         vector_size=embedding_dim, 
                         window=5, 
                         min_count=1, 
                         workers=4, 
                         seed=random_seed)
    
    pretrained_weights = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    for mapped_id in range(1, vocab_size):
        str_id = str(mapped_id)
        if str_id in w2v_model.wv:
            pretrained_weights[mapped_id] = w2v_model.wv[str_id]
        else:
            pretrained_weights[mapped_id] = np.random.normal(0, 0.1, embedding_dim)
            
    return torch.FloatTensor(pretrained_weights)
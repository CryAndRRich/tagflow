from typing import List, Tuple, Union, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class UserBehaviorDataset(Dataset):
    def __init__(self, 
                 x_df: pd.DataFrame, 
                 y_df: Optional[pd.DataFrame],
                 augment: bool = False) -> None:
        """
        Dataset tùy chỉnh cho dữ liệu hành vi người dùng, hỗ trợ tăng cường dữ liệu
        
        Tham số:
            x_df: DataFrame chứa đặc trưng đầu vào
            y_df: DataFrame chứa nhãn, có thể là None nếu không có nhãn
            attr_cols: Danh sách tên cột thuộc tính trong y_df
            augment: Nếu True, sẽ áp dụng tăng cường dữ liệu ngẫu nhiên khi truy cập mẫu
        """
        self.x_data = x_df.drop(columns=["id"]).values
        self.augment = augment 
        
        self.has_labels = y_df is not None
        if self.has_labels:
            self.y_data = y_df.drop(columns=["id"]).values

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_tensor = torch.tensor(self.x_data[idx], dtype=torch.long)
        
        # Tăng cường dữ liệu ngẫu nhiên:
        if self.augment:
            # Với xác suất 50%, thực hiện phép giãn nở ngẫu nhiên trên chuỗi đặc trưng
            if torch.rand(1).item() < 0.5:
                actions = x_tensor[x_tensor != 0]
                n_actions = len(actions)
                total_slots = len(x_tensor) 
                
                if 0 < n_actions < total_slots:
                    new_indices = torch.randperm(total_slots)[:n_actions].sort()[0]
                    
                    dilated_x = torch.zeros_like(x_tensor)
                    dilated_x[new_indices] = actions
                    
                    x_tensor = dilated_x
            
            # Thực hiện masking 10% chuỗi hành động
            mask_prob = torch.rand(x_tensor.shape)
            mask_drop = (mask_prob < 0.1) & (x_tensor != 0) 
            x_tensor.masked_fill_(mask_drop, 0)
        
        if self.has_labels:
            y_tensor = torch.tensor(self.y_data[idx], dtype=torch.long)
            return (x_tensor, y_tensor)
        
        return x_tensor


def create_dataloaders(x_train: pd.DataFrame, 
                       y_train: pd.DataFrame, 
                       x_val: pd.DataFrame, 
                       y_val: pd.DataFrame, 
                       x_test: pd.DataFrame, 
                       feature_cols: List[str],
                       short_seq_length: int,
                       duplicate_factor: int,
                       batch_size: int,
                       seed_worker,
                       data_generator: torch.Generator) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho tập train, test và val, với tăng cường dữ liệu cho các chuỗi ngắn
    
    Tham số:
        x_train, y_train, x_val, y_val, x_test: DataFrame chứa dữ liệu đặc trưng và nhãn
        feature_cols: Danh sách tên cột đặc trưng
        short_seq_length: Độ dài tối đa để xác định chuỗi ngắn
        duplicate_factor: Hệ số nhân để tăng cường dữ liệu cho chuỗi ngắn
        batch_size: Kích thước batch cho DataLoader
        seed_worker: Hàm để cài đặt seed cho mỗi worker trong DataLoader
        data_generator: Generator để đảm bảo tính tái lập khi tạo DataLoader
    
    Trả về:
        Tuple[DataLoader, DataLoader, DataLoader]: Tuple chứa DataLoader cho tập train, val và test
    """
    # Xác định chuỗi ngắn và nhân bản chúng trong tập train
    seq_lengths = (x_train[feature_cols] != 0).sum(axis=1)
    short_seq_mask = seq_lengths <= short_seq_length
    x_short = x_train[short_seq_mask]
    y_short = y_train[short_seq_mask]

    # Chỉ nhân bản nếu có chuỗi ngắn và số lượng chuỗi ngắn nhỏ hơn 1000 để tránh làm mất cân bằng dữ liệu
    if len(x_short) > 0 and len(x_short) < 1000:
        x_train = pd.concat([x_train] + [x_short] * duplicate_factor, ignore_index=True)
        y_train = pd.concat([y_train] + [y_short] * duplicate_factor, ignore_index=True)

    train_dataset = UserBehaviorDataset(x_train, y_train, augment=True)
    val_dataset = UserBehaviorDataset(x_val, y_val)
    test_dataset = UserBehaviorDataset(x_test, None)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        worker_init_fn=seed_worker, 
        generator=data_generator
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return (train_loader, val_loader, test_loader)
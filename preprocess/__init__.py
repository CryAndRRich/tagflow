import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocess.preprocess_data import format_dtypes, drop_duplicates_and_leaks, build_vocab_mapping, apply_vocab_mapping, train_w2v_model
from preprocess.dataloader import create_dataloaders, create_masked_dataloader
from config import CONFIG_DATA

__all__ = ["DataManager"]

class DataManager:
    def __init__(self, 
                 input_root: str, 
                 work_dir: str, 
                 config_data: CONFIG_DATA,
                 seed_worker,
                 data_generator: torch.Generator,
                 random_seed: int) -> None:
        
        self.INPUT_ROOT = input_root
        self.WORK_DIR = work_dir
        self.SEED_WORKER = seed_worker
        self.DATA_GENERATOR = data_generator
        self.RANDOM_SEED = random_seed

        # Cấu hình hằng số
        self.SEQ_LENGTH = config_data.SEQ_LENGTH
        self.FEATURE_COLS = config_data.FEATURE_COLS
        self.ATTRIBUTE_LIST = config_data.ATTRIBUTE_LIST
        self.ATTRIBUTE_COLS = config_data.ATTRIBUTE_COLS
        self.EMBEDDING_DIM = config_data.EMBEDDING_DIM
        self.BATCH_SIZE = config_data.BATCH_SIZE
        self.SHORT_SEQ_LENGTH = config_data.SHORT_SEQ_LENGTH
        self.DUPLICATE_FACTOR = config_data.DUPLICATE_FACTOR

        # Biến trạng thái quan trọng
        self.id_to_idx = {}
        self.VOCAB_SIZE = 0
        self.MASK_TOKEN = 0
        self.NUM_CLASSES_LIST = []
        self.W2V_TENSOR = None

        # Khởi chạy Pipeline
        self._load_raw_csv()
        self._setup_initial_pipeline()
        self.__create_dataloader(seed_worker=self.SEED_WORKER, data_generator=self.DATA_GENERATOR)

    def _load_raw_csv(self) -> None:
        """Tải dữ liệu thô ban đầu"""
        self.x_train = pd.read_csv(f"{self.INPUT_ROOT}/X_train.csv")
        self.y_train = pd.read_csv(f"{self.INPUT_ROOT}/Y_train.csv")
        self.x_val = pd.read_csv(f"{self.INPUT_ROOT}/X_val.csv")
        self.y_val = pd.read_csv(f"{self.INPUT_ROOT}/Y_val.csv")
        self.x_test = pd.read_csv(f"{self.INPUT_ROOT}/X_test.csv")

    def _setup_initial_pipeline(self) -> None:
        """Pipeline tiền xử lý gốc rễ"""
        # Định dạng kiểu dữ liệu
        for df in ["x_train", "x_val", "x_test"]:
            setattr(self, df, format_dtypes(getattr(self, df), self.FEATURE_COLS))
        for df in ["y_train", "y_val"]:
            setattr(self, df, format_dtypes(getattr(self, df), self.ATTRIBUTE_COLS))

        # Loại bỏ trùng lặp và rò rỉ
        self.x_train, self.y_train, self.x_val, self.y_val = drop_duplicates_and_leaks(
            self.x_train, self.y_train, self.x_val, self.y_val, self.FEATURE_COLS
        )

        # Tạo từ điển và ánh xạ
        self.id_to_idx, self.VOCAB_SIZE = build_vocab_mapping(
            [self.x_train, self.x_val, self.x_test], self.FEATURE_COLS
        )
        for df in ["x_train", "x_val", "x_test"]:
            setattr(self, df, apply_vocab_mapping(getattr(self, df), self.id_to_idx, self.FEATURE_COLS))

        self.MASK_TOKEN = self.VOCAB_SIZE 
        self.VOCAB_SIZE += 1

        # Tính toán tham số Model
        self._update_num_classes()
        self.W2V_TENSOR = train_w2v_model(
            self.x_train, self.FEATURE_COLS, self.VOCAB_SIZE, self.EMBEDDING_DIM, self.RANDOM_SEED
        )

    def _update_num_classes(self) -> None:
        """Cập nhật lại số lượng classes"""
        y_combined = pd.concat([self.y_train, self.y_val], ignore_index=True)
        self.NUM_CLASSES_LIST = [(int(y_combined[col].max()) + 1) for col in self.ATTRIBUTE_COLS]

    def __create_dataloader(self,
                            seed_worker,
                            data_generator: torch.Generator) -> None:
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.x_train, 
            self.y_train, 
            self.x_val, 
            self.y_val, 
            self.x_test, 
            self.FEATURE_COLS, 
            self.SHORT_SEQ_LENGTH, 
            self.DUPLICATE_FACTOR, 
            self.BATCH_SIZE, 
            seed_worker=seed_worker, 
            data_generator=data_generator
        )

    def add_data(self, 
                 extra_x: pd.DataFrame, 
                 extra_y: pd.DataFrame, 
                 is_raw_x: bool = False, 
                 retrain_w2v: bool = False) -> None:
        """
        Thêm dữ liệu mới vào tập huấn luyện
        
        Tham số:
            extra_x: DataFrame chứa dữ liệu X mới
            extra_y: DataFrame chứa dữ liệu Y tương ứng
            is_raw_x: Nếu extra_x là dữ liệu thô chưa qua map vocab (True)
            retrain_w2v: Cờ quyết định có cần huấn luyện lại W2V hay không.
        """
        # Clean và ép kiểu Y mới
        extra_y_clean = format_dtypes(extra_y, self.ATTRIBUTE_COLS)

        # Map X mới nếu nó là dữ liệu thô
        if is_raw_x:
            extra_x_clean = format_dtypes(extra_x, self.FEATURE_COLS)
            extra_x_ready = apply_vocab_mapping(extra_x_clean, self.id_to_idx, self.FEATURE_COLS)
        else:
            extra_x_ready = extra_x.copy()

        # Gộp vào Train
        self.x_train = pd.concat([self.x_train, extra_x_ready], ignore_index=True)
        self.y_train = pd.concat([self.y_train, extra_y_clean], ignore_index=True)

        # Cập nhật các biến quan trọng
        self._update_num_classes()

        if retrain_w2v:
            self.W2V_TENSOR = train_w2v_model(
                self.x_train, self.FEATURE_COLS, self.VOCAB_SIZE, self.EMBEDDING_DIM, self.RANDOM_SEED
            )

        # Tạo lại DataLoader với dữ liệu mới
        self.__create_dataloader(seed_worker=self.SEED_WORKER, data_generator=self.DATA_GENERATOR)

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Trả về dữ liệu đã được xử lý"""
        return (self.x_train, self.y_train, self.x_val, self.y_val, self.x_test)
    
    def get_dataloaders(self, masked: bool = False) -> Union[Tuple[DataLoader, DataLoader, DataLoader], DataLoader]:
        """Trả về DataLoader cho tập train, test và val"""
        if masked:
            # Gộp x_train và x_val để mô hình học nhiều pattern nhất có thể
            x_combined = pd.concat([self.x_train, self.x_val], ignore_index=True)
            
            masked_loader = create_masked_dataloader(
                x_df=x_combined,
                vocab_size=self.VOCAB_SIZE,
                mask_token=self.MASK_TOKEN,
                batch_size=self.BATCH_SIZE,
                seed_worker=self.SEED_WORKER,
                data_generator=self.DATA_GENERATOR
            )
            return masked_loader
        
        return (self.train_loader, self.val_loader, self.test_loader)
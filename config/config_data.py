from typing import List

class CONFIG_DATA:
    SEQ_LENGTH : int = 37
    FEATURE_COLS: List[str] = [f"feature_{i}" for i in range(1, SEQ_LENGTH + 1)]

    ATTRIBUTE_LIST: List[int] = [1, 2, 3, 4, 5, 6]
    ATTRIBUTE_COLS: List[str] = [f"attr_{i}" for i in ATTRIBUTE_LIST]

    EMBEDDING_DIM: int = 256
    BATCH_SIZE: int = 256
    SHORT_SEQ_LENGTH: int = 5
    DUPLICATE_FACTOR: int = 40
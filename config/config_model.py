from typing import Dict, Any
import torch

class CONFIG_MODEL:
    RANDOM_SEED: int = 42
    DEVICE: torch.device = torch.device("cuda")

    NUM_EPOCHS_STAGE_1: int = 50
    NUM_EPOCHS_STAGE_2: int = 150

    EARLY_STOPPING_STAGE_1: int = 8
    EARLY_STOPPING_STAGE_2: int = 15

    LABEL_SMOOTHING: float = 0.01
    ALPHA_MAX_LOSS: float = 2.0

    MODEL_KWARGS: Dict[str, Dict[str, Any]] = {
        "tagnet": {
            "vocab_size": None, 
            "num_classes_list": None,
            "w2v_tensor": None, 
            "seq_length": None, 
            "embedding_dim": None, 
            "window": 3, 
            "dilations": [1, 2, 4], 
            "heads": 4, 
            "dropout_rate": 0.3
        },
        "tacnet": {
            "vocab_size": None, 
            "num_classes_list": None,
            "w2v_tensor": None, 
            "seq_length": None, 
            "embedding_dim": None, 
            "kernel_sizes": [3, 5, 7], 
            "expansion_factor": 2, 
            "dropout_rate": 0.3
        },
        "tarnet": {
            "vocab_size": None, 
            "num_classes_list": None,
            "w2v_tensor": None, 
            "embedding_dim": None, 
            "hidden_dim": 256, 
            "num_layers": 2, 
            "dropout_rate": 0.3
        },
        "taanet": {
            "vocab_size": None, 
            "num_classes_list": None,
            "embedding_dim": None, 
            "num_heads": 4, 
            "num_layers": 2,
            "max_seq_len": 500,
            "pad_token": 0
        },
        "baseline_rnn": {
            "vocab_size": None, 
            "num_classes_list": None,
            "w2v_tensor": None, 
            "embedding_dim": None, 
            "hidden_dim": 256, 
            "num_layers": 2, 
            "dropout_rate": 0.3
        },
        "baseline_lstm": {
            "vocab_size": None, 
            "num_classes_list": None,
            "w2v_tensor": None, 
            "embedding_dim": None, 
            "hidden_dim": 256, 
            "num_layers": 2, 
            "dropout_rate": 0.3
        },
        "baseline_gru": {
            "vocab_size": None, 
            "num_classes_list": None,
            "w2v_tensor": None, 
            "embedding_dim": None, 
            "hidden_dim": 256, 
            "num_layers": 2, 
            "dropout_rate": 0.3
        }
    }
    
    OTIMIZER_KWARGS: Dict[str, Any] = {
        "lr": 1e-3,
        "weight_decay": 1e-2
    }

    SCHEDULER_KWARGS: Dict[str, Any] = {
        "max_lr": 1e-3,
        "pct_start": 0.1,
        "anneal_strategy": "cos",
        "div_factor": 25.0,
        "final_div_factor": 1000.0,
    }
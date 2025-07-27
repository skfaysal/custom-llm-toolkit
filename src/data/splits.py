from typing import List, Any, Tuple
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from configs.config import TRAIN_SPLIT, TEST_SPLIT


def list_data_splitter(data: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    train_portion = int(len(data) * TRAIN_SPLIT)  # 85% for training
    test_portion = int(len(data) * TEST_SPLIT)    # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    return train_data, test_data, val_data
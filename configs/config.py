import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"
VISUALIZATION_DIR = PROJECT_ROOT / "src/visualizations"
TOKENIZER = "gpt2"
TRAIN_SPLIT = 0.85
TEST_SPLIT = 0.10
NUM_WORKER = 0
BATCH_SIZE = 8
EPOCHS = 6
CHOOSE_MODEL = "gpt2-medium (355M)"


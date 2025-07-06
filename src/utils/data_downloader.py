import os
import sys
import urllib.request
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from configs.config import DATA_DIR

## get project root directory
file_path = DATA_DIR / "the-verdict.txt"

if not os.path.exists(file_path):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    
    urllib.request.urlretrieve(url, file_path)
import os
import sys
import json
import urllib
import urllib.request
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from configs.config import DATA_DIR


def download_the_verdict():
       ## get project root directory
       file_path = DATA_DIR / "the-verdict.txt"

       if not os.path.exists(file_path):
              url = ("https://raw.githubusercontent.com/rasbt/"
                     "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
                     "the-verdict.txt")

       urllib.request.urlretrieve(url, file_path)



def download_instruction_data_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


if __name__ == "__main__":

       file_path = DATA_DIR / "instruction-data.json"
       url = (
       "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
       "/main/ch07/01_main-chapter-code/instruction-data.json"
       )

       data = download_instruction_data_and_load_file(file_path, url)
       print("Number of entries:", len(data))
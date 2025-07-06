from importlib.metadata import version
import tiktoken
import sys
from pathlib import Path
import re

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.dataset import SimpleTokenizerV1, SimpleTokenizerV2
from configs.config import DATA_DIR



if __name__ == "__main__":

    #============ Load Dataset = The Verdict =============#
    file_path = DATA_DIR / "the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

    # ============ Example Preprocessing ============ #
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])

    # ============ Create Vocabulary ============ #

    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)

    vocab = {token:integer for integer,token in enumerate(all_words)}

    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 5:
            break

    # ============ Tokenizer V1 without handling unknown words ============ #
    tokenizer_v1 = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer_v1.encode(text)
    print(ids)

    

import tiktoken

from configs.config import TOKENIZER

tokenizer = tiktoken.get_encoding(TOKENIZER)

# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
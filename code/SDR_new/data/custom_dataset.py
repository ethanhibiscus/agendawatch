import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import nltk

nltk.download("punkt")

class CustomTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_dir, block_size, mode="train"):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_dir = data_dir
        self.mode = mode
        self.examples = []
        self.load_data()

    def load_data(self):
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        for file in files:
            with open(file, 'r') as f:
                text = f.read()
                sentences = nltk.sent_tokenize(text)
                for sent in sentences:
                    tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))
                    tokenized = tokenized[:self.block_size]
                    self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

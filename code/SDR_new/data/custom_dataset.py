import os
import nltk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
from tqdm import tqdm

nltk.download("punkt")

class CustomTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_dir: str, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.examples = []
        self.labels = []

        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

        for file in tqdm(files, desc="Processing files"):
            with open(file, 'r') as f:
                text = f.read()
            paragraphs = text.split('\n\n')
            for paragraph in paragraphs:
                sentences = nltk.sent_tokenize(paragraph)
                tokenized_sentences = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))[:block_size] for sent in sentences]
                for tokenized_sentence in tokenized_sentences:
                    self.examples.append(tokenized_sentence)
                    self.labels.append(os.path.basename(file))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(self.examples[idx]), dtype=torch.long)[:self.block_size],
            self.labels[idx]
        )

# Example usage
# tokenizer = PreTrainedTokenizer.from_pretrained('bert-base-uncased')
# dataset = CustomTextDataset(tokenizer, data_dir='./data/text_files', bloc

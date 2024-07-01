import os
import torch
from torch.utils.data import Dataset  # Import the Dataset class
from transformers import PreTrainedTokenizer
from nltk.tokenize import sent_tokenize
class CustomTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, block_size, mode="train"):
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data_dir = hparams.data_dir  # Use data_dir from hparams
        self.mode = mode
        
        self.examples = []
        self.indices_map = []

        self.load_data()

    def load_data(self):
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    sentences = sent_tokenize(text)
                    self.process_sentences(sentences, file)

    def process_sentences(self, sentences, title):
        for idx, sentence in enumerate(sentences):
            tokenized_text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sentence))[:self.block_size]
            self.examples.append((tokenized_text, len(tokenized_text), title, idx, idx, idx, idx, idx, idx))
            self.indices_map.append((len(self.examples) - 1, len(tokenized_text), title, idx, idx, idx, idx, idx, idx))

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, item):
        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(self.examples[item][0]), dtype=torch.long),
            self.examples[item][2],  # title
            self.examples[item][1],  # token length
            self.examples[item][3],  # idx_article
            self.examples[item][4],  # idx_section
            self.examples[item][5],  # idx_sentence
            self.examples[item][6],  # additional index 1
            self.examples[item][7],  # additional index 2
            self.examples[item][8],  # label
        )

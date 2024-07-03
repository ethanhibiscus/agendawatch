import ast
from data.data_utils import get_gt_seeds_titles, raw_data_link
import nltk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import json
import csv
import sys
from models.reco.recos_utils import index_amp


nltk.download("punkt")


class WikipediaTextDatasetParagraphsSentences(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="train"):
        self.hparams = hparams
        self.block_size = min(block_size, tokenizer.model_max_length)
        self.tokenizer = tokenizer

        cached_features_file = os.path.join(
            f"data/datasets/cached_proccessed/{dataset_name}",
            f"bs_{block_size}_{dataset_name}_{type(self).__name__}_tokenizer_{str(type(tokenizer)).split('.')[-1][:-2]}_mode_{mode}",
        )
        self.cached_features_file = cached_features_file
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)

        raw_data_path = f"./data/text_files"  # Adjusted path to your text files

        if os.path.exists(cached_features_file) and (self.hparams is None or not self.hparams.overwrite_data_cache):
            print("\nLoading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples, self.indices_map = pickle.load(handle)
        else:
            print("\nCreating features from dataset file at ", cached_features_file)

            self.examples = []
            self.indices_map = []
            all_files = os.listdir(raw_data_path)
            for idx_file, file_name in enumerate(tqdm(all_files)):
                with open(os.path.join(raw_data_path, file_name), 'r') as f:
                    text = f.read()
                paragraphs = text.split('\n\n')
                valid_paragraphs = [p for p in paragraphs if len(p.strip()) > 0]
                for paragraph in valid_paragraphs:
                    sentences = nltk.sent_tokenize(paragraph)
                    tokenized_sentences = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent))[:self.block_size] for sent in sentences]
                    self.examples.append((tokenized_sentences, file_name))
                    for idx_sent, sent in enumerate(tokenized_sentences):
                        self.indices_map.append((idx_file, idx_sent))

            print("\nSaving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.examples, self.indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.labels = [idx_file for idx_file, _ in self.indices_map]

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, item):
        idx_file, idx_sentence = self.indices_map[item]
        sentence = self.examples[idx_file][0][idx_sentence]

        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sentence), dtype=torch.long)[:self.hparams.limit_tokens],
            self.examples[idx_file][1],
            sentence,
            idx_file,
            idx_sentence,
            item,
            self.labels[item],
        )

class WikipediaTextDatasetParagraphsSentencesTest(WikipediaTextDatasetParagraphsSentences):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="test"):
        super().__init__(tokenizer, hparams, dataset_name, block_size, mode=mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sentences = self.examples[item][0]
        processed_sentences = []
        for idx_sentence, sentence in enumerate(sentences):
            processed_sentences.append(
                (
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sentence), dtype=torch.long),
                    self.examples[item][1],
                    sentence,
                    item,
                    idx_sentence,
                    item,
                    self.labels[item],
                )
            )
        return processed_sentences
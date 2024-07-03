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
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.mode = mode

        cached_features_file = os.path.join(
            f"data/datasets/cached_proccessed/{dataset_name}",
            f"bs_{block_size}_{dataset_name}_{type(self).__name__}_tokenizer_{str(type(tokenizer)).split('.')[-1][:-2]}_mode_{mode}",
        )
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)

        raw_data_path = f"data/text_files"

        self.examples, self.indices_map = self.load_and_cache_examples(cached_features_file, raw_data_path)

        self.labels = [idx_article for idx_article, _, _ in self.indices_map]

    def load_and_cache_examples(self, cached_features_file, raw_data_path):
        if os.path.exists(cached_features_file) and not self.hparams.overwrite_data_cache:
            print("\nLoading features from cached file", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples, indices_map = pickle.load(handle)
        else:
            print("\nCreating features from dataset file at", cached_features_file)

            examples = []
            indices_map = []
            max_article_len = int(1e6)
            max_sentences = 16
            max_sent_len = 10000

            for idx_article, filename in enumerate(tqdm(os.listdir(raw_data_path))):
                filepath = os.path.join(raw_data_path, filename)
                with open(filepath, 'r') as file:
                    content = file.read()
                    sections = nltk.sent_tokenize(content[:max_article_len])
                    valid_sections_count = 0
                    for section_idx, section in enumerate(sections[:max_sentences]):
                        tokenized_desc = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(section[:max_sent_len]))[:self.block_size]
                        examples.append((tokenized_desc, len(tokenized_desc), idx_article, valid_sections_count, section))
                        indices_map.append((idx_article, valid_sections_count))
                        valid_sections_count += 1

            with open(cached_features_file, "wb") as handle:
                pickle.dump((examples, indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)

        return examples, indices_map

    def __len__(self):
        return len(self.indices_map)
    
    def __getitem__(self, item):
        idx_article, idx_section = self.indices_map[item]
        sent = self.examples[idx_article][0][idx_section]

        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sent[0]), dtype=torch.long)[:self.hparams.limit_tokens],
            self.examples[idx_article][1],
            self.examples[idx_article][0][idx_section][1],
            sent[1],
            idx_article,
            idx_section,
            item,
            self.labels[item],
        )

class WikipediaTextDatasetParagraphsSentencesTest(WikipediaTextDatasetParagraphsSentences):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="test"):
        super().__init__(tokenizer, hparams, dataset_name, block_size, mode=mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sections = []
        for idx_section, section in enumerate(self.examples[item][0]):
            sentences = []
            sentences.append(
                (
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(section[0]), dtype=torch.long),
                    self.examples[item][1],
                    section[1],
                    section[1],
                    item,
                    idx_section,
                    item,
                    self.labels[item],
                )
            )
            sections.append(sentences)
        return sections
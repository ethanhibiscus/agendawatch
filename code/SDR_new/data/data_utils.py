import pickle
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence


def get_gt_seeds_titles(titles=None, dataset_name="wines"):
    idxs = None
    gt_path = f"data/datasets/{dataset_name}/gt"
    popular_titles = list(pickle.load(open(gt_path, "rb")).keys())
    if titles != None:
        idxs = [titles.index(pop_title) for pop_title in popular_titles if pop_title in titles]
    return popular_titles, idxs, gt_path


def reco_sentence_collate(examples: List[torch.Tensor], tokenizer):
    return (
        pad_sequence([i[0] for i in examples], batch_first=True, padding_value=tokenizer.pad_token_id),
        [i[1] for i in examples],  # Title
        [i[2] for i in examples],  # Section Title
        [i[3] for i in examples],  # Length of sentence
        [i[4] for i in examples],  # idx_article
        [i[5] for i in examples],  # idx_section
        [i[6] for i in examples],  # idx_sentence
        torch.tensor([i[7] for i in examples]),  # Item index
    )

def reco_sentence_test_collate(examples: List[torch.Tensor], tokenizer):
    sections = []
    for example in examples:
        sentences = []
        for sentence in example:
            sentences.append(
                (
                    pad_sequence([i[0] for i in sentence], batch_first=True, padding_value=tokenizer.pad_token_id),
                    [i[1] for i in sentence],  # Title
                    [i[2] for i in sentence],  # Section Title
                    [i[3] for i in sentence],  # Length of sentence
                    [i[4] for i in sentence],  # idx_article
                    [i[5] for i in sentence],  # idx_section
                    [i[6] for i in sentence],  # idx_sentence
                    torch.tensor([i[7] for i in sentence]),  # Item index
                )
            )
        sections.append(sentences)
    return sections

def raw_data_link(dataset_name):
    if dataset_name == "wines":
        return "https://zenodo.org/record/4812960/files/wines.txt?download=1"
    if dataset_name == "video_games":
        return "https://zenodo.org/record/4812962/files/video_games.txt?download=1"

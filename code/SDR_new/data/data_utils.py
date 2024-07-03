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


def reco_sentence_collate(examples, tokenizer):
    input_ids = pad_sequence([i[0] for i in examples], batch_first=True, padding_value=tokenizer.pad_token_id)
    lengths = [i[1] for i in examples]
    labels = [i[2] for i in examples]
    return input_ids, lengths, labels

def reco_sentence_test_collate(examples, tokenizer):
    input_ids = [pad_sequence([s[0] for s in sec], batch_first=True, padding_value=tokenizer.pad_token_id) for sec in examples]
    lengths = [[s[1] for s in sec] for sec in examples]
    labels = [[s[2] for s in sec] for sec in examples]
    return input_ids, lengths, labels


def raw_data_link(dataset_name):
    if dataset_name == "wines":
        return "https://zenodo.org/record/4812960/files/wines.txt?download=1"
    if dataset_name == "video_games":
        return "https://zenodo.org/record/4812962/files/video_games.txt?download=1"

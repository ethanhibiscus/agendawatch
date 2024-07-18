import pickle
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence

def get_gt_seeds_titles(titles=None, dataset_name="wines"):
    """
    Retrieves ground truth seeds and titles for a given dataset.
    This function helps in identifying which titles from the dataset are popular 
    and their respective indices.

    Args:
        titles (list, optional): List of titles. Defaults to None.
        dataset_name (str): Name of the dataset. Defaults to "wines".

    Returns:
        tuple: A tuple containing the list of popular titles, their indices, and the ground truth path.
    """
    idxs = None
    # Path to the ground truth data
    gt_path = f"data/datasets/{dataset_name}/gt"
    # Load the popular titles from the ground truth file
    popular_titles = list(pickle.load(open(gt_path, "rb")).keys())
    # If titles are provided, find the indices of popular titles in the given titles list
    if titles is not None:
        idxs = [titles.index(pop_title) for pop_title in popular_titles if pop_title in titles]
    return popular_titles, idxs, gt_path

def reco_sentence_test_collate(examples: List[torch.Tensor], tokenizer):
    """
    Collate function for preparing batches of data for testing.
    This function handles the padding and organization of text data 
    into a format suitable for model testing.

    Args:
        examples (list): List of examples where each example is a list of sections.
        tokenizer: Tokenizer for text processing and padding.

    Returns:
        list: Processed list of sections with padded sequences and metadata.
    """
    examples_ = []
    for example in examples:
        sections = []
        for section in example:
            if section == []:
                continue
            # For each section, create a tuple with padded sentences and various metadata
            sections.append(
                (
                    pad_sequence([i[0] for i in section], batch_first=True, padding_value=tokenizer.pad_token_id),
                    [i[2] for i in section],  # Article indices
                    [i[3] for i in section],  # Section indices
                    [i[4] for i in section],  # Sentence indices
                    [i[5] for i in section],  # Original sentences
                    [i[6] for i in section],  # Sentence lengths
                    [i[7] for i in section],  # Item indices
                    torch.tensor([i[8] for i in section]),  # Labels as tensors
                )
            )
        # Append the processed sections for each example
        examples_.append(sections)
    return examples_

def reco_sentence_collate(examples: List[torch.Tensor], tokenizer):
    """
    Collate function for preparing batches of data for training and validation.
    This function handles the padding and organization of text data 
    into a format suitable for model training and validation.

    Args:
        examples (list): List of examples where each example is a tensor.
        tokenizer: Tokenizer for text processing and padding.

    Returns:
        tuple: A tuple containing the padded sequences and metadata.
    """
    # Create a tuple with padded sentences and various metadata
    return (
        pad_sequence([i[0] for i in examples], batch_first=True, padding_value=tokenizer.pad_token_id),  # Padded sequences
        [i[2] for i in examples],  # Article indices
        [i[3] for i in examples],  # Section indices
        [i[4] for i in examples],  # Sentence indices
        [i[5] for i in examples],  # Original sentences
        [i[6] for i in examples],  # Sentence lengths
        [i[7] for i in examples],  # Item indices
        torch.tensor([i[8] for i in examples]),  # Labels as tensors
    )

def raw_data_link(dataset_name):
    """
    Returns the download link for raw data based on the dataset name.
    This function provides URLs to download datasets needed for the model.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        str: URL link to download the raw data.
    """
    if dataset_name == "wines":
        return "https://zenodo.org/record/4812960/files/wines.txt?download=1"
    if dataset_name == "video_games":
        return "https://zenodo.org/record/4812962/files/video_games.txt?download=1"

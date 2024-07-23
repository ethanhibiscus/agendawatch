import os
import torch
import random
from tqdm import tqdm
from pre_process import preprocess_documents
from similarity_matrices import compute_similarity_matrices
from normalization import normalize_matrices
from ranking import rank_documents
from utils.argparse_init import init_parse_argparse_default_params
from argparse import Namespace

def main(model_weights_path, data_dir, cache_dir):
    # Initialize hyperparameters
    parser = init_parse_argparse_default_params(argparse.ArgumentParser(), dataset_name="custom_dataset", arch="SDR")
    hparams = Namespace(**vars(parser.parse_args()))
    
    # Step 1: Pre-process text files or load caches
    print("Step 1: Pre-processing documents...")
    processed_data = preprocess_documents(data_dir, cache_dir, model_weights_path, hparams)
    print(f"Pre-processed and generated embeddings for {len(processed_data)} documents.")
    
    # Select a random document as the source document
    source_document_path = random.choice(processed_data)[0]
    print(f"Selected '{source_document_path}' as the source document.")

    # Step 2: Compute similarity matrices
    print("Step 2: Computing similarity matrices...")
    similarity_matrices = compute_similarity_matrices(processed_data, source_document_path)
    print(f"Computed similarity matrices for {len(similarity_matrices)} documents.")

    # Step 3: Normalize matrices
    print("Step 3: Normalizing matrices...")
    normalized_matrices = normalize_matrices(similarity_matrices)
    print("Normalized similarity matrices.")

    # Step 4: Rank documents
    print("Step 4: Ranking documents...")
    ranked_documents = rank_documents(normalized_matrices)
    print("Ranked documents based on similarity scores.")

    return ranked_documents

if __name__ == "__main__":
    model_weights_path = "~/03_07_2024-23_10_34"  # Path to the trained model weights
    data_dir = "./data/text_files"
    cache_dir = "./SDR_inference/cache"
    
    ranked_documents = main(model_weights_path, data_dir, cache_dir)
    print("Ranked Documents:", ranked_documents)

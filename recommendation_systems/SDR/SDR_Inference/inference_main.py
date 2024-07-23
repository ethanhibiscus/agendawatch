import os
import argparse
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
import torch
import pickle

from pre_process import preprocess_documents
from data.datasets import init_parse_argparse_default_params

def main(model_weights_path, data_dir, cache_dir):
    print("Initializing argument parser...")
    parser = init_parse_argparse_default_params(argparse.ArgumentParser(), dataset_name="custom_dataset", arch="SDR")
    hparams = vars(parser.parse_args())

    # Add default values for missing parameters
    default_params = {
        'limit_tokens': 512,
        'batch_size': 8,
        'num_workers': 4
    }
    for key, value in default_params.items():
        if key not in hparams:
            hparams[key] = value
    hparams = argparse.Namespace(**hparams)

    print("Step 1: Pre-processing documents...")
    processed_data = preprocess_documents(data_dir, cache_dir, model_weights_path, hparams)

    print("Step 2: Selecting a random source document...")
    source_document = random.choice(processed_data)
    source_embedding = source_document[1]

    print("Step 3: Calculating similarity scores...")
    similarity_scores = []
    for filename, embedding in tqdm(processed_data, desc="Calculating similarity scores", unit="doc"):
        similarity = torch.cosine_similarity(source_embedding, embedding)
        similarity_scores.append((filename, similarity.item()))

    print("Step 4: Ranking documents...")
    ranked_documents = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return ranked_documents

if __name__ == "__main__":
    model_weights_path = os.path.expanduser('~/03_07_2024-23_10_34/last.ckpt')
    data_dir = './data/text_files'
    cache_dir = './cache_dir'
    ranked_documents = main(model_weights_path, data_dir, cache_dir)
    print("Top 5 similar documents:")
    for doc in ranked_documents[:5]:
        print(doc)

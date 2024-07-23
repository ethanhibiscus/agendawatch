import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_utils import extract_model_path_for_hyperparams
from models.SDR.SDR import SDR
from SDR_inference.data_loader import load_data
from utils.inference_utils import compute_sentence_embeddings, compute_similarity_scores, rank_documents

def get_latest_checkpoint(directory):
    checkpoints = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ckpt')]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in directory: {directory}")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint

def main():
    print("Starting inference...")

    # Define paths
    model_dir = os.path.expanduser("~/03_07_2024-23_10_34")
    data_dir = "./data/text_files"
    print(f"Loading model from: {model_dir}")
    model_path = get_latest_checkpoint(model_dir)
    
    # Load the model
    print(f"Loading model from checkpoint: {model_path}")
    model = SDR.load_from_checkpoint(model_path)
    model.eval()
    print("Model loaded successfully.")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    documents = load_data(data_dir)
    print("Data loaded and preprocessed successfully.")

    # Compute sentence embeddings for each document
    print("Computing sentence embeddings for each document...")
    doc_embeddings = compute_sentence_embeddings(model, documents)
    print("Sentence embeddings computed successfully.")

    # Compute similarity scores
    print("Computing similarity scores...")
    similarity_scores = compute_similarity_scores(doc_embeddings)
    print("Similarity scores computed successfully.")

    # Rank documents
    print("Ranking documents...")
    ranked_docs = rank_documents(similarity_scores)
    print("Documents ranked successfully.")

    # Print or save the ranked documents
    print("Ranked documents:")
    for rank, doc in enumerate(ranked_docs):
        print(f"Rank {rank + 1}: {doc}")

if __name__ == "__main__":
    main()

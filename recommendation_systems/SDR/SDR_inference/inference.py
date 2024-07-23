import os
import torch
from utils.model_utils import extract_model_path_for_hyperparams
from models.SDR.SDR import SDR
from SDR_inference.data_loader import load_data
from SDR_inference.utils import compute_sentence_embeddings, compute_similarity_scores, rank_documents

def main():
    # Define paths
    model_dir = "./output/document_similarity/arch_SDR/dataset_name_custom_dataset/test_only_False"
    data_dir = "./data/text_files"
    model_path = extract_model_path_for_hyperparams(model_dir, SDR)
    
    # Load the model
    model = SDR.load_from_checkpoint(model_path)
    model.eval()

    # Load and preprocess data
    documents = load_data(data_dir)

    # Compute sentence embeddings for each document
    doc_embeddings = compute_sentence_embeddings(model, documents)

    # Compute similarity scores
    similarity_scores = compute_similarity_scores(doc_embeddings)

    # Rank documents
    ranked_docs = rank_documents(similarity_scores)

    # Print or save the ranked documents
    for rank, doc in enumerate(ranked_docs):
        print(f"Rank {rank + 1}: {doc}")

if __name__ == "__main__":
    main()

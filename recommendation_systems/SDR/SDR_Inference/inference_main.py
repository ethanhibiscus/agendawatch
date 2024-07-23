import os
import torch
import random
from pre_process import preprocess_documents
from sentence_embeddings import generate_sentence_embeddings
from similarity_matrices import compute_similarity_matrices
from normalization import normalize_matrices
from ranking import rank_documents

def main(model_weights_path, data_dir, cache_dir):
    print("Starting inference process...")
    
    # Step 1: Pre-process text files or load caches
    print("Step 1: Pre-processing text files or loading caches...")
    processed_data = preprocess_documents(data_dir, cache_dir)
    print(f"Processed {len(processed_data)} documents.")
    
    # Select a random source document
    source_document = random.choice(processed_data)[0]
    print(f"Selected source document: {source_document}")
    
    # Step 2: Generate sentence embeddings
    print("Step 2: Generating sentence embeddings...")
    sentence_embeddings = generate_sentence_embeddings(processed_data, model_weights_path)
    print("Generated sentence embeddings.")
    
    # Step 3: Compute similarity matrices
    print("Step 3: Computing similarity matrices...")
    similarity_matrices = compute_similarity_matrices(sentence_embeddings, source_document)
    print("Computed similarity matrices.")
    
    # Step 4: Normalize matrices
    print("Step 4: Normalizing matrices...")
    normalized_matrices = normalize_matrices(similarity_matrices)
    print("Normalized matrices.")
    
    # Step 5: Rank documents
    print("Step 5: Ranking documents...")
    ranked_documents = rank_documents(normalized_matrices)
    print("Ranked documents.")
    
    return ranked_documents

if __name__ == "__main__":
    model_weights_path = "~/03_07_2024-23_10_34"  # Path to the trained model weights
    data_dir = "./data/text_files"
    cache_dir = "./SDR_inference/cache"
    
    ranked_documents = main(model_weights_path, data_dir, cache_dir)
    print("Ranked Documents:", ranked_documents)

import torch
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

def compute_similarity_matrices(processed_data, source_document_path):
    source_embeddings = None
    for filename, embeddings in processed_data:
        if filename == source_document_path:
            source_embeddings = embeddings
            break
    
    if source_embeddings is None:
        raise ValueError("Source document not found in the processed data.")
    
    similarity_matrices = []
    for filename, embeddings in tqdm(processed_data, desc="Computing Similarity Matrices"):
        if filename != source_document_path:
            sim_matrix = cosine_similarity(source_embeddings, embeddings.unsqueeze(1)).mean(dim=1)
            similarity_matrices.append((filename, sim_matrix))
    
    return similarity_matrices

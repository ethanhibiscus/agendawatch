import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def compute_similarity_matrices(sentence_embeddings, source_document):
    source_embeddings = None
    for filename, embeddings in sentence_embeddings:
        if filename == source_document:
            source_embeddings = embeddings
            break
    
    if source_embeddings is None:
        raise ValueError("Source document not found in the processed data.")
    
    similarity_matrices = []
    print("Computing similarity matrices...")
    for filename, embeddings in tqdm(sentence_embeddings, desc="Computing similarity"):
        if filename != source_document:
            sim_matrix = cosine_similarity(source_embeddings, embeddings.unsqueeze(1)).mean(dim=1)
            similarity_matrices.append((filename, sim_matrix))
    
    return similarity_matrices

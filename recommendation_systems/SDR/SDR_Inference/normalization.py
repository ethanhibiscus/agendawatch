import torch
from tqdm import tqdm

def normalize_matrices(similarity_matrices):
    normalized_matrices = []
    for filename, sim_matrix in tqdm(similarity_matrices, desc="Normalizing Matrices"):
        mean = torch.mean(sim_matrix)
        std = torch.std(sim_matrix)
        normalized_matrix = (sim_matrix - mean) / std
        normalized_matrices.append((filename, normalized_matrix))
    
    return normalized_matrices

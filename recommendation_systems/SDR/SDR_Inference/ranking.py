def rank_documents(normalized_matrices):
    print("Ranking documents...")
    document_scores = []
    for filename, normalized_matrix in normalized_matrices:
        score = normalized_matrix.mean().item()
        document_scores.append((filename, score))
    
    ranked_documents = sorted(document_scores, key=lambda x: x[1], reverse=True)
    
    return ranked_documents

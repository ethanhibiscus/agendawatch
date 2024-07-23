import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def compute_sentence_embeddings(model, documents):
    embeddings = {}
    for doc_id, doc in tqdm(documents.items(), desc="Computing embeddings"):
        sentences = doc.split(".")
        sentence_embeddings = []
        for sentence in sentences:
            inputs = model.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            sentence_embeddings.append(embeddings)
        embeddings[doc_id] = torch.stack(sentence_embeddings)
    return embeddings

def compute_similarity_scores(doc_embeddings):
    scores = {}
    for doc_id, embeddings in tqdm(doc_embeddings.items(), desc="Computing similarity scores"):
        sim_matrix = cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        scores[doc_id] = sim_matrix
    return scores

def rank_documents(similarity_scores):
    total_scores = {doc_id: torch.sum(scores).item() for doc_id, scores in similarity_scores.items()}
    ranked_docs = sorted(total_scores.items(), key=lambda item: item[1], reverse=True)
    return [doc[0] for doc in ranked_docs]

import torch
from tqdm import tqdm

def compute_sentence_embeddings(model, documents):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for doc in tqdm(documents, desc="Computing embeddings"):
            inputs = model.tokenizer(doc, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings.append(outputs[0].mean(dim=1).squeeze().cpu().numpy())
    return embeddings

def compute_similarity_scores(doc_embeddings):
    scores = []
    for i in tqdm(range(len(doc_embeddings)), desc="Computing similarity scores"):
        for j in range(i + 1, len(doc_embeddings)):
            score = torch.cosine_similarity(
                torch.tensor(doc_embeddings[i]).unsqueeze(0),
                torch.tensor(doc_embeddings[j]).unsqueeze(0)
            )
            scores.append((i, j, score.item()))
    return scores

def rank_documents(similarity_scores):
    ranked_docs = sorted(similarity_scores, key=lambda x: x[2], reverse=True)
    return ranked_docs

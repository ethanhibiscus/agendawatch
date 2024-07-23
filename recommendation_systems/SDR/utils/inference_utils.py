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
            print(f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            try:
                # Adjust the forward pass call according to the model's expected inputs
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=None,  # Adjust if necessary
                    position_ids=None,  # Adjust if necessary
                    head_mask=None,  # Adjust if necessary
                    inputs_embeds=None,  # Adjust if necessary
                    run_mlm=False,  # Set to False if not using MLM during inference
                    run_similarity=True  # Set to True if using similarity computation
                )
                embeddings.append(outputs[2].mean(dim=1).squeeze().cpu().numpy())  # Assuming outputs[2] is the relevant output
            except TypeError as e:
                print(f"Error in model forward pass: {e}")
                print(f"Model input args: input_ids={input_ids}, attention_mask={attention_mask}")
                raise e
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

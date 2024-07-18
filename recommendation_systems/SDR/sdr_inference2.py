# ./sdr_inference_hierarchical.py

import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.functional import cosine_similarity
import numpy as np
from tqdm import tqdm

def load_model_and_tokenizer(model_path, tokenizer_path):
    model = RobertaModel.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def preprocess_document(tokenizer, document, max_length=512):
    paragraphs = document.split("\n\n")
    inputs = [tokenizer(paragraph, padding=True, truncation=True, max_length=max_length, return_tensors='pt') for paragraph in paragraphs]
    return inputs

def generate_sentence_embeddings(model, inputs):
    embeddings = []
    with torch.no_grad():
        for input_batch in inputs:
            outputs = model(input_batch['input_ids'], attention_mask=input_batch['attention_mask'])
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            embeddings.append(sentence_embeddings)
    return embeddings

def compute_sentence_similarity_matrix(source_paragraph, candidate_paragraph):
    sentence_sim_matrix = cosine_similarity(source_paragraph.unsqueeze(1), candidate_paragraph.unsqueeze(0), dim=2)
    return sentence_sim_matrix

def compute_paragraph_similarity(source_embeddings, candidate_embeddings):
    paragraph_similarities = []
    for source_paragraph in source_embeddings:
        paragraph_scores = []
        for candidate_paragraph in candidate_embeddings:
            sentence_sim_matrix = compute_sentence_similarity_matrix(source_paragraph, candidate_paragraph)
            paragraph_score = sentence_sim_matrix.max(dim=-1)[0].mean().item()
            paragraph_scores.append(paragraph_score)
        paragraph_similarities.append(paragraph_scores)
    return paragraph_similarities

def normalize_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    normalized_scores = [(score - mean) / (std if std > 0 else 1) for score in scores]
    return normalized_scores

def compute_document_similarity(paragraph_similarities):
    normalized_similarities = [normalize_scores(row) for row in paragraph_similarities]
    document_similarity_score = np.mean([max(row) for row in normalized_similarities])
    return document_similarity_score

def rank_documents(source_document, candidate_documents, model, tokenizer):
    source_inputs = preprocess_document(tokenizer, source_document)
    source_embeddings = generate_sentence_embeddings(model, source_inputs)

    candidate_embeddings_list = []
    for candidate_document in candidate_documents:
        candidate_inputs = preprocess_document(tokenizer, candidate_document)
        candidate_embeddings = generate_sentence_embeddings(model, candidate_inputs)
        candidate_embeddings_list.append(candidate_embeddings)

    similarity_scores = []
    for candidate_embeddings in candidate_embeddings_list:
        paragraph_similarities = compute_paragraph_similarity(source_embeddings, candidate_embeddings)
        document_similarity_score = compute_document_similarity(paragraph_similarities)
        similarity_scores.append(document_similarity_score)
    
    rankings = np.argsort(similarity_scores)[::-1]
    return rankings, similarity_scores

if __name__ == "__main__":
    model_path = "path/to/your/model"
    tokenizer_path = "path/to/your/tokenizer"
    source_document = "Your source document text."
    candidate_documents = ["Candidate document text 1.", "Candidate document text 2.", ...]

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    rankings, scores = rank_documents(source_document, candidate_documents, model, tokenizer)
    
    print("Rankings:", rankings)
    print("Scores:", scores)

# ./sdr_inference_hierarchical.py

import torch
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from models.reco.hierarchical_reco import vectorize_reco_hierarchical
from data.datasets import CustomTextDatasetParagraphsSentencesTest
from utils.torch_utils import to_numpy

def load_model_and_tokenizer(checkpoint_path, model_class, tokenizer_class, model_args):
    model = model_class.load_from_checkpoint(checkpoint_path, **model_args)
    tokenizer = tokenizer_class.from_pretrained("roberta-large")
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

def rank_documents(source_document, candidate_documents, model, tokenizer):
    source_inputs = preprocess_document(tokenizer, source_document)
    source_embeddings = generate_sentence_embeddings(model, source_inputs)

    candidate_embeddings_list = []
    for candidate_document in candidate_documents:
        candidate_inputs = preprocess_document(tokenizer, candidate_document)
        candidate_embeddings = generate_sentence_embeddings(model, candidate_inputs)
        candidate_embeddings_list.append(candidate_embeddings)

    source_embeddings_np = [to_numpy(embedding) for embedding in source_embeddings]
    candidate_embeddings_np_list = [[to_numpy(embedding) for embedding in candidate_embeddings] for candidate_embeddings in candidate_embeddings_list]

    titles = ["source"] + [f"candidate_{i}" for i in range(len(candidate_documents))]
    all_embeddings = [source_embeddings_np] + candidate_embeddings_np_list

    recos, metrics = vectorize_reco_hierarchical(all_embeddings, titles, gt_path="")

    similarity_scores = [metrics['mrr']]  # Assuming 'mrr' as the similarity score for demonstration
    rankings = np.argsort(similarity_scores)[::-1]
    
    return rankings, similarity_scores

if __name__ == "__main__":
    checkpoint_path = "~/03_07_2024-23_10_34/last.ckpt"
    model_class = RobertaModel  # Replace with the correct model class if different
    tokenizer_class = RobertaTokenizer
    model_args = {}  # Additional arguments if needed

    source_document = "Your source document text."
    candidate_documents = ["Candidate document text 1.", "Candidate document text 2.", ...]

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, model_class, tokenizer_class, model_args)
    
    rankings, scores = rank_documents(source_document, candidate_documents, model, tokenizer)
    
    print("Rankings:", rankings)
    print("Scores:", scores)

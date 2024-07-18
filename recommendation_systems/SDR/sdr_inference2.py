# ./sdr_inference_hierarchical.py

import os
import torch
from transformers import RobertaTokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
from models.reco.hierarchical_reco import vectorize_reco_hierarchical
from data.datasets import CustomTextDatasetParagraphsSentencesTest
from utils.torch_utils import to_numpy
from models.SDR.SDR import SDR  # Assuming SDR is your custom model class
import pytorch_lightning as pl

def load_model_and_tokenizer(checkpoint_path, model_class, tokenizer_name):
    checkpoint_path = os.path.expanduser(checkpoint_path)  # Expand the tilde to the home directory
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = checkpoint['hyper_parameters']
    model = model_class.load_from_checkpoint(checkpoint_path, hparams=hparams)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def preprocess_documents(tokenizer, source_document, candidate_documents):
    # Combine source and candidate documents into a single list
    all_documents = [source_document] + candidate_documents
    processed_documents = []

    for document in all_documents:
        paragraphs = document.split("\n\n")
        inputs = [tokenizer(paragraph, padding='max_length', truncation=True, max_length=512, return_tensors='pt') for paragraph in paragraphs]
        processed_documents.append(inputs)
    
    return processed_documents

def generate_sentence_embeddings(model, processed_documents):
    embeddings = []
    with torch.no_grad():
        for document in processed_documents:
            document_embeddings = []
            for paragraph in document:
                outputs = model(**paragraph)
                paragraph_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                document_embeddings.append(paragraph_embedding)
            embeddings.append(document_embeddings)
    return embeddings

def rank_documents(source_document, candidate_documents, model, tokenizer):
    processed_documents = preprocess_documents(tokenizer, source_document, candidate_documents)
    all_embeddings = generate_sentence_embeddings(model, processed_documents)

    source_embeddings_np = [to_numpy(embedding) for embedding in all_embeddings[0]]
    candidate_embeddings_np_list = [[to_numpy(embedding) for embedding in candidate_embeddings] for candidate_embeddings in all_embeddings[1:]]

    titles = ["source"] + [f"candidate_{i}" for i in range(len(candidate_documents))]
    all_embeddings = [source_embeddings_np] + candidate_embeddings_np_list

    recos, metrics = vectorize_reco_hierarchical(all_embeddings, titles, gt_path="")

    similarity_scores = [metrics['mrr']]  # Assuming 'mrr' as the similarity score for demonstration
    rankings = np.argsort(similarity_scores)[::-1]
    
    return rankings, similarity_scores

if __name__ == "__main__":
    checkpoint_path = "~/03_07_2024-23_10_34/last.ckpt"
    model_class = SDR  # Replace with the correct model class
    tokenizer_name = "roberta-large"

    source_document = "Your source document text."
    candidate_documents = ["Candidate document text 1.", "Candidate document text 2."]

    model, tokenizer = load_model_and_tokenizer(checkpoint_path, model_class, tokenizer_name)
    
    rankings, scores = rank_documents(source_document, candidate_documents, model, tokenizer)
    
    print("Rankings:", rankings)
    print("Scores:", scores)

import os
import torch
from transformers import RobertaTokenizer
from tqdm import tqdm
from models.SDR.SDR import SDR
from utils.pytorch_lightning_utils.pytorch_lightning_utils import load_params_from_checkpoint
from models.reco.recos_utils import sim_matrix
from utils.torch_utils import to_numpy
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def load_model(checkpoint_path):
    print("Loading model...")
    hparams = torch.load(checkpoint_path, map_location=torch.device('cpu'))['hyper_parameters']
    model = SDR.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location=torch.device('cpu'))
    model.eval()
    print("Model loaded successfully!")
    return model, hparams

def load_documents(data_path):
    print("Loading documents...")
    text_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    documents = []
    for file in tqdm(text_files, desc="Reading text files"):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            documents.append((file, content))
    print(f"Loaded {len(documents)} documents successfully!")
    return documents

def tokenize_and_pad(text, tokenizer, block_size):
    sentences = tokenizer.tokenize(text)
    tokenized_sentences = tokenizer.convert_tokens_to_ids(sentences[:block_size])
    return torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenized_sentences), dtype=torch.long)

def get_embeddings(documents, model, tokenizer, block_size=512):
    print("Generating embeddings for documents...")
    embeddings = []
    for doc in tqdm(documents, desc="Generating embeddings"):
        tokenized = tokenize_and_pad(doc[1], tokenizer, block_size)
        with torch.no_grad():
            embedding = model(tokenized.unsqueeze(0)).last_hidden_state.mean(1).squeeze(0)
        embeddings.append(embedding)
    print("Embeddings generated successfully!")
    return embeddings

def compute_similarity(embeddings):
    print("Computing similarity matrix...")
    embeddings = torch.stack(embeddings)
    similarity_matrix = sim_matrix(embeddings, embeddings)
    print("Similarity matrix computed successfully!")
    return similarity_matrix

def rank_documents(similarity_matrix, documents):
    print("Ranking documents based on similarity scores...")
    scores = similarity_matrix[0].cpu().numpy()  # Assuming the first document is the source document
    ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
    ranked_documents = [documents[i] for i in ranked_indices]
    print("Documents ranked successfully!")
    return ranked_documents, scores[ranked_indices]

def main():
    checkpoint_path = '~/03_07_2024-23_10_34/epoch=11.ckpt'
    data_path = './data/text_files'

    model, hparams = load_model(checkpoint_path)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    documents = load_documents(data_path)
    embeddings = get_embeddings(documents, model, tokenizer)
    similarity_matrix = compute_similarity(embeddings)
    ranked_documents, scores = rank_documents(similarity_matrix, documents)
    
    print("Ranking Results:")
    for doc, score in zip(ranked_documents, scores):
        print(f"Document: {doc[0]}, Score: {score}")

if __name__ == "__main__":
    main()

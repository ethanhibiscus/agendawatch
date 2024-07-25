import os
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer
from models.SDR.SDR import SDR
from models.reco.recos_utils import sim_matrix
import pickle

def load_model(checkpoint_path):
    """Loads the pretrained SDR model and its hyperparameters from a checkpoint file."""
    print("Loading model...")
    hparams = torch.load(checkpoint_path, map_location=torch.device('cpu'))['hyper_parameters']
    model = SDR.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")
    return model, hparams

def load_documents(data_path):
    """Loads all text documents from the specified directory."""
    print("Loading documents...")
    text_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')]
    documents = []
    for file in tqdm(text_files, desc="Reading text files"):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            documents.append((os.path.basename(file), content))
    print(f"Loaded {len(documents)} documents successfully!")
    return documents

def tokenize_and_pad(text, tokenizer, block_size):
    """Tokenizes and pads a text to a fixed block size using the specified tokenizer."""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens[:block_size-2])  # Reserve space for special tokens
    token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]  # Add special tokens
    token_ids += [tokenizer.pad_token_id] * (block_size - len(token_ids))  # Pad to block_size
    return torch.tensor(token_ids, dtype=torch.long)

def generate_embeddings(documents, model, tokenizer, block_size=512, output_dir='./inference_outputs'):
    """Generates embeddings for each document using the SDR model."""
    os.makedirs(output_dir, exist_ok=True)
    embeddings = []
    embedding_cache = {}

    # Load existing embeddings if available
    cache_path = os.path.join(output_dir, 'embedding_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            embedding_cache = pickle.load(f)
        print("Loaded existing embeddings from cache.")

    print("Generating embeddings for documents...")
    for doc in tqdm(documents, desc="Generating embeddings"):
        doc_title = doc[0]
        if doc_title in embedding_cache:
            embeddings.append(embedding_cache[doc_title])
        else:
            tokenized = tokenize_and_pad(doc[1], tokenizer, block_size).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                # Compute embeddings and average non-padded tokens
                sentence_embeddings = model.model(
                    tokenized, masked_lm_labels=None, run_similarity=False
                )[5].squeeze(0).mean(dim=0)
            embeddings.append(sentence_embeddings)
            embedding_cache[doc_title] = sentence_embeddings

    # Save updated embeddings cache
    with open(cache_path, 'wb') as f:
        pickle.dump(embedding_cache, f)
    print("Embeddings generated and cache updated successfully!")
    return embeddings

def compute_similarity_matrix(embeddings):
    """Computes the similarity matrix for the document embeddings."""
    print("Computing similarity matrix...")
    embeddings = torch.stack(embeddings)
    similarity_matrix = sim_matrix(embeddings, embeddings)
    print("Similarity matrix computed successfully!")
    return similarity_matrix

def save_intermediate_results(results, filename, output_dir):
    """Saves intermediate results to a specified file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    torch.save(results, file_path)
    print(f"Saved intermediate results to {file_path}")

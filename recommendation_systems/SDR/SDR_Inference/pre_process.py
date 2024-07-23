import os
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch

# Ensure the root directory is in the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets import CustomTextDatasetParagraphsSentences

def preprocess_documents(data_dir, cache_dir, model_weights_path, hparams):
    cache_path = os.path.join(cache_dir, "processed_data_with_embeddings.pkl")
    
    if os.path.exists(cache_path):
        print("Loading cached processed data...")
        with open(cache_path, "rb") as f:
            processed_data = pickle.load(f)
        return processed_data
    
    print("Initializing tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    # Load the checkpoint and extract the state_dict
    checkpoint = torch.load(model_weights_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    
    # Remove the 'roberta.' prefix from the keys in the state dictionary if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('roberta.'):
            new_state_dict[k[len('roberta.'):]] = v
        else:
            new_state_dict[k] = v
    
    print("Loading model state dictionary...")
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("Loading dataset...")
    dataset = CustomTextDatasetParagraphsSentences(tokenizer, hparams, dataset_name="custom_dataset", block_size=512, mode="test")
    
    processed_data = []
    with torch.no_grad():
        for data in tqdm(dataset, desc="Processing Documents and Generating Embeddings", unit="doc"):
            filename, tokens = data[1], data[0]
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            processed_data.append((filename, embeddings))
    
    print("Saving processed data to cache...")
    with open(cache_path, "wb") as f:
        pickle.dump(processed_data, f)
    
    return processed_data

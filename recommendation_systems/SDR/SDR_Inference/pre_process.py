import os
import pickle
from transformers import RobertaTokenizer
from tqdm import tqdm

def preprocess_documents(data_dir, cache_dir):
    cache_path = os.path.join(cache_dir, "processed_data.pkl")
    
    if os.path.exists(cache_path):
        print("Loading cached processed data...")
        with open(cache_path, "rb") as f:
            processed_data = pickle.load(f)
        return processed_data
    
    print("Processing documents...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    processed_data = []
    
    for filename in tqdm(os.listdir(data_dir), desc="Processing documents"):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                tokens = tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=512)
                processed_data.append((filename, tokens))
    
    print("Saving processed data to cache...")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(processed_data, f)
    
    return processed_data

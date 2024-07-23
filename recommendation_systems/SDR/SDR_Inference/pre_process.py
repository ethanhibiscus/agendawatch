import os
import pickle
from tqdm import tqdm
from data.datasets import CustomTextDatasetParagraphsSentences
from transformers import RobertaTokenizer

def preprocess_documents(data_dir, cache_dir, hparams):
    cache_path = os.path.join(cache_dir, "processed_data.pkl")
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            processed_data = pickle.load(f)
        return processed_data
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = CustomTextDatasetParagraphsSentences(tokenizer, hparams, dataset_name="custom_dataset", block_size=512, mode="test")
    
    processed_data = []
    for data in tqdm(dataset, desc="Processing Documents"):
        processed_data.append((data[1], data[0]))
    
    with open(cache_path, "wb") as f:
        pickle.dump(processed_data, f)
    
    return processed_data

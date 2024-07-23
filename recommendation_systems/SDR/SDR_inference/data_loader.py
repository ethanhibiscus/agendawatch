import os
from tqdm import tqdm

def load_data(data_dir):
    documents = {}
    print(f"Loading data from {data_dir}...")
    for filename in tqdm(os.listdir(data_dir), desc="Loading files"):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                documents[filename] = file.read().strip()
    return documents

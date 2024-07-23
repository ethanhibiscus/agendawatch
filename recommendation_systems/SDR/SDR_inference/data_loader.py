import os
from tqdm import tqdm

def load_data(data_dir):
    print(f"Loading data from: {data_dir}")
    documents = {}
    for filename in tqdm(os.listdir(data_dir), desc="Loading documents"):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                documents[filename] = file.read().strip()
    return documents

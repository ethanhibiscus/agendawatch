import os

def load_data(data_dir):
    documents = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                documents[filename] = file.read().strip()
    return documents

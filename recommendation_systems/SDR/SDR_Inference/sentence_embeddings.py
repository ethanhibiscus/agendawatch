import torch
from transformers import RobertaModel
from tqdm import tqdm

def generate_sentence_embeddings(processed_data, model_weights_path):
    print("Loading model...")
    model = RobertaModel.from_pretrained('roberta-base')
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()
    
    sentence_embeddings = []
    
    print("Generating sentence embeddings...")
    with torch.no_grad():
        for filename, tokens in tqdm(processed_data, desc="Generating embeddings"):
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            sentence_embeddings.append((filename, embeddings))
    
    return sentence_embeddings

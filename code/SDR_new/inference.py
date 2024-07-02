import torch
from transformers import RobertaTokenizer
from models.SDR.SDR import SDR

def load_model(model_path, hparams):
    model = SDR.load_from_checkpoint(model_path, hparams=hparams)
    return model

def generate_embeddings(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.roberta(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def compute_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)

if __name__ == "__main__":
    model_path = 'path/to/your/checkpoint.ckpt'
    hparams = ... # Load or define the hyperparameters used during training

    model = load_model(model_path, hparams)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Example usage
    text1 = "First document text."
    text2 = "Second document text."

    embedding1 = generate_embeddings(model, tokenizer, text1)
    embedding2 = generate_embeddings(model, tokenizer, text2)

    similarity = compute_similarity(embedding1, embedding2)
    print(f"Similarity: {similarity.item()}")

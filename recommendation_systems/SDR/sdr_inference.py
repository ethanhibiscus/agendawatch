import os
import torch
from transformers import RobertaTokenizer
from models.SDR.similarity_modeling import SimilarityModeling
from data.datasets import CustomTextDatasetParagraphsSentencesTest
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

# Load the model checkpoint
checkpoint_path = '~/03_07_2024-23_10_34/epoch=10.ckpt'
model = SimilarityModeling.load_from_checkpoint(checkpoint_path)

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# Load source document and target documents
source_document_path = './data/text_files/source.txt'
target_documents_dir = './data/text_files/targets/'

# Read source document
with open(source_document_path, 'r', encoding='utf-8') as file:
    source_text = file.read()

# Read target documents
target_texts = []
for filename in os.listdir(target_documents_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(target_documents_dir, filename), 'r', encoding='utf-8') as file:
            target_texts.append((filename, file.read()))

# Preprocess documents using the CustomTextDatasetParagraphsSentencesTest
source_dataset = CustomTextDatasetParagraphsSentencesTest(tokenizer, None, 'custom_dataset', tokenizer.max_len, mode='test')
target_datasets = [CustomTextDatasetParagraphsSentencesTest(tokenizer, None, 'custom_dataset', tokenizer.max_len, mode='test') for _, text in target_texts]

# Create DataLoader for source and target documents
source_loader = DataLoader(source_dataset, batch_size=1, shuffle=False)
target_loaders = [DataLoader(target_dataset, batch_size=1, shuffle=False) for target_dataset in target_datasets]

# Get embeddings for source document
source_embeddings = []
model.eval()
with torch.no_grad():
    for batch in source_loader:
        inputs = batch[0].to(model.device)
        outputs = model(inputs)
        source_embeddings.append(outputs[2].cpu())

# Get embeddings for target documents
target_embeddings = []
for target_loader in target_loaders:
    target_embedding = []
    with torch.no_grad():
        for batch in target_loader:
            inputs = batch[0].to(model.device)
            outputs = model(inputs)
            target_embedding.append(outputs[2].cpu())
    target_embeddings.append(torch.cat(target_embedding, dim=0))

# Compute cosine similarity
# Flatten source embeddings
source_embedding = torch.cat(source_embeddings, dim=0).mean(dim=0).numpy()

# Compute similarity scores for each target document
similarity_scores = []
for target_embedding in target_embeddings:
    target_embedding_mean = target_embedding.mean(dim=0).numpy()
    score = cosine_similarity([source_embedding], [target_embedding_mean])[0][0]
    similarity_scores.append(score)

# Rank target documents based on similarity scores
ranked_targets = sorted(zip(target_texts, similarity_scores), key=lambda x: x[1], reverse=True)

# Display ranked documents
print("Ranked target documents based on similarity to source document:")
for (filename, _), score in ranked_targets:
    print(f"Document: {filename}, Similarity Score: {score}")

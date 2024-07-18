import os
import torch
import nltk
import random
from transformers import RobertaTokenizer
from models.SDR.SDR import SDR
from utils.argparse_init import default_arg_parser, init_parse_argparse_default_params
import torch.nn.functional as F

nltk.download('punkt')  # Ensure that the punkt package is downloaded

# Function to tokenize the new documents
def tokenize_document(document, tokenizer):
    paragraphs = document.split('\n\n')
    tokenized_paragraphs = []
    for paragraph in paragraphs:
        sentences = nltk.sent_tokenize(paragraph)
        tokenized_sentences = [tokenizer.encode(sentence, add_special_tokens=True, max_length=128, truncation=True) for sentence in sentences]
        tokenized_paragraphs.append(tokenized_sentences)
    return tokenized_paragraphs

# Function to compute embeddings
def compute_embeddings(tokenized_documents, model, tokenizer):
    all_embeddings = []
    for tokenized_paragraphs in tokenized_documents:
        paragraph_embeddings = []
        for tokenized_sentences in tokenized_paragraphs:
            sentences_tensor = [torch.tensor(sent, dtype=torch.long).unsqueeze(0) for sent in tokenized_sentences]
            with torch.no_grad():
                sentences_embeddings = [model.roberta(sent)[0].mean(1) for sent in sentences_tensor]  # Mean pooling
            paragraph_embeddings.append(torch.stack(sentences_embeddings))
        all_embeddings.append(paragraph_embeddings)
    return all_embeddings

# Function to compute similarity
def compute_similarity(doc_embeddings1, doc_embeddings2):
    similarities = []
    for para1 in doc_embeddings1:
        for para2 in doc_embeddings2:
            sim_matrix = F.cosine_similarity(para1.unsqueeze(1), para2.unsqueeze(0), dim=-1)
            paragraph_similarity = sim_matrix.max(dim=-1)[0].mean().item()  # Aggregating similarities
            similarities.append(paragraph_similarity)
    return sum(similarities) / len(similarities)

# Function to read all documents from a directory
def read_documents_from_directory(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                documents[filename] = content
    return documents

# Main function for inference
def main():
    parser = default_arg_parser()
    init_parse_argparse_default_params(parser)

    # Parse the arguments
    hparams = parser.parse_args()

    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    # Read documents from the directory
    document_dir = './data/text_files'
    documents = read_documents_from_directory(document_dir)

    if not documents:
        print(f"No documents found in the directory {document_dir}")
        return

    # Randomly select a source document
    source_filename = random.choice(list(documents.keys()))
    source_document = documents.pop(source_filename)
    
    # Tokenize source document and other documents
    tokenized_source_document = tokenize_document(source_document, tokenizer)
    tokenized_documents = {filename: tokenize_document(content, tokenizer) for filename, content in documents.items()}

    # Load the trained model
    hparams.resume_from_checkpoint = os.path.expanduser('~/03_07_2024-23_10_34/epoch=10.ckpt')  # Update with actual path
    model = SDR(hparams)
    checkpoint = torch.load(hparams.resume_from_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Compute embeddings
    source_embeddings = compute_embeddings([tokenized_source_document], model, tokenizer)[0]
    document_embeddings = {filename: compute_embeddings([tokenized_content], model, tokenizer)[0] for filename, tokenized_content in tokenized_documents.items()}

    # Compute similarity and find top 15 similar documents
    similarities = []
    for filename, embeddings in document_embeddings.items():
        similarity_score = compute_similarity(source_embeddings, embeddings)
        similarities.append((filename, similarity_score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_15_similar_documents = similarities[:15]

    # Print out the title of the top 15 most similar documents
    print(f"Source document: {source_filename}")
    print("Top 15 most similar documents:")
    for filename, score in top_15_similar_documents:
        print(f"{filename}: {score}")

if __name__ == "__main__":
    main()

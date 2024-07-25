# inference_main.py

import os
import torch
import pickle
from tqdm import tqdm
from transformers import RobertaTokenizer
from models.SDR.SDR import SDR
from utils.argparse_init import default_arg_parser
from utils.switch_functions import model_class_pointer
from utils.model_utils import extract_model_path_for_hyperparams
from data.datasets import CustomTextDatasetParagraphsSentences
from models.reco.recos_utils import sim_matrix
from models.reco.hierarchical_reco import vectorize_reco_hierarchical

def load_model_and_tokenizer(model_path):
    """ Load the trained SDR model and tokenizer """
    print(f"Loading model and tokenizer from {model_path}...")
    hparams_path = os.path.join(model_path, 'hparams.pkl')
    with open(hparams_path, 'rb') as f:
        hparams = pickle.load(f)

    model_class = model_class_pointer(hparams.task_name, hparams.arch)
    model = model_class.load_from_checkpoint(
        checkpoint_path=os.path.join(model_path, 'last.ckpt'),
        hparams=hparams
    )

    tokenizer = RobertaTokenizer.from_pretrained(hparams.tokenizer_name)

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer, hparams

def prepare_datasets(tokenizer, hparams):
    """ Prepare datasets for inference """
    print("Preparing datasets for inference...")
    dataset = CustomTextDatasetParagraphsSentences(
        tokenizer=tokenizer,
        hparams=hparams,
        dataset_name=hparams.dataset_name,
        block_size=hparams.block_size,
        mode="test"
    )
    print("Datasets prepared successfully.")
    return dataset

def compute_similarity(model, source_doc, candidate_docs):
    """ Compute similarity scores between source document and candidate documents """
    print("Computing similarity scores...")
    source_embeddings = model.encode_document(source_doc)
    candidate_embeddings = [model.encode_document(doc) for doc in tqdm(candidate_docs, desc="Encoding candidate documents")]

    similarities = []
    for candidate_embedding in tqdm(candidate_embeddings, desc="Computing similarities"):
        sim = sim_matrix(source_embeddings, candidate_embedding)
        similarities.append(sim.mean().item())

    print("Similarity scores computed successfully.")
    return similarities

def rank_documents(similarities, candidate_docs):
    """ Rank candidate documents based on similarity scores """
    print("Ranking candidate documents...")
    ranked_docs = sorted(zip(similarities, candidate_docs), reverse=True, key=lambda x: x[0])
    print("Candidate documents ranked successfully.")
    return [doc for _, doc in ranked_docs]

def main():
    parser = default_arg_parser()
    args = parser.parse_args()

    print("loading model...")
    model_path = "~/03_07_2024-23_10_34"
    model, tokenizer, hparams = load_model_and_tokenizer(model_path)
    model.eval()
    print("model loaded")

    dataset = prepare_datasets(tokenizer, hparams)

    source_doc_path = os.path.join(args.source_doc)
    print(f"Loading source document from {source_doc_path}...")
    with open(source_doc_path, 'r') as f:
        source_doc = f.read()
    print("Source document loaded successfully.")

    print("Loading candidate documents from ./data/text_files...")
    candidate_docs = []
    for filename in tqdm(os.listdir('./data/text_files'), desc="Loading candidate documents"):
        if filename.endswith('.txt'):
            with open(os.path.join('./data/text_files', filename), 'r') as f:
                candidate_docs.append(f.read())
    print("Candidate documents loaded successfully.")

    similarities = compute_similarity(model, source_doc, candidate_docs)
    ranked_docs = rank_documents(similarities, candidate_docs)

    output_path = os.path.join(model_path, 'ranked_docs.txt')
    print(f"Saving ranked documents to {output_path}...")
    with open(output_path, 'w') as f:
        for doc in ranked_docs:
            f.write(doc + '\n')
    print("Ranked documents saved successfully.")

if __name__ == "__main__":
    main()

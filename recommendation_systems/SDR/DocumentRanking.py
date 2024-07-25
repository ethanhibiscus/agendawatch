import os
import torch
import random
import numpy as np
from utils.ranking_utils import load_model, load_documents, generate_embeddings, compute_similarity_matrix, save_intermediate_results
from transformers import RobertaTokenizer

class DocumentRanking:
    def __init__(self, checkpoint_path='~/03_07_2024-23_10_34/epoch=11.ckpt', data_path='./data/text_files', output_dir='./inference_outputs'):
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.data_path = data_path
        self.output_dir = output_dir
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.documents = []
        self.embeddings = []
        self.similarity_matrix = None
        self.model, self.hparams = load_model(self.checkpoint_path)
        self.documents = load_documents(self.data_path)
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)

    def add_new_document(self, docs):
        """Adds a new document or directory of documents, generates embeddings, and updates the similarity matrix."""
        if os.path.isdir(docs):
            new_docs = [os.path.join(docs, f) for f in os.listdir(docs) if f.endswith('.txt')]
        else:
            new_docs = [docs]
        
        for doc in new_docs:
            with open(doc, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.documents.append((doc, content))
        
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)
        save_intermediate_results(self.embeddings, 'embeddings.pt', self.output_dir)
        save_intermediate_results(self.similarity_matrix, 'similarity_matrix.pt', self.output_dir)

    def rank_documents(self, source=None):
        """Ranks documents based on their similarity to the source document."""
        if source is None:
            source_idx = random.randint(0, len(self.documents) - 1)
        elif isinstance(source, int):
            source_idx = source
        else:
            source_idx = next(i for i, doc in enumerate(self.documents) if doc[0] == source)
        
        scores = self.similarity_matrix[source_idx].cpu().numpy()
        ranked_indices = np.argsort(scores)[::-1]
        ranked_documents = [(self.documents[i], scores[i]) for i in ranked_indices[:100]]
        return ranked_documents

    def update_checkpoint_path(self, path):
        """Updates the checkpoint path for the model."""
        self.checkpoint_path = os.path.expanduser(path)
        self.model, self.hparams = load_model(self.checkpoint_path)

    def update_document(self, doc_name, new_document):
        """Updates a document by name or index with new content."""
        if isinstance(doc_name, int):
            index = doc_name
        else:
            index = next(i for i, doc in enumerate(self.documents) if doc[0] == doc_name)
        
        self.documents[index] = (self.documents[index][0], new_document)
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)
        save_intermediate_results(self.embeddings, 'embeddings.pt', self.output_dir)
        save_intermediate_results(self.similarity_matrix, 'similarity_matrix.pt', self.output_dir)

    def delete_document(self, doc_name):
        """Deletes a document by name or index."""
        if isinstance(doc_name, int):
            index = doc_name
        else:
            index = next(i for i, doc in enumerate(self.documents) if doc[0] == doc_name)
        
        del self.documents[index]
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)
        save_intermediate_results(self.embeddings, 'embeddings.pt', self.output_dir)
        save_intermediate_results(self.similarity_matrix, 'similarity_matrix.pt', self.output_dir)

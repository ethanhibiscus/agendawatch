'''
DocumentRanking class:
This class was designed to allow easy interfacing with Self-Supervised Document Ranking based on Microsoft's research. 

Class:
    DocumentRanking:
        Initialization:
            checkpoint_path ==> initial checkpoint path 
            data_path => initial path for documents #might remove in the future as right now the model will recompute new docs from the last time it was called
            output_dir => directory where intermediate outputs are saved
        Methods:
            add_documents(docs): Adds new documents or a directory of documents, generates embeddings, and updates the similarity matrix.
            rank_documents(source=None): Ranks documents based on their similarity to a source document.
            update_checkpoint_path(path): Updates the checkpoint path for the model.
            update_document(doc_name, new_document): Updates a document by name or index with new content.
            delete_document(doc_name): Deletes a document by name or index.
note: you cannot have two documents with the same name: functionality in this case is unpredictable!
'''

import os
import torch
import random
import numpy as np
from utils.ranking_utils import load_model, load_documents, generate_embeddings, compute_similarity_matrix, save_intermediate_results
from transformers import RobertaTokenizer

class DocumentRanking:
    """
    DocumentRanking class for managing and ranking text documents based on their similarity.

    Attributes:
        checkpoint_path (str): Path to the model checkpoint.
        output_dir (str): Directory to save intermediate outputs.
        tokenizer (RobertaTokenizer): Tokenizer for processing text documents.
        documents (list): List of tuples containing document titles and contents.
        embeddings (list): List of document embeddings.
        similarity_matrix (torch.Tensor): Matrix of similarity scores between documents.
        model (SDR): Loaded SDR model.
        hparams (Namespace): Hyperparameters of the loaded model.
    """
    def __init__(self, checkpoint_path='~/03_07_2024-23_10_34/epoch=11.ckpt', output_dir='./inference_outputs'):
        """
        Initializes the DocumentRanking instance.

        Args:
            checkpoint_path (str): Path to the model checkpoint. Default is '~/03_07_2024-23_10_34/epoch=11.ckpt'.
            output_dir (str): Directory to save intermediate outputs. Default is './inference_outputs'.
        """
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.output_dir = output_dir
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.documents = []
        self.embeddings = []
        self.similarity_matrix = None
        self.model, self.hparams = load_model(self.checkpoint_path)

    def add_documents(self, docs):
        """
        Adds new documents or a directory of documents, generates embeddings, and updates the similarity matrix.

        Args:
            docs (str): Path to a document or a directory containing documents.
        """
        if os.path.isdir(docs):
            new_docs = [os.path.join(docs, f) for f in os.listdir(docs) if f.endswith('.txt')]
        else:
            new_docs = [docs]
        
        for doc in new_docs:
            with open(doc, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                self.documents.append((os.path.basename(doc), content))
        
        # Generate embeddings and update similarity matrix
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer, output_dir=self.output_dir)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)
        
        # Save intermediate results
        save_intermediate_results(self.embeddings, 'embeddings.pt', self.output_dir)
        save_intermediate_results(self.similarity_matrix, 'similarity_matrix.pt', self.output_dir)

    def rank_documents(self, source=None):
        """
        Ranks documents based on their similarity to a source document.

        Args:
            source (str or int, optional): Source document by name or index. If None, a random document is selected. Default is None.

        Returns:
            list: List of tuples containing the top 100 most relevant documents and their similarity scores.
        """
        # Determine the source document index
        if source is None:
            source_idx = random.randint(0, len(self.documents) - 1)
        elif isinstance(source, int):
            source_idx = source
        else:
            source_idx = next(i for i, doc in enumerate(self.documents) if doc[0] == source)
        
        # Compute similarity scores and rank documents
        scores = self.similarity_matrix[source_idx].cpu().numpy()
        ranked_indices = np.argsort(scores)[::-1]
        ranked_documents = [(self.documents[i], scores[i]) for i in ranked_indices[:100]]
        return ranked_documents

    def update_checkpoint_path(self, path):
        """
        Updates the checkpoint path for the model.

        Args:
            path (str): New path to the model checkpoint.
        """
        self.checkpoint_path = os.path.expanduser(path)
        self.model, self.hparams = load_model(self.checkpoint_path)

    def update_document(self, doc_name, new_document):
        """
        Updates a document by name or index with new content.

        Args:
            doc_name (str or int): Name or index of the document to update.
            new_document (str): New content for the document.
        """
        # Determine the document index
        if isinstance(doc_name, int):
            index = doc_name
        else:
            index = next(i for i, doc in enumerate(self.documents) if doc[0] == doc_name)
        
        # Update document content
        self.documents[index] = (self.documents[index][0], new_document)
        
        # Generate embeddings and update similarity matrix
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer, output_dir=self.output_dir)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)
        
        # Save intermediate results
        save_intermediate_results(self.embeddings, 'embeddings.pt', self.output_dir)
        save_intermediate_results(self.similarity_matrix, 'similarity_matrix.pt', self.output_dir)

    def delete_document(self, doc_name):
        """
        Deletes a document by name or index.

        Args:
            doc_name (str or int): Name or index of the document to delete.
        """
        # Determine the document index
        if isinstance(doc_name, int):
            index = doc_name
        else:
            index = next(i for i, doc in enumerate(self.documents) if doc[0] == doc_name)
        
        # Delete the document
        del self.documents[index]
        
        # Generate embeddings and update similarity matrix
        self.embeddings = generate_embeddings(self.documents, self.model, self.tokenizer, output_dir=self.output_dir)
        self.similarity_matrix = compute_similarity_matrix(self.embeddings)
        
        # Save intermediate results
        save_intermediate_results(self.embeddings, 'embeddings.pt', self.output_dir)
        save_intermediate_results(self.similarity_matrix, 'similarity_matrix.pt', self.output_dir)
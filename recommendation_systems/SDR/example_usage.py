from DocumentRanking import DocumentRanking

if __name__ == "__main__":
    doc_ranker = DocumentRanking(checkpoint_path='~/03_07_2024-23_10_34/epoch=11.ckpt') #initialize with checkpoint path
    doc_ranker.add_documents('./data/text_files') #add documents in text_files to matrix
    ranked_docs = doc_ranker.rank_documents() #since we are not specifying an input (document name), the algorithm will randomly choose a document as source
    print("Top 100 relevant documents and their scores:")
    for doc, score in ranked_docs:
        print(f"Document: {doc[0]}, Score: {score}")

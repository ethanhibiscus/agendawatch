from DocumentRanking import DocumentRanking

if __name__ == "__main__":
    doc_ranker = DocumentRanking()
    ranked_docs = doc_ranker.rank_documents()
    print("Top 100 relevant documents and their scores:")
    for doc, score in ranked_docs:
        print(f"Document: {doc[0]}, Score: {score}")

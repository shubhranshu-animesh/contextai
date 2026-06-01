import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.config import settings

class HybridRetriever:
    def __init__(self, raw_documents: list[str] = None):
        """Initializes both the Dense and Sparse retrieval spaces."""
        
        # 1. Initialize Dense Retriever (ChromaDB)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=self.embeddings
        )

        # 2. Initialize Sparse Retriever (Rank-BM25)
        self.documents = raw_documents if raw_documents else []
        self.bm25 = None
        if self.documents:
            self._build_bm25_index(self.documents)

    def _build_bm25_index(self, docs: list[str]):
        """Tokenizes documents and builds the probabilistic BM25 index."""
        # Simple whitespace tokenization for the sparse index
        tokenized_corpus = [doc.lower().split(" ") for doc in docs]
        
        # Applying the mathematical tuning parameters from config
        self.bm25 = BM25Okapi(
            tokenized_corpus, 
            k1=settings.BM25_K1, 
            b=settings.BM25_B
        )

    def dense_search(self, query: str, k: int = 5):
        """Executes cosine similarity search in the continuous vector space."""
        # Returns chunks and their distance scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def sparse_search(self, query: str, k: int = 5):
        """Executes lexical search using BM25 probabilistic saturation."""
        if not self.bm25:
            raise ValueError("BM25 index not initialized with documents.")
        
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and return the top k scoring documents
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], scores[i]) for i in top_n_indices]

# Quick sanity check block
if __name__ == "__main__":
    print("Retrieval Engine initialized.")
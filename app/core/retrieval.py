import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.config import settings

class HybridRetriever:
    def __init__(self, raw_documents: list[str] = None):
        """Initializes both the Dense and Sparse retrieval spaces."""
        
        # 1. Initialize Dense Retriever (Pinecone Cloud)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is missing!")
            
        self.index_name = "contextai"
        
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key
        )

        # 2. Initialize Sparse Retriever (Rank-BM25)
        # BM25 still runs in-memory using the active document
        self.documents = raw_documents if raw_documents else []
        self.bm25 = None
        if self.documents:
            self._build_bm25_index(self.documents)

    def _build_bm25_index(self, docs: list[str]):
        """Tokenizes documents and builds the probabilistic BM25 index."""
        tokenized_corpus = [doc.lower().split(" ") for doc in docs]
        self.bm25 = BM25Okapi(
            tokenized_corpus, 
            k1=settings.BM25_K1, 
            b=settings.BM25_B
        )

    def dense_search(self, query: str, k: int = 5):
        """Executes cosine similarity search in the cloud vector space."""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def sparse_search(self, query: str, k: int = 5):
        """Executes lexical search using BM25 probabilistic saturation."""
        if not self.bm25:
            raise ValueError("BM25 index not initialized with documents.")
        
        tokenized_query = query.lower().split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.documents[i], scores[i]) for i in top_n_indices]

if __name__ == "__main__":
    print("Cloud Retrieval Engine initialized.")
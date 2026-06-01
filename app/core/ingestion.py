from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings
import os

class IngestionPipeline:
    def __init__(self):
        # 1. Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 3. CRITICAL FIX: Always initialize the database connection on boot
        self.vector_store = Chroma(
            persist_directory="./data/chroma_db",
            embedding_function=self.embeddings
        )

    # ─── PASTE THE HELPER FUNCTION HERE ────────────────────────────────
    def get_vector_store_instance(self):
        """Exposes the internal persistent Chroma client to external modules."""
        return self.vector_store
    # ──────────────────────────────────────────────────────────────────

    def process_document(self, file_path: str) -> list:
        """Parses a PDF file and slices it into continuous chunk matrices."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document missing at: {file_path}")
            
        loader = PyMuPDFLoader(file_path)
        raw_documents = loader.load()
        
        # Deconstruct pages into uniform chunks
        chunked_docs = self.text_splitter.split_documents(raw_documents)
        return chunked_docs

    def populate_vector_store(self, chunks):
        """Adds new document chunks to the existing database."""
        self.vector_store.add_documents(chunks)

# Self-contained testing block
if __name__ == "__main__":
    print("Testing Ingestion Subsystem...")
    # This pipeline is ready to ingest documents. 
    # Next, we will connect this to the companion Sparse Engine (BM25) to complete the dual-indexing system.
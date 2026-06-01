from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from app.config import settings
import os

class IngestionPipeline:
    def __init__(self):
        # 1. Initialize the embedding model (Runs locally on Hugging Face CPU)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # 3. Connect to Pinecone Cloud Database
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is missing from Hugging Face!")
            
        self.index_name = "contextai" 
        
        # --- EXPLICIT CONNECTION FIX ---
        # Explicitly spin up the official Pinecone client connection first
        pc = Pinecone(api_key=pinecone_api_key)
        active_index = pc.Index(self.index_name)
        
        # Pass the active database connection to LangChain
        self.vector_store = PineconeVectorStore(
            index=active_index,
            embedding=self.embeddings
        )

    def get_vector_store_instance(self):
        """Exposes the internal persistent Pinecone client to external modules."""
        return self.vector_store

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
        """Uploads the mathematical embeddings directly to Pinecone's servers."""
        self.vector_store.add_documents(chunks)

if __name__ == "__main__":
    print("Testing Cloud Ingestion Subsystem...")
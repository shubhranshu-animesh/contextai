import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Force the OS to load the .env file BEFORE evaluating any class variables
load_dotenv() 

class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Paths & Models
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # Mathematical Constants for BM25 Sparse Search
    BM25_K1: float = 1.5 
    BM25_B: float = 0.75 

    # RRF (Reciprocal Rank Fusion) Blending Weights
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5

settings = Settings()
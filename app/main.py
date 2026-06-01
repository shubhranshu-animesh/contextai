import os
from app.core.ingestion import IngestionPipeline
from app.core.retrieval import HybridRetriever
from app.core.reranker import RerankerEngine
from app.core.generation import GenerationEngine

def run_contextai_pipeline(pdf_path: str, query: str):
    """
    The Master Orchestrator for the CONTEXTai RAG backend.
    This function coordinates the sequential flow of data across the AI subsystems.
    """
    print(f"--- Starting CONTEXTai Pipeline ---")
    print(f"Target Document: {pdf_path}")
    print(f"User Query: '{query}'\n")

    # 1. INGESTION PHASE: Deconstruct the PDF into mathematical vectors
    print("[1/5] Ingesting and Chunking Document...")
    ingestion = IngestionPipeline()
    raw_chunks = ingestion.process_document(pdf_path)
    
    print(f"[2/5] Building Continuous Vector Space (Dense)...")
    ingestion.populate_vector_store(raw_chunks)
    
    # Extract string content to feed the sparse index
    chunk_strings = [chunk.page_content for chunk in raw_chunks]

    # 2. RETRIEVAL PHASE: Map the query into both spaces
    print("[3/5] Executing Dual-Space Retrieval...")
    retriever = HybridRetriever(raw_documents=chunk_strings)
    
    # Retrieve top 5 candidates from semantic space
    dense_results = retriever.dense_search(query, k=5)
    # Retrieve top 5 candidates from probabilistic lexical space
    sparse_results = retriever.sparse_search(query, k=5)

    # 3. RERANKING PHASE: Unify distributions via Reciprocal Rank Fusion
    print("[4/5] Applying Mathematical Reranking (RRF)...")
    reranker = RerankerEngine(rrf_k=60)
    blended_context = reranker.reciprocal_rank_fusion(dense_results, sparse_results, top_n=3)

    # 4. GENERATION PHASE: LLM Synthesis via deterministic LPU inference
    print("[5/5] Synthesizing Final Output via Groq LPU...\n")
    generator = GenerationEngine()
    final_answer = generator.generate_answer(query, blended_context)

    print("=========================================")
    print("🤖 CONTEXTai RESPONSE:")
    print("=========================================")
    print(final_answer)
    print("=========================================")

if __name__ == "__main__":
    # Point this to your actual resume PDF
    # Make sure to put a copy of your resume in the 'data' folder!
    SAMPLE_PDF = "./data/resume.pdf" 
    
    # A highly specific query that requires both semantic understanding and exact keyword matching
    SAMPLE_QUERY = "What was the exact Sharpe Ratio of Shubhranshu's portfolio project, and what ML models were used?"
    
    if os.path.exists(SAMPLE_PDF):
        run_contextai_pipeline(SAMPLE_PDF, SAMPLE_QUERY)
    else:
        print(f"ERROR: Please place a test PDF at '{SAMPLE_PDF}' to run the pipeline.")
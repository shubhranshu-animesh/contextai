from app.config import settings

class RerankerEngine:
    def __init__(self, rrf_k: int = 60):
        """
        Initializes the reranker using Reciprocal Rank Fusion (RRF).
        rrf_k is a constant penalty factor that ensures low-ranked items 
        do not disproportionately impact the final combined score.
        """
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(self, dense_results: list, sparse_results: list, top_n: int = 5) -> list[dict]:
        """
        Unifies disparate score spaces by evaluating the reciprocal relative ranks
        of retrieved document chunks.
        """
        rrf_scores = {}

        # 1. Process Dense Rankings
        # dense_results is expected as a list of tuples: (Document, score)
        for rank, (doc, _) in enumerate(dense_results):
            doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            # Store metadata along with content for frontend handoffs
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            if doc_content not in rrf_scores:
                rrf_scores[doc_content] = {"score": 0.0, "metadata": metadata}
            
            # Apply RRF formula for Dense list position
            rrf_scores[doc_content]["score"] += settings.DENSE_WEIGHT * (1.0 / (self.rrf_k + (rank + 1)))

        # 2. Process Sparse Rankings
        # sparse_results is expected as a list of tuples: (doc_string, score)
        for rank, (doc_content, _) in enumerate(sparse_results):
            if doc_content not in rrf_scores:
                rrf_scores[doc_content] = {"score": 0.0, "metadata": {}}
                
            # Apply RRF formula for Sparse list position
            rrf_scores[doc_content]["score"] += settings.SPARSE_WEIGHT * (1.0 / (self.rrf_k + (rank + 1)))

        # 3. Sort candidates by their consolidated RRF scores
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1]["score"], reverse=True)
        
        # Format the output for API delivery
        final_results = [
            {"content": content, "score": data["score"], "metadata": data["metadata"]}
            for content, data in sorted_docs[:top_n]
        ]
        
        return final_results

if __name__ == "__main__":
    print("Reranker Engine initialized with RRF protocol.")
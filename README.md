---
title: CONTEXTai
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# 🧠 CONTEXTai: Adaptive Hybrid-RAG Document Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg?logo=fastapi)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-black.svg)
![Groq](https://img.shields.io/badge/LLM-Groq-f56042.svg)
![Docker](https://img.shields.io/badge/Deployed_on-Hugging_Face-yellow.svg?logo=docker)

CONTEXTai is a production-grade, containerized Document Intelligence application engineered on a **Dual-Space Hybrid Retrieval-Augmented Generation (RAG)** architecture. Designed to eliminate semantic blind spots, it unifies the abstract contextual understanding of deep learning models with the mathematically precise lexical matching required for enterprise workloads.

---

## 📖 Table of Contents

- [Theoretical Background](#-theoretical-background)
- [Design Principles & Architecture](#-design-principles--architecture)
- [Codebase & Implementation Details](#-codebase--implementation-details)
- [Deployment & Local Setup](#-deployment--local-setup)

---

## 🔬 Theoretical Background

Standard RAG systems rely exclusively on dense vector spaces via semantic search embeddings. While highly effective at identifying thematic patterns, semantic-only structures struggle with token-exact retrievals—such as product SKUs, specialized alphanumeric jargon, system metrics, or proper nouns. 

CONTEXTai orchestrates an advanced **Dual-Space Hybrid Search** methodology to solve this trade-off:

1. **Dense Vector Space (Semantic Retrieval):** Raw text blocks are transformed into continuous mathematical vector spaces utilizing the `all-MiniLM-L6-v2` model. This Maps structural meaning to a 384-dimensional space, relying on Cosine Similarity to capture intent.
2. **Sparse Lexical Space (Probabilistic Retrieval):** Concurrently, a secondary pipeline relies on the algorithmic **BM25 (Best Matching 25)** mechanism. BM25 functions as a non-linear, bag-of-words ranking model that evaluates query intersections across term documents, penalizing hyper-frequent terms while boosting unique keyword identities through a tuned IDF (Inverse Document Frequency) factor.

### Reciprocal Rank Fusion (RRF)

Blending continuous cosine similarities (bounded between $0$ and $1$) with unbounded BM25 scores presents a significant mathematical normalization challenge. CONTEXTai resolves this by integrating **Reciprocal Rank Fusion (RRF)**. Rather than relying on raw similarity outputs, RRF calculates absolute weights using reciprocal positions inside dense and sparse lists:

$$RRF\_Score = \frac{1}{k + rank_{dense}} + \frac{1}{k + rank_{sparse}}$$

Where $k$ represents a smoothing constant (configured to $60$). This normalizes cross-database variances, feeding an optimized token context directly to the inference target.

---

## 🏗️ Design Principles & Architecture

The system is configured for serverless elasticity, complete statelessness, and microservice decoupling, matching architectural requirements for reliable cloud distributions.

* **Stateless Topography:** The container maintains zero persistent physical disk space across runs. State boundaries map instantly to cloud vector clusters, ensuring horizonal scaling properties.
* **Asynchronous Foundations:** Engineered with FastAPI around an `asyncio` loop, long-running vector updates and chunk generation queries operate non-blockingly, maintaining system availability under load.
* **Separation of Concerns (SoC):** Logical boundaries strictly segregate data streams into ingestion engines, independent retrievers, a ranking arbiter, and contextual generators.

---

## 💻 Codebase & Implementation Details

The operations of CONTEXTai are structured within the `app/` execution directory:

### 1. `app/api/server.py`
* Serves as the central API orchestration gateway utilizing **FastAPI**.
* Exposes transactional endpoints: `/api/v1/upload` (for document streams and parsing orchestration) and `/api/v1/query` (the unified generative pipeline).
* Implements dynamic fallback routing: routes queries through vector context parsing paths if an active `pdf_name` reference is provided; otherwise, transitions to direct conversational completions.
* Incorporates an asynchronous, zero-dependency Tailwind CSS user interface utilizing `HTMLResponse` and the Web Streams API for true typewriter-effect token generation.

### 2. `app/core/ingestion.py`
* Executes low-overhead byte parsing of binary PDFs using `PyMuPDFLoader`.
* Configures text sequences through a `RecursiveCharacterTextSplitter` operating at `chunk_size=1000` tokens and a `chunk_overlap=200` token window to eliminate semantic fragmentation at split lines.
* Employs an explicit `Pinecone` serverless architecture vector layout to route embedded data directly to cloud systems (`contextai` database cluster), avoiding memory spikes inside container environments.

### 3. `app/core/retrieval.py`
* Handles the independent retrieval spaces simultaneously.
* **Dense Core:** Establishes runtime connections to remote Pinecone engines using LangChain abstractions processing locally on host hardware.
* **Sparse Core:** Spins up volatile, reactive `BM25Okapi` indices inside container RAM allocations, populating matrices using tokenized document variations on demand.

### 4. `app/core/reranker.py`
* Houses the custom `RerankerEngine` execution layer.
* Collates the top $5$ records returned from both dense and sparse retrieval spaces and recalculates positions using the RRF algorithm.
* Truncates output arrays down to a strict `top_n=3` document delivery context to eliminate LLM input bloating and minimize generation hallucination vectors.

### 5. `app/core/generation.py`
* Interfaces with low-latency hardware loops via the **Groq Cloud API**.
* Structures the retrieved documentation alongside existing conversation memory profiles into clean prompt packages.
* Streams output tokens down to the active stream utilizing asynchronous Python generators (`StreamingResponse`).

---

## 🚀 Deployment & Local Setup

### Hugging Face Spaces (Docker Deploy)

This repository is optimized natively for continuous delivery to **Hugging Face Spaces**.

1. The frontmatter header block signals deployment guidelines directly to the space platform, forcing a Docker build exposing internal port `7860`.
2. Map your system credentials securely under the **Variables and Secrets** panel within your cloud project console:
   * `PINECONE_API_KEY`: Your Pinecone cluster credential token.
   * `GROQ_API_KEY`: Your hyper-speed Groq access token.
3. The platform reads the local `Dockerfile` configuration and builds the image automatically.

### Local Development

To run the pipeline within your local test environments, execute the following commands:

```bash
# 1. Clone the repository locally
git clone [https://github.com/shubhranshu-animesh/contextai.git](https://github.com/shubhranshu-animesh/contextai.git)
cd contextai

# 2. Establish your virtual environment and resolve requirements
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Provision local environment variables
export PINECONE_API_KEY="your-pinecone-key"
export GROQ_API_KEY="your-groq-key"

# 4. Launch the application via Uvicorn development server
uvicorn app.api.server:app --host 0.0.0.0 --port 7860 --reload
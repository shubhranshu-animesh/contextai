from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from app.core.ingestion import IngestionPipeline
from app.core.retrieval import HybridRetriever
from app.core.reranker import RerankerEngine
from app.core.generation import GenerationEngine
import os
import shutil

app = FastAPI(
    title="CONTEXTai API",
    description="Production-grade Dual-Space Hybrid RAG Engine",
    version="1.0.0"
)

# ==========================================
# PYDANTIC DATA SCHEMAS
# ==========================================
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    pdf_name: str | None = Field(default=None, description="The target document file name. None if general chat.")
    history: list[Message] = Field(default=[], description="Previous conversation history")

@app.post("/api/v1/upload", status_code=status.HTTP_201_CREATED)
async def upload_and_index_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Ensure the directory exists before trying to save the file
    os.makedirs("./data", exist_ok=True) 
    
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        print(f"[ENGINE INFO] Indexing new document: {file.filename}")
        ingestion = IngestionPipeline()
        raw_chunks = ingestion.process_document(file_path)
        ingestion.populate_vector_store(raw_chunks)
        return {"status": "success", "filename": file.filename, "message": "File indexed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index document: {str(e)}")

@app.post("/api/v1/query")
async def process_rag_query(payload: QueryRequest):
    try:
        generator = GenerationEngine()

        # BRANCH 1: RAG Mode (User provided a document)
        if payload.pdf_name:
            pdf_path = f"./data/{payload.pdf_name}"
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=404, detail=f"Target file '{payload.pdf_name}' not found. Please upload it first.")

            ingestion = IngestionPipeline()
            
            # Re-parse the active document for the local BM25 index
            # (Dense embeddings are safely stored in Pinecone!)
            raw_chunks = ingestion.process_document(pdf_path)
            chunk_strings = [chunk.page_content for chunk in raw_chunks]

            retriever = HybridRetriever(raw_documents=chunk_strings)
            dense_results = retriever.dense_search(payload.query, k=5)
            sparse_results = retriever.sparse_search(payload.query, k=5)

            reranker = RerankerEngine(rrf_k=60)
            blended_context = reranker.reciprocal_rank_fusion(dense_results, sparse_results, top_n=3)

            return StreamingResponse(
                generator.stream_answer(payload.query, blended_context, payload.history, use_rag=True),
                media_type="text/plain"
            )
        
        # BRANCH 2: General Chat Mode (No document provided)
        else:
            return StreamingResponse(
                generator.stream_answer(payload.query, context="", history=payload.history, use_rag=False),
                media_type="text/plain"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core engine error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "CONTEXTai Core"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CONTEXTai Interface</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col justify-between font-sans">
        
        <header class="border-b border-slate-800 bg-slate-950 p-5 shadow-xl">
            <div class="max-w-4xl mx-auto flex justify-between items-center">
                <h1 class="text-2xl font-bold tracking-tight text-blue-400">🧠 CONTEXT<span class="text-slate-100">ai</span></h1>
                <span class="bg-emerald-500/10 text-emerald-400 text-xs px-3 py-1 rounded-full border border-emerald-500/20 font-mono">Backend Active</span>
            </div>
        </header>

        <main class="flex-grow p-6 max-w-4xl w-full mx-auto flex flex-col justify-start gap-6 h-[80vh]">
            
            <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 shadow-md flex justify-between items-center shrink-0">
                <p class="text-sm text-slate-400">📍 Active Document: <code id="activeDoc" class="text-slate-500 font-mono bg-slate-900 px-2 py-0.5 rounded">None (General Chat Mode)</code></p>
                <div class="flex gap-2">
                    <input type="file" id="fileUpload" accept=".pdf" class="hidden">
                    <button id="uploadTriggerBtn" onclick="document.getElementById('fileUpload').click()" class="text-sm bg-blue-600 hover:bg-blue-500 px-4 py-2 rounded-lg transition-colors text-white font-medium border border-blue-500 shadow-sm flex items-center gap-2">
                        <span>Upload PDF</span>
                    </button>
                </div>
            </div>

            <div id="responseContainer" class="bg-slate-950 border border-slate-800 rounded-xl p-6 flex-grow overflow-y-auto shadow-inner flex flex-col gap-4">
                <p id="placeholderText" class="text-slate-500 text-center my-auto italic">Start chatting globally, or upload a document for RAG analysis...</p>
                <div id="chatLog" class="flex flex-col gap-4 pb-4"></div>
            </div>

            <div class="shrink-0">
                <form id="queryForm" class="flex gap-3">
                    <input type="text" id="queryInput" required placeholder="Ask a question..." 
                           class="w-full bg-slate-950 border border-slate-800 rounded-xl px-5 py-4 focus:outline-none focus:border-blue-500 transition-colors placeholder-slate-600 text-slate-100 shadow-lg">
                    <button type="submit" id="submitBtn" class="bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white font-medium px-8 rounded-xl transition-all shadow-lg flex items-center justify-center min-w-[140px]">
                        <span>Analyze</span>
                    </button>
                </form>
            </div>
        </main>

        <script>
            let currentDocument = null; 
            let conversationHistory = []; 
            
            const fileUpload = document.getElementById('fileUpload');
            const uploadTriggerBtn = document.getElementById('uploadTriggerBtn');
            const activeDoc = document.getElementById('activeDoc');
            const chatLog = document.getElementById('chatLog');
            const placeholderText = document.getElementById('placeholderText');
            const responseContainer = document.getElementById('responseContainer');
            
            // Unified File Selection & Upload Event
            fileUpload.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if(!file) return;

                const formData = new FormData();
                formData.append("file", file);

                // UI Loading State
                uploadTriggerBtn.disabled = true;
                uploadTriggerBtn.innerHTML = `<span class="animate-pulse">Indexing...</span>`;
                uploadTriggerBtn.classList.replace('bg-blue-600', 'bg-amber-600');
                uploadTriggerBtn.classList.replace('border-blue-500', 'border-amber-500');
                
                activeDoc.textContent = file.name + " (Processing...)";
                activeDoc.classList.replace('text-slate-500', 'text-amber-400');
                activeDoc.classList.replace('text-blue-400', 'text-amber-400'); // in case they upload a 2nd doc

                try {
                    const response = await fetch('/api/v1/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    if(response.ok) {
                        currentDocument = data.filename;
                        activeDoc.textContent = currentDocument;
                        activeDoc.classList.replace('text-amber-400', 'text-blue-400');
                        
                        // Reset chat on new document
                        conversationHistory = []; 
                        chatLog.innerHTML = '';
                        placeholderText.classList.remove('hidden');
                        placeholderText.textContent = `Document '${currentDocument}' indexed. Ready for questions.`;
                    } else {
                        activeDoc.textContent = "None (General Chat Mode)";
                        activeDoc.classList.replace('text-amber-400', 'text-slate-500');
                        throw new Error(data.detail);
                    }
                } catch (error) {
                    alert("Upload failed: " + error.message);
                } finally {
                    // Reset Button State
                    uploadTriggerBtn.disabled = false;
                    uploadTriggerBtn.innerHTML = `<span>Change PDF</span>`;
                    uploadTriggerBtn.classList.replace('bg-amber-600', 'bg-blue-600');
                    uploadTriggerBtn.classList.replace('border-amber-500', 'border-blue-500');
                    fileUpload.value = ""; 
                }
            });

            document.getElementById('queryForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const queryInput = document.getElementById('queryInput');
                const submitBtn = document.getElementById('submitBtn');
                
                const queryText = queryInput.value.trim();
                if(!queryText) return;

                placeholderText.classList.add('hidden');
                const userDiv = document.createElement('div');
                userDiv.className = 'self-end bg-blue-600/20 text-blue-100 border border-blue-500/30 px-5 py-3 rounded-2xl rounded-tr-sm max-w-[80%]';
                userDiv.textContent = queryText;
                chatLog.appendChild(userDiv);
                
                const aiDiv = document.createElement('div');
                aiDiv.className = 'self-start bg-slate-800/50 text-slate-200 border border-slate-700 px-5 py-3 rounded-2xl rounded-tl-sm max-w-[80%] whitespace-pre-line leading-relaxed';
                aiDiv.innerHTML = '<span class="animate-pulse text-slate-400 text-sm">Thinking...</span>';
                chatLog.appendChild(aiDiv);
                responseContainer.scrollTop = responseContainer.scrollHeight;

                submitBtn.disabled = true;
                queryInput.value = "";
                let fullAiResponse = "";

                try {
                    const response = await fetch('/api/v1/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            query: queryText, 
                            pdf_name: currentDocument,
                            history: conversationHistory 
                        })
                    });
                    
                    if(!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || "Unknown backend failure");
                    }

                    aiDiv.textContent = ""; 
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder("utf-8");
                    let done = false;

                    while (!done) {
                        const { value, done: readerDone } = await reader.read();
                        done = readerDone;
                        if (value) {
                            const chunk = decoder.decode(value, { stream: true });
                            fullAiResponse += chunk;
                            aiDiv.textContent += chunk;
                            responseContainer.scrollTop = responseContainer.scrollHeight;
                        }
                    }

                    conversationHistory.push({ role: "user", content: queryText });
                    conversationHistory.push({ role: "assistant", content: fullAiResponse });

                } catch (error) {
                    aiDiv.innerHTML = `<span class="text-red-400 font-mono">Error: ${error.message}</span>`;
                } finally {
                    submitBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
        
    return HTMLResponse(content=html_content, status_code=200)
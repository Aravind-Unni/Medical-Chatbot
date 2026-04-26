import uvicorn
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import the newly updated engine
from rag_engine import initialize_medical_rag_pipeline
from database import add_message, get_chat_history

app = FastAPI()

# Ensure the static directory exists before mounting
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

rag_app = None

class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default="default-session")

@app.on_event("startup")
async def startup_event():
    global rag_app
    print("⚙️ Booting up Medical RAG Server...")

    try:
        # This initializes the pipeline (Chroma, BM25, Reranker, and Gemini)
        rag_app = initialize_medical_rag_pipeline()
        print("✅ Server successfully connected to Medical Databases!")
        
    except Exception as e:
        print(f"❌ Critical Error during Pipeline Setup: {e}")
        rag_app = None

@app.get("/")
async def index():
    return FileResponse('static/index.html')

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    global rag_app
    if not rag_app:
        raise HTTPException(
            status_code=500,
            detail="The Medical Knowledge Base failed to initialize. Please check the server logs."
        )
    
    try:
        session_id = request.session_id
        print(f"📝 Received Clinical Query [{session_id}]: {request.query}")
        
        # Retrieve recent history from your database module
        chat_history = get_chat_history(session_id, limit=6)
        
        # Prepare the state for the RAG Pipeline
        initial_state = {
            "original_question": request.query,
            "chat_history": chat_history,
            "session_id": session_id
        }
        
        # Invoke the pipeline
        result = rag_app.invoke(initial_state)
        answer = result.get("generation", "I couldn't generate an answer based on the current context.")
        
        # Log the interaction to the database
        add_message(session_id, "User", request.query)
        add_message(session_id, "Assistant", answer)
        
        # Determine source status for UI display
        used_tools = ["Medical Knowledge Base (Hybrid Search & Reranked)"] if result.get("documents") else ["No medical literature found for this query"]
            
        return {"answer": answer, "sources": used_tools}
    
    except Exception as e:
        print(f"❌ Error during request processing: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

if __name__ == "__main__":
    # Note: 'main:app' assumes this file is named main.py
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
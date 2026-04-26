import os
import pickle
from dotenv import load_dotenv

# Embeddings & Vector Stores
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Sparse Retriever
from langchain_community.retrievers import BM25Retriever

from sentence_transformers import CrossEncoder

# LLM & Prompts (Google GenAI / Gemma)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import NVIDIARerank
# Langfuse
from langfuse.langchain import CallbackHandler
from langfuse import get_client

# --- IMPORT YOUR PREPROCESSOR ---
from rag_preprocess import preprocess_and_index

load_dotenv()

# --- CONSTANTS & PATHS ---
COLLECTION_NAME = "medical_knowledge_hybrid"
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
RERANKER_MODEL_NAME = "nvidia/llama-nemotron-rerank-1b-v2"

BASE_DIR = r"C:/RAG_Medical"
TARGET_PDF_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
BM25_SAVE_PATH = os.path.join(BASE_DIR, "bm25_retriever.pkl")


class MedicalRAGPipeline:
    def __init__(self, chroma_retriever, bm25_retriever, cross_encoder, prompt, llm):
        self.chroma_retriever = chroma_retriever
        self.bm25_retriever = bm25_retriever
        self.cross_encoder = cross_encoder 
        self.prompt = prompt
        self.llm = llm
        self.langfuse_client = get_client()

    def retrieve_and_rerank(self, query, top_k=5):
        print(f"🔍 Searching Medical Knowledge Base for: '{query}'")
        dense_docs = self.chroma_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)
        
        # Deduplicate the results
        unique_docs = {}
        for doc in dense_docs + sparse_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
                
        doc_list = list(unique_docs.values())
        if not doc_list:
            return ""
            
        print(f"⚖️ Reranking {len(doc_list)} medical snippets...")
        
        # --- THE FIX: Use NVIDIA's built-in LangChain compressor ---
        reranked_docs = self.cross_encoder.compress_documents(
            query=query, 
            documents=doc_list
        )
        
        # Extract the results (NVIDIA already sorted them and limited to top_n)
        top_docs = [f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(reranked_docs[:top_k])]
        
        return "\n\n---\n\n".join(top_docs)

    def invoke(self, state: dict):
        original_question = state.get("original_question", "")
        chat_history = state.get("chat_history", [])
        session_id = state.get("session_id", "anonymous_session")
        
        langfuse_handler = CallbackHandler()
        langfuse_config = {
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_tags": ["medical_rag_query"]
            }
        }
        
        if chat_history:
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
            rewrite_sys = """Given the following patient conversation history and a follow-up medical question, rephrase the follow-up question to be a standalone query.
            Do NOT answer the question, just reformulate it so it contains all the necessary medical context. If it's already standalone, return it exactly as is."""
            rewrite_prompt = PromptTemplate.from_template(f"{rewrite_sys}\n\nChat History:\n{{history}}\n\nFollow-Up: {{question}}\nStandalone Search Query:")
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            
            search_query = rewrite_chain.invoke(
                {"history": history_str, "question": original_question},
                config=langfuse_config
            )
            print(f"🔄 Rewrote Medical Query to: {search_query}")
        else:
            search_query = original_question
            history_str = "No previous history."
            
        context_str = self.retrieve_and_rerank(search_query)
        
        print("🤖 Generating Clinical Response...")
        qa_chain = self.prompt | self.llm | StrOutputParser()
        answer = qa_chain.invoke(
            {"context": context_str, "chat_history": history_str, "question": original_question},
            config=langfuse_config
        )
        
        self.langfuse_client.flush()
        return {"generation": answer, "documents": True if context_str else False}


def initialize_medical_rag_pipeline():
    print("🚀 Initializing Medical Pipeline...")
    
    # --- THE AUTO-FALLBACK LOGIC ---
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.path.exists(BM25_SAVE_PATH):
        print("⚠️ Databases not found! Triggering automatic preprocessing...")
        if not os.path.exists(TARGET_PDF_DIR):
            raise FileNotFoundError(f"❌ Cannot build databases. PDF folder missing at: {TARGET_PDF_DIR}")
        
        # Call the function from your preprocessor script!
        preprocess_and_index(TARGET_PDF_DIR)
        print("🔄 Preprocessing complete. Resuming engine startup...")

    print("🧠 Loading HuggingFace Embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("💾 Loading ChromaDB from disk...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR
    )
    chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    print("💾 Loading BM25 Retriever from disk...")
    with open(BM25_SAVE_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)

    # 4. Initialize NVIDIA Reranker
    print(f"⚖️ Initializing NVIDIA Reranker: {RERANKER_MODEL_NAME}...")
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    if not nvidia_api_key:
        raise ValueError("❌ NVIDIA_API_KEY not found in your .env file!")
        
    reranker = NVIDIARerank(
        model=RERANKER_MODEL_NAME,
        top_n=5,
        truncate="END" # Automatically handles context limits cleanly
    )

    print("🤖 Initializing Gemma 3 Generator...")
    llm = ChatGroq(
        model="llama-3.3-70b-Versatile",
        temperature=0.1
    )

    system_prompt = """You are an expert, objective AI medical assistant.
    Your role is to extract and summarize information strictly from the provided medical literature/documents.
    
    CRITICAL MEDICAL GUARDRAILS:
    1. STRICT ADHERENCE: Base your answers ONLY on the provided context. Do NOT use outside knowledge, guess, or infer clinical guidelines that are not explicitly stated.
    2. NO DIAGNOSES: Never attempt to diagnose a patient, prescribe medication, or offer definitive treatment plans.
    3. MISSING INFORMATION: If the answer is not contained in the provided text, you must explicitly state: "The provided medical documents do not contain information to answer this query."

    Previous Conversation History:
    {chat_history}

    Medical Context:
    {context}

    Patient/User Query: {question}
    Clinical Response:"""
    
    prompt = PromptTemplate.from_template(system_prompt)
    
    print("✅ Advanced Medical RAG Pipeline Ready.")
    return MedicalRAGPipeline(chroma_retriever, bm25_retriever, reranker, prompt, llm)
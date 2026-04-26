import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def add_message(session_id: str, role: str, content: str):
    """
    Saves a single message to the Supabase database.
    Role must be either 'User' or 'Assistant'.
    """
    try:
        data = {
            "session_id": session_id,
            "role": role,
            "content": content
        }
        # Insert the message into the chat_history table
        supabase.table("chat_history").insert(data).execute()
    except Exception as e:
        print(f"❌ Failed to save message to database: {str(e)}")

def get_chat_history(session_id: str, limit: int = 6) -> list:
    """
    Retrieves the most recent chat history for a specific session.
    Returns a list of dictionaries formatted for the LangChain prompt.
    Defaults to the last 6 messages (3 turns).
    """
    try:
        # Query Supabase: match session_id, order by oldest first, but limit the query
        # To get the *latest* N messages chronologically, we order by created_at DESC, 
        # limit N, then reverse the list in Python.
        response = supabase.table("chat_history") \
            .select("role, content") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        records = response.data
        
        # Reverse the records so the oldest of the recent messages comes first
        records.reverse()
        
        # Ensure the output matches the exact format our RAG pipeline expects
        formatted_history = [
            {"role": record["role"], "content": record["content"]}
            for record in records
        ]
        
        return formatted_history

    except Exception as e:
        print(f"❌ Failed to retrieve chat history: {str(e)}")
        return []
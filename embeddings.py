from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import Entry, Embedding
from typing import List, Dict, Any

# Load a smaller model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # 61MB vs 90MB for all-MiniLM-L6-v2

def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text"""
    return model.encode(text).tolist()

def search_entries(db: Session, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search using vector similarity"""
    query_embedding = get_embedding(query)
    
    # SQL query using pgvector's <-> operator (cosine distance)
    result = db.execute(
        text("""
        SELECT 
            e.id, 
            e.url, 
            e.content, 
            e.thoughts, 
            e.created_at,
            1 - (emb.embedding <-> :query_embedding) AS similarity
        FROM 
            entries e
        JOIN 
            embeddings emb ON e.id = emb.entry_id
        ORDER BY 
            similarity DESC
        LIMIT :limit
        """),
        {
            "query_embedding": query_embedding,
            "limit": limit
        }
    )
    
    entries = []
    for row in result:
        entries.append({
            "id": row.id,
            "url": row.url,
            "content": row.content,
            "thoughts": row.thoughts,
            "created_at": row.created_at,
            "similarity": float(row.similarity)
        })
    
    return entries 
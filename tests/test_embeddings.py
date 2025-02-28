import pytest
from embeddings import get_embedding

def test_get_embedding():
    """Test that embedding generation works"""
    text = "This is a test text"
    embedding = get_embedding(text)
    
    # Check embedding is the right shape
    assert len(embedding) == 384  # Embedding dimension for all-MiniLM-L6-v2
    assert all(isinstance(val, float) for val in embedding) 
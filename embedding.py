from sentence_transformers import SentenceTransformer

def generate_embedding(text):
    try:
        # Old code might be using 'embedding' parameter
        # model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        # embedding = model.encode(text, embedding=...)
        
        # New API likely expects:
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        embedding = model.encode(text)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None 
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Loads the SentenceTransformer model."""
    return SentenceTransformer(model_name)

def encode_text(model, text_list):
    """Encodes a list of text into embeddings."""
    return model.encode(text_list, convert_to_tensor=False)
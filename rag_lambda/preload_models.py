import os
from fastembed import TextEmbedding, SparseTextEmbedding

def preload():
    cache_dir = "/var/task/fastembed_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["FASTEMBED_CACHE_DIR"] = cache_dir
    
    print("Downloading Dense Model (BAAI/bge-large-en-v1.5)...")
    TextEmbedding(model_name="BAAI/bge-large-en-v1.5", cache_dir=cache_dir)
    
    print("Downloading Sparse Model (Qdrant/bm25)...")
    SparseTextEmbedding(model_name="Qdrant/bm25", cache_dir=cache_dir)
    
    print("Models preloaded successfully!")

if __name__ == "__main__":
    preload()

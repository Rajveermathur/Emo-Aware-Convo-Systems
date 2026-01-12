# Dependency imports
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

# Function to L2 normalize vectors
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def semantic_vectors_section(client, chat_data):
    # Create or get collection
    collection = client.get_or_create_collection("semantic_vectors", metadata={"hnsw:space": "cosine"})
    print("✅ Connected to persistent Semantic Vectors collection.")

    # Prepare documents and IDs
    docs = []
    ids = []
    meta = []
    # Process each chat chunk
    for chunk in chat_data:
        docs.append(chunk["text"])
        ids.append(chunk["id"])
        structured_log = {
            "timestamp": chunk["timestamp"],
            "ai_response": chunk["ai_response"],
            "emotion_analysis": json.dumps(chunk["emotions_metadata"].get("emotion_scores", {})),
            "emotional_embedding": json.dumps(chunk["emotions_metadata"].get("embedding", []))
        }
        meta.append(structured_log)

    # Create embeddings
    embeddings = embedding_model.encode(docs, convert_to_numpy=True)
    # Normalize embeddings
    embeddings = l2_normalize(embeddings)
    # Add documents to collection
    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=meta,
        ids=ids
    )
    print(f"✅ Added {len(docs)} documents to the Chroma Semantic Vectors collection.")

# -------------------------------------------------------------------------------------------------
def emotion_vectors_section(client, chat_data):
    # Create or get collection
    collection = client.get_or_create_collection("emotions_vectors", metadata={"hnsw:space": "l2"})
    print("✅ Connected to persistent Emotions Vectors collection.")

    # Prepare documents and IDs
    emotion_embeddings = []
    docs = []
    ids = []
    meta = []

    # Process each chat chunk
    for chunk in chat_data:
        emotion_embeddings.append(chunk["emotions_metadata"].get("embedding"))
        docs.append(chunk["text"])
        ids.append(chunk["id"])
        structured_log = {
            "timestamp": chunk["timestamp"],
            "ai_response": chunk["ai_response"],
            "emotion_analysis": json.dumps(chunk["emotions_metadata"].get("emotion_scores", {})),
            "emotional_embedding": json.dumps(chunk["emotions_metadata"].get("embedding", []))
        }
        meta.append(structured_log)
    # print(emotion_embeddings)

    # # Normalize embeddings
    embeddings = l2_normalize(np.array(emotion_embeddings))

    # Add documents to collection
    collection.add(
        documents=docs,
        embeddings=embeddings.tolist(),
        metadatas=meta,
        ids=ids
    )
    print(f"✅ Added {len(docs)} documents to the Chroma Emotions collection.")


if __name__ == "__main__":
    # Load embedding model - 768 Dimensions 
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    
    # Directory to persist Chroma database
    persist_dir = "./chroma_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    # Load chat chunks from JSON file
    with open("data/chunks_ready.json", "r") as f:
        chat_data = json.load(f)
    print(f"Loaded {len(chat_data)} chat chunks.")

    # Process Semantic Vectors
    semantic_vectors_section(client, chat_data)

    # Process Emotion Vectors
    emotion_vectors_section(client, chat_data)

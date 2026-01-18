# Dependency imports
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from datetime import datetime
import json
from preprocessing import emotional_embedder  # Assuming emotional_embedder is defined in preprocessing.py

# Function to L2 normalize vectors
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def retrieve_semantic_vectors(client, database_name, embedding_model, query_text: str, top_k: int = 5, filter_ids: set = None):
# Create or get collection
    collection = client.get_or_create_collection(database_name, metadata={"hnsw:space": "cosine"})
    print("✅ Connected to persistent Chroma collection.")
    
    query_embedding = embedding_model.encode(
        [query_text], convert_to_numpy=True
    )

    query_embedding = l2_normalize(query_embedding)

    where_filter = None
    if filter_ids:
        where_filter = {
            "doc_id": {"$in": list(filter_ids)}
        }
        print("Applying filter IDs:", where_filter)

    data = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    output = {
        "query": query_text,
        "type": "semantic_retrieval",
        "retrieval_timestamp": datetime.now().isoformat(),
        "results": []
    }

    for i, doc in enumerate(data["documents"][0]):
        meta = data["metadatas"][0][i]
        output["results"].append({
            "id": data["ids"][0][i],
            "distance": data["distances"][0][i],
            "document": doc,
            "emotions": json.loads(meta["emotion_analysis"]),
            "ai_response": meta["ai_response"],
            "timestamp": meta["timestamp"]
        })
        
    return(output)

# Function to retrieve emotion vectors
def retrieve_emotion_vectors(client, database_name, query_text: str, top_k: int = 5, filter_ids: set = None):
    # Create or get collection
    collection = client.get_or_create_collection(database_name, metadata={"hnsw:space": "l2"})
    print("✅ Connected to persistent Chroma collection.")

    result = emotional_embedder(query_text)

    query_embedding = np.array(result["embedding"]).reshape(1, -1)

    where_filter = None
    if filter_ids:
        where_filter = {
            "doc_id": {"$in": list(filter_ids)}
        }
        print("Applying filter IDs:", where_filter)
        
    data = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    print(data)
    output = {
        "query": query_text,
        "type": "emotion_retrieval",
        "retrieval_timestamp": datetime.now().isoformat(),
        "results": []
    }

    for i, doc in enumerate(data["documents"][0]):
        meta = data["metadatas"][0][i]
        output["results"].append({
            "id": data["ids"][0][i],
            "distance": data["distances"][0][i],
            "document": doc,
            "emotions": json.loads(meta["emotion_analysis"]),
            "ai_response": meta["ai_response"],
            "timestamp": meta["timestamp"]
        })
        
    return(output)

# Function to save logs for each retrieval
def logs_save(type: str = "retrieval", output: dict = None):
    os.makedirs("logs", exist_ok=True)
    filename = f"{type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join("logs", filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {file_path}")

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Directory to persist Chroma database
    persist_dir = "./chroma_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    query_text = "I am sleepless"

    semantic_output = retrieve_semantic_vectors(client, "semantic_vectors", embedding_model, query_text)
    print(semantic_output)
    logs_save(type="semantic_retrieval", output=semantic_output)

    emotion_output = retrieve_emotion_vectors(client, "emotions_vectors", query_text)
    print(emotion_output)
    logs_save(type="emotion_retrieval", output=emotion_output)
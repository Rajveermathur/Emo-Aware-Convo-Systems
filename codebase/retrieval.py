# Dependency imports
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from datetime import datetime
import json

embedding_model = SentenceTransformer("all-mpnet-base-v2")

# Function to L2 normalize vectors
def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

# Directory to persist Chroma database
persist_dir = "./chroma_store"
os.makedirs(persist_dir, exist_ok=True)

# Persistent client
client = chromadb.PersistentClient(path=persist_dir)


# Create or get collection
collection = client.get_or_create_collection("semantic_vectors", metadata={"hnsw:space": "cosine"})
print("✅ Connected to persistent Chroma collection.")

query_text = "I am sleepless"

query_embedding = embedding_model.encode(
    [query_text], convert_to_numpy=True
)

query_embedding = l2_normalize(query_embedding)

data = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5,
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
    
print(json.dumps(output, indent=2))

filename = f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
file_path = os.path.join("logs", filename)

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Saved results to {file_path}")
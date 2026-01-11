# Dependency imports
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

# Load embedding model - 768 Dimensions 
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

# Load chat chunks from JSON file
with open("data/chunks_ready.json", "r") as f:
    chat_data = json.load(f)
print(f"Loaded {len(chat_data)} chat chunks.")

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
print(f"✅ Added {len(docs)} documents to the Chroma collection.")


# query_text = "sleepless"

# query_embedding = embedding_model.encode(
#     [query_text], convert_to_numpy=True
# )

# query_embedding = l2_normalize(query_embedding)

# results = collection.query(
#     query_embeddings=query_embedding.tolist(),
#     n_results=2
# )

# print(results)

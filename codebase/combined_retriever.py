import os
import chromadb
from simple_retrieval import retrieve_semantic_vectors, retrieve_emotion_vectors, logs_save
from sentence_transformers import SentenceTransformer

def l2_to_similarity(l2_distance: float) -> float:
    return 1.0 / (1.0 + l2_distance)

def cosine_to_similarity(cosine_distance: float) -> float:
    return 1.0 - cosine_distance

def multiplicative_retriever(client, semantic_db_name, emotion_db_name, embedding_model, query_text: str):
    semantic_output = retrieve_semantic_vectors(client, semantic_db_name, embedding_model, query_text)
    emotion_output = retrieve_emotion_vectors(client, emotion_db_name, query_text)

    combined_output = {
        "query": query_text,
        "type": "multiplicative_retrieval",
        "retrieval_timestamp": semantic_output["retrieval_timestamp"],
        "semantic_results": semantic_output["results"],
        "emotion_results": emotion_output["results"]
    }

    return combined_output

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Directory to persist Chroma database
    persist_dir = "./chroma_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    query_text = "I am sleepless"

    # semantic_output = retrieve_semantic_vectors(client, "semantic_vectors", embedding_model, query_text)
    # print(semantic_output)
    # logs_save(type="semantic_retrieval", output=semantic_output)

    # emotion_output = retrieve_emotion_vectors(client, "emotions_vectors", query_text)
    # print(emotion_output)
    # logs_save(type="emotion_retrieval", output=emotion_output)

    combined_output = multiplicative_retriever(client, "semantic_vectors", "emotions_vectors", embedding_model, query_text)
    print(combined_output)
    logs_save(type="combined_retrieval", output=combined_output)
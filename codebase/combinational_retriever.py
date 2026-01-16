import os
import chromadb
from simple_retrieval import retrieve_semantic_vectors, retrieve_emotion_vectors, logs_save
from sentence_transformers import SentenceTransformer

# Convert Euclidean distances to similarity scores in [0, 1] range 
def l2_to_similarity(l2_distance: float) -> float:
    return 1.0 / (1.0 + l2_distance) # Larger means more similar

# Convert Cosine distances to similarity scores in [0, 1] range
def cosine_to_similarity(cosine_distance: float) -> float:
    return 1.0 - cosine_distance # Larger means more similar

# Combine semantic and emotional retrievals
def _combination_retriever(
    client,
    semantic_db_name,
    emotion_db_name,
    embedding_model,
    query_text: str,
    score_fn,
    retrieval_type: str,
):  
    # Retrieve from both databases
    semantic_output = retrieve_semantic_vectors(client, semantic_db_name, embedding_model, query_text, top_k=5)
    emotion_output = retrieve_emotion_vectors(client, emotion_db_name, query_text, top_k=5)

    sem_sims = [cosine_to_similarity(r["distance"]) for r in semantic_output["results"]]
    emo_sims = [l2_to_similarity(r["distance"]) for r in emotion_output["results"]]

    # Define fallback similarities for missing entries
    fallback_sem = min(sem_sims) - 0.01 * (max(sem_sims) - min(sem_sims))
    fallback_emo = min(emo_sims) - 0.01 * (max(emo_sims) - min(emo_sims))

    scores = {}

    # Process semantic results
    for r in semantic_output["results"]:
        scores[r["id"]] = {
            "semantic_similarity": cosine_to_similarity(r["distance"]),
            "emotional_similarity": fallback_emo,
            "document": r["document"],
            "ai_response": r.get("ai_response"),
            "last_asked": r.get("timestamp"),
        }
    # Process emotional results, updating existing entries or adding new ones
    for r in emotion_output["results"]:
        scores.setdefault(r["id"], {
            "semantic_similarity": fallback_sem,
            "emotional_similarity": None,
            "document": r.get("document"),
            "ai_response": r.get("ai_response"),
            "last_asked": r.get("timestamp"),
        })["emotional_similarity"] = l2_to_similarity(r["distance"])

    # Compute final combined scores
    for s in scores.values():
        s["final_score"] = score_fn(s["semantic_similarity"], s["emotional_similarity"])

    # Select top 5 results based on final scores
    top_scores = dict(
        sorted(scores.items(), key=lambda x: x[1]["final_score"], reverse=True)[:5]
    )

    return {
        "query": query_text,
        "type": retrieval_type,
        "retrieval_timestamp": semantic_output["retrieval_timestamp"],
        "results": top_scores,
    }

def multiplicative_retriever(client, semantic_db_name, emotion_db_name, embedding_model, query_text):
    return _combination_retriever(
        client,
        semantic_db_name,
        emotion_db_name,
        embedding_model,
        query_text,
        score_fn=lambda s, e: (s ** alpha) * (e ** beta), # Multiplicative scoring
        retrieval_type="multiplicative_retrieval",
    )


def additive_retriever(client, semantic_db_name, emotion_db_name, embedding_model, query_text):
    return _combination_retriever(
        client,
        semantic_db_name,
        emotion_db_name,
        embedding_model,
        query_text,
        score_fn=lambda s, e: (s * alpha) + (e * beta), # Additive scoring
        retrieval_type="additive_retrieval",
    )

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Directory to persist Chroma database
    persist_dir = "./chroma_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    query_text = "I am not able to sleep properly at night"

    alpha = 0.70  # Weight for semantic similarity
    beta = 1 - alpha  # Weight for emotional similarity
    
    comb_output = multiplicative_retriever(client, "semantic_vectors", "emotions_vectors", embedding_model, query_text)
    logs_save(type="combination_retrieval", output=comb_output)
    
    comb_output = additive_retriever(client, "semantic_vectors", "emotions_vectors", embedding_model, query_text)
    logs_save(type="combination_retrieval", output=comb_output)
import os
import chromadb
from torch import cosine_similarity
from simple_retrieval import retrieve_semantic_vectors, retrieve_emotion_vectors, logs_save
from sentence_transformers import SentenceTransformer

alpha = 0.65  # Weight for semantic similarity
beta = 1 - alpha  # Weight for emotional similarity

# multiplicative_score = (cosine_similarity ** alpha) * (l2_scale ** beta) # Combined similarity score - multiplicative approach
# additive_score = (alpha * cosine_similarity) + (beta * l2_scale) # Combined similarity score - additive approach

# Convert Euclidean distances to similarity scores in [0, 1] range 
def l2_to_similarity(l2_distance: float) -> float:
    return 1.0 / (1.0 + l2_distance) # Larger means more similar

# Convert Cosine distances to similarity scores in [0, 1] range
def cosine_to_similarity(cosine_distance: float) -> float:
    return 1.0 - cosine_distance # Larger means more similar

def multiplicative_retriever(client, semantic_db_name, emotion_db_name, embedding_model, query_text: str):
    semantic_output = retrieve_semantic_vectors(client, semantic_db_name, embedding_model, query_text)
    emotion_output = retrieve_emotion_vectors(client, emotion_db_name, query_text)

    max_semantic_similarity = max(cosine_to_similarity(r["distance"]) for r in semantic_output["results"])
    min_semantic_similarity = min(cosine_to_similarity(r["distance"]) for r in semantic_output["results"])
    SEMANTIC_EPSILON = 0.01*(max_semantic_similarity - min_semantic_similarity)
    fallback_semantic_similarity = min_semantic_similarity - SEMANTIC_EPSILON

    max_emotional_similarity = max(l2_to_similarity(r["distance"]) for r in emotion_output["results"])
    min_emotional_similarity = min(l2_to_similarity(r["distance"]) for r in emotion_output["results"])
    EMOTIONAL_EPSILON = 0.01*(max_emotional_similarity - min_emotional_similarity)
    fallback_emotional_similarity = min_emotional_similarity - EMOTIONAL_EPSILON
    
    intermediate_scores = {}
    for sem_result in semantic_output["results"]:
        sem_id = sem_result["id"]
        document = sem_result["document"]
        ai_response = sem_result.get("ai_response")
        last_asked = sem_result.get("timestamp")
        intermediate_scores[sem_id] = {"semantic_similarity": cosine_to_similarity(sem_result["distance"]), "emotional_similarity": fallback_emotional_similarity, "document": document, "ai_response": ai_response, "last_asked": last_asked}

    for emo_result in emotion_output["results"]:
        emo_id = emo_result["id"]

        if emo_id in intermediate_scores: # ID exists → just add emotion distance
            intermediate_scores[emo_id]["emotional_similarity"] = l2_to_similarity(emo_result["distance"])
        else:
            # ID missing → create new entry
            intermediate_scores[emo_id] = {
                "semantic_similarity": fallback_semantic_similarity,  # safe fallback
                "emotional_similarity": l2_to_similarity(emo_result["distance"]),
                "document": emo_result.get("document"),
                "ai_response": emo_result.get("ai_response"),
                "last_asked": emo_result.get("timestamp")
            }

    for chunk_id, scores in intermediate_scores.items():
        semantic_scores = scores["semantic_similarity"]
        emotional_scores = scores["emotional_similarity"]

        scores["final_score"] = (semantic_scores ** alpha) * (emotional_scores ** beta)
    # Keep only top 5 results based on final_score
    intermediate_scores = dict(sorted(intermediate_scores.items(),key=lambda item: item[1]["final_score"],reverse=True)[:5])

    combined_output = {
        "query": query_text,
        "type": "multiplicative_retrieval",
        "retrieval_timestamp": semantic_output["retrieval_timestamp"],
        "results": intermediate_scores
    }

    return combined_output

if __name__ == "__main__":
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Directory to persist Chroma database
    persist_dir = "./chroma_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    query_text = "I am not able to sleep properly at night and feel anxious."

    combined_output = multiplicative_retriever(client, "semantic_vectors", "emotions_vectors", embedding_model, query_text)
    # print(combined_output)
    logs_save(type="combined_retrieval", output=combined_output)
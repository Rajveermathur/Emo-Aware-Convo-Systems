import os
import chromadb
from sentence_transformers import SentenceTransformer
from combinational_retriever import l2_to_similarity, cosine_to_similarity
from simple_retrieval import retrieve_semantic_vectors, retrieve_emotion_vectors, logs_save
import sys

def rank_dict(scores, top_k=5):
    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

def emotion_first_retriever(
    client,
    emotion_db_name,
    semantic_db_name,
    embedding_model,
    query_text,
    emotion_top_k=5,
    final_top_k=5,
):
    emotion_output = retrieve_emotion_vectors(
        client,
        emotion_db_name,
        query_text,
        top_k=emotion_top_k
    )

    # Convert L2 distances to similarity
    emotion_scores = {
        r["id"]: l2_to_similarity(r["distance"])
        for r in emotion_output["results"]
    }
    print("Emotion Scores:", emotion_scores)

    # Emotion gate
    gated_ids = {
        doc_id
        for doc_id, sim in emotion_scores.items()
    }
    print("Gated IDs:", gated_ids)

    # Semantic retrieval
    semantic_output = retrieve_semantic_vectors(
        client,
        semantic_db_name,
        embedding_model,
        query_text,
        filter_ids=gated_ids
    )
    print("Semantic Output:", semantic_output)
    
    # --- Build structured scores ---
    scores = {}
    ranking_signal = {}

    for r in semantic_output["results"]:
        doc_id = r["id"]

        semantic_sim = cosine_to_similarity(r["distance"])
        emotional_sim = emotion_scores.get(doc_id)  # fallback handled here

        scores[doc_id] = {
            "id": doc_id,
            "semantic_similarity": semantic_sim,
            "emotional_similarity": emotional_sim,
            "document": r["document"],
            "emotions": r["emotions"],
            "ai_response": r.get("ai_response"),
            "last_asked": r.get("timestamp"),
        }

        # Ranking is purely semantic (as you specified)
        ranking_signal[doc_id] = semantic_sim

    # --- Rank ---
    ranked_ids = rank_dict(ranking_signal, final_top_k)

    # Preserve order and return full objects
    final_results = [
        scores[doc_id]
        for doc_id, _ in ranked_ids
    ]

    return {
        "type": "emotion_first_rerank",
        "query": query_text,
        "retrieval_timestamp": emotion_output["retrieval_timestamp"],
        "results": final_results,
    }  

def semantic_first_retriever(
    client,
    emotion_db_name,
    semantic_db_name,
    embedding_model,
    query_text,
    semantic_top_k=5,
    final_top_k=5,
):
    semantic_output = retrieve_semantic_vectors(
        client,
        semantic_db_name,
        embedding_model,
        query_text,
        top_k=semantic_top_k
    )

    # Convert L2 distances to similarity
    semantic_scores = {
        r["id"]: cosine_to_similarity(r["distance"])
        for r in semantic_output["results"]
    }
    print("Semantic Scores:", semantic_scores)
    # Emotion gate
    gated_ids = {
        doc_id
        for doc_id, sim in semantic_scores.items()
    }
    print("Gated IDs:", gated_ids)

    # Semantic retrieval
    emotion_output = retrieve_emotion_vectors(
        client,
        emotion_db_name,
        query_text,
        filter_ids=gated_ids
    )
    print("Semantic Output:", semantic_output)
    
    # --- Build structured scores ---
    scores = {}
    ranking_signal = {}

    for r in emotion_output["results"]:
        doc_id = r["id"]

        emotional_sim = l2_to_similarity(r["distance"])  # fallback handled here
        semantic_sim = semantic_scores.get(doc_id)

        scores[doc_id] = {
            "id": doc_id,
            "semantic_similarity": semantic_sim,
            "emotional_similarity": emotional_sim,
            "document": r["document"],
            "emotions": r["emotions"],
            "ai_response": r.get("ai_response"),
            "last_asked": r.get("timestamp"),
        }

        # Ranking is purely emotion (as you specified)
        ranking_signal[doc_id] = emotional_sim

    # --- Rank ---
    ranked_ids = rank_dict(ranking_signal, final_top_k)

    # Preserve order and return full objects
    final_results = [
        scores[doc_id]
        for doc_id, _ in ranked_ids
    ]

    return {
        "type": "semantic_first_rerank",
        "query": query_text,
        "retrieval_timestamp": semantic_output["retrieval_timestamp"],
        "results": final_results
    }  


def main():
    embedding_model = SentenceTransformer("all-mpnet-base-v2")

    # Directory to persist Chroma database
    persist_dir = "./chroma_store"
    os.makedirs(persist_dir, exist_ok=True)

    # Persistent client
    client = chromadb.PersistentClient(path=persist_dir)

    # query_text = "I am not able to sleep properly at night"
    query_text = sys.argv[1]

    output = emotion_first_retriever(client, "emotions_vectors", "semantic_vectors", embedding_model, query_text)
    logs_save("emotion_first_retriever", output)

    output = semantic_first_retriever(client, "emotions_vectors", "semantic_vectors", embedding_model, query_text)
    logs_save("semantic_first_retriever", output)

if __name__ == "__main__":
    main()

from combinational_retriever import l2_to_similarity, cosine_to_similarity
from simple_retrieval import retrieve_semantic_vectors, retrieve_emotion_vectors, logs_save

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
    emotion_top_k=20,
    final_top_k=5,
    emotion_threshold=0.4
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

    # Emotion gate
    gated_ids = {
        doc_id
        for doc_id, sim in emotion_scores.items()
        if sim >= emotion_threshold
    }

    if not gated_ids:
        return {"warning": "No emotionally aligned results"}

    # Semantic retrieval
    semantic_output = retrieve_semantic_vectors(
        client,
        semantic_db_name,
        embedding_model,
        query_text,
        filter_ids=gated_ids
    )

    # Final ranking purely semantic
    ranked = rank_dict(
        {
            r["id"]: 1.0 - r["distance"]
            for r in semantic_output["results"]
        },
        final_top_k
    )

    return {
        "type": "emotion_first_rerank",
        "query": query_text,
        "results": ranked
    }

# def semantic_first_retriever(
#     client,
#     semantic_db_name,
#     emotion_db_name,
#     embedding_model,
#     query_text,
#     semantic_top_k=20,
#     final_top_k=5,
#     emotion_weight=0.3
# ):
#     semantic_output = retrieve_semantic_vectors(
#         client,
#         semantic_db_name,
#         embedding_model,
#         query_text,
#         top_k=semantic_top_k
#     )

#     semantic_scores = {
#         r["id"]: 1.0 - r["distance"]
#         for r in semantic_output["results"]
#     }

#     candidate_ids = set(semantic_scores.keys())

#     # Emotion rerank
#     emotion_output = retrieve_emotion_vectors(
#         client,
#         emotion_db_name,
#         query_text,
#         filter_ids=candidate_ids
#     )

#     emotion_scores = {
#         r["id"]: l2_to_similarity(r["distance"])
#         for r in emotion_output["results"]
#     }

#     # Combine scores (additive rerank)
#     final_scores = {}
#     for doc_id in candidate_ids:
#         final_scores[doc_id] = (
#             (1 - emotion_weight) * semantic_scores.get(doc_id, 0.0)
#             + emotion_weight * emotion_scores.get(doc_id, 0.0)
#         )

#     ranked = rank_dict(final_scores, final_top_k)

#     return {
#         "type": "semantic_first_rerank",
#         "query": query_text,
#         "results": ranked
#     }

# Emotion-Aware Long-Term Conversational Memory

This repository presents an **emotion-aware long-term conversational memory framework** for Large Language Model (LLM)–based dialogue agents. The system enhances traditional Retrieval-Augmented Generation (RAG) by integrating **emotional context** into memory representation and retrieval, enabling more **human-like, emotionally consistent, and personalized conversations** over extended interactions.

---

## 🚀 Motivation

While modern LLMs excel at short-term dialogue, they struggle with:
- Long-term memory retention across sessions  
- Emotional continuity and consistency  
- Contextually appropriate recall of past interactions  

Human memory is strongly influenced by emotion. Inspired by **Mood-Dependent Memory theory**, this project introduces emotion-aware memory retrieval, allowing conversational agents to recall information not only based on *what was said* but also *how it felt*.

---

## 🧠 Key Features

- **Emotion Modeling with Plutchik’s Framework**  
  Extracts an 8-dimensional emotion vector (joy, acceptance, fear, surprise, sadness, disgust, anger, anticipation), with intensity scores from 1–10.

- **Emotion-Aware Memory Retrieval**  
  Combines semantic similarity and emotional alignment using cosine similarity and Euclidean distance.

- **Long-Term Memory Support**  
  Designed for multi-session conversations with persistent memory storage and retrieval.

- **LLM-Based Emotion Extraction**  
  Uses **LLaMA-3.3-70B-Versatile** with structured prompts to generate stable emotion embeddings.

- **Reflective Memory Design (Inspired by RMM)**  
  Supports flexible memory granularity and improved retrieval quality over time.

---

## 🏗️ System Architecture (High-Level)

1. **User Query Processing**
   - Extract semantic embedding  
   - Extract emotion vector  

2. **Memory Retrieval**
   - Semantic similarity (text embeddings)  
   - Emotional similarity (emotion vectors)  
   - Weighted joint scoring  

3. **Response Generation**
   - Retrieved memory + current context  
   - Emotionally aligned response generation  

---

## 📐 Emotion Representation

Each user query is mapped to an emotion vector:

```text
emotion_q ∈ ℝ⁸
emotion_q = [joy, acceptance, fear, surprise, sadness, disgust, anger, anticipation]
```

Where each dimension is an intensity score from 1 to 10.

---
## ⚙️ Memory Retrieval Algorithm
1. Compute semantic similarity:
   ```math
   sim_semantic = cosine\_similarity(embedding_q, embedding_m)
   ```
2. Compute emotional similarity:
   ```math
    dist_emotion = euclidean\_distance(emotion_q, emotion_m)
    sim_emotion = 1 / (1 + dist_emotion)
    ```
3. Combine scores:
   ```math
   sim_total = α * sim_semantic + (1 - α) * sim_emotion
   ```
   Where `α` is a tunable parameter (0 ≤ α ≤ 1).
4. Retrieve top-K memories based on `sim_total`.
---

# SYSTEM PROMPT
You are a mental health counselor. Your task is to provide thoughtful and offer 
constructive advice or encouragement when appropriate. Keep replies to one line.

# USER PROMPT
You are a compassionate mental health counselor.

Your role:
- Respond with empathy and emotional validation
- Gently reflect recurring emotional patterns without sounding clinical
- Offer practical, non-overwhelming coping suggestions
- Never mention internal data, scores, timestamps, or memory systems

Current user concern:
**{query}**

Relevant memories from previous conversations:
**{memory_block}**

Guidelines for using memory:
- If similar feelings occurred recently, acknowledge recurrence gently
- Use soft temporal language (e.g., "recently", "before", "this has come up again")
- Do NOT mention exact dates or frequency
- Do NOT quote past messages verbatim
- Integrate memory naturally into reflection

Response structure:
1. Validate the current feeling
2. Reflect any recurring emotional pattern if present
3. Offer one grounding or coping suggestion relevant to the concern

Tone:
- Warm, calm, non-judgmental
- Human and supportive, not clinical
- Avoid advice overload
# Preprocessing module for chat data to create chunks and generate emotional embeddings
# Dependencies
import json
from uuid import uuid4
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
import ast
from datetime import datetime
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()
api_key = os.environ.get("API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=10)

# Function to create chunks from chat data
def make_chunks(chat_data):
    chunks = []
    i = 0
    while i < len(chat_data):
        if chat_data[i]["speaker"] == "Human":  # Only process human messages
            human_msg = chat_data[i]["message"] # Human message
            ai_msg = chat_data[i + 1]["message"] if (i + 1 < len(chat_data) and chat_data[i + 1]["speaker"] == "AI") else "" # Handle missing AI response
            
            # Create chunk
            chunk = {
                "id": str(uuid4()),
                "text": human_msg.strip(),
                "timestamp": datetime.now().isoformat(),
                "ai_response": ai_msg.strip(),
                "exchange_number": i // 2 + 1
            }
            chunks.append(chunk)
        i += 1
    return chunks

# Function to generate emotional embeddings using LLM
def emotional_embedder(user_query):
    chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a master of sentiment analysis. Carefully discern the subtle emotions "
                "underlying each interviewer's question. Analyze questions across 8 dimensions: "
                "acceptance, anger, anticipation, disgust, fear, joy, sadness, surprise. "
                "Score each from 1-10. Your answer must be a valid python list so that it can "
                "be parsed directly, with no extra content! Format: "
                "[{\"analysis\": <REASON>, \"dim\": \"joy\", \"score\": <SCORE>}, ...]"
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ],
    temperature=0.1,
    max_tokens=1024,
    stream=False,
    stop=None,
    )
    
    # Parse and structure LLM output
    try:
        raw_output = chat_completion.choices[0].message.content
        parsed_data = ast.literal_eval(raw_output.strip()) # Safely parse string to Python object
        # print("Input: ", user_query)
        # print(validate_emotion_schema(parsed_data)) # Validate schema
        # print("Step 1: ",parsed_data) 

        dimension_scores = { # Create dimension-score mapping
        item["dim"]: item["score"]
        for item in parsed_data
        }
        # print("Step 2: ",dimension_scores)
        
        sorted_emotions = dict(sorted(dimension_scores.items())) # Sort by dimension name
        # print("Step 3: ",sorted_emotions)
        
        score_list = [sorted_emotions[dim] for dim in sorted_emotions] # Extract scores in sorted order
        # print("Step 4: ",score_list)
        
        structured_log = { # Final structured log
            "embed_timestamp": datetime.now().isoformat(),
            "emotion_logs": parsed_data,
            "emotion_scores": sorted_emotions,
            "embedding": score_list
        }
    # Handle parsing errors
    except Exception as e:
        print(f"Failed to parse LLM output: {e}")
        print(f"Raw output was: {raw_output}")

    return structured_log

# Function to validate the schema of LLM output
def validate_emotion_schema(data):
    if not isinstance(data, list):
        raise TypeError("Expected a list of emotion objects")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Item {i} is not a dict")

        if "dim" not in item or "score" not in item:
            raise KeyError(f"Item {i} missing 'dim' or 'score'")

        if not isinstance(item["dim"], str):
            raise TypeError(f"Item {i} 'dim' must be a string")

        if not isinstance(item["score"], (int, float)):
            raise TypeError(f"Item {i} 'score' must be a number")
        
    return 'LLM output schema is valid ✅'

# Main execution
if __name__ == "__main__":
    # Load raw chat data
    with open("data\\raw_chat_v2.json", "r") as f:
        chat_data = json.load(f)
        print(f"Loaded {len(chat_data)} chat messages ✅")
    
    # # Alternatively, load pre-made chunks for testing
    # with open("chunks_ready_single.json", "r") as f:
    #     chunks = json.load(f)

    chunks = make_chunks(chat_data) # Create chunks from chat data
    for chunk in tqdm(chunks, desc="Embedding emotions", unit="chunk", total=len(chunks)): 
        emotion_results  = emotional_embedder(chunk["text"]) # Generate emotional embeddings
        chunk["emotions_metadata"] = emotion_results # Attach embeddings to chunk
        time.sleep(6)  # To avoid hitting rate limits
        # break # For testing, process only the first chunk

    # Save processed chunks to file
    with open("data\\chunks_ready_v2.json", "w") as f:
        json.dump(chunks, f, indent=2)
        
    print(f"Created {len(chunks)} chunks ✅")
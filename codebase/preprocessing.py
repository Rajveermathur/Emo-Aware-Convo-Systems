import json
from uuid import uuid4
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import ast
from datetime import datetime

load_dotenv()
api_key = os.environ.get("API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=10)

def make_chunks(chat_data):
    chunks = []
    i = 0
    while i < len(chat_data):
        if chat_data[i]["speaker"] == "Human":
            human_msg = chat_data[i]["message"]
            ai_msg = chat_data[i + 1]["message"] if (i + 1 < len(chat_data) and chat_data[i + 1]["speaker"] == "AI") else ""
            
            chunk = {
                # "id": f"turn_{i}_{uuid4().hex[:6]}",
                "text": human_msg.strip(),
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "ai_response": ai_msg.strip(),
                    "exchange_number": i // 2 + 1
                }
            }
            chunks.append(chunk)
        i += 1
    return chunks

def emotional_embedder(user_query):
    chat_completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a master of sentiment analysis. Carefully discern the subtle emotions "
                "underlying each interviewer's question. Analyze questions across 8 dimensions: "
                "joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation. "
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
    print(chat_completion.choices[0].message.content)

    # 1. Get the raw content from your Groq completion
    raw_output = chat_completion.choices[0].message.content
    # query_text = "Today I am feeling very confident about my upcoming exams."

    try:
        # 2. Parse the string into a Python list
        # We use ast.literal_eval as a safer alternative to eval() for python lists/dicts
        # Or json.loads() if the LLM output is strictly valid JSON format
        parsed_data = ast.literal_eval(raw_output.strip())

        # 3. Create the final structured object
        structured_log = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "model": 'llama-3.3-70b-versatile',
            "emotion_results": parsed_data
        }

        # 4. Output or save the results
        print(json.dumps(structured_log, indent=4))

    except Exception as e:
        print(f"Failed to parse LLM output: {e}")
        print(f"Raw output was: {raw_output}")
    
    dimension_scores = {
    item["dim"]: item["score"]
    for item in structured_log["emotion_results"]
    }
    print(dimension_scores)

    dimensions = sorted(item["dim"] for item in structured_log["emotion_results"])
    print(dimensions)

    # Create a mapping of dimension to score
    score_map = {item["dim"]: item["score"] for item in structured_log["emotion_results"]}
    print(score_map)
    # Generate the score list in alphabetical order
    score_list = [score_map[dim] for dim in dimensions]

    print(score_list)
    return score_list

# Main execution
if __name__ == "__main__":
    # with open("..\\raw_data\\chat.json", "r") as f:
    #     chat_data = json.load(f)

    with open("chunks_ready_try.json", "r") as f:
        chunks = json.load(f)

    # chunks = make_chunks(chat_data)
    for chunk in chunks:
        user_query = chunk["text"]
        embedding = emotional_embedder(user_query)
        chunk["embedding"] = embedding
        break  # Remove this break to process all chunks

    with open("chunks_ready_try.json", "w") as f:
        json.dump(chunks, f, indent=2)

    # print(f"Created {len(chunks)} chunks ✅")
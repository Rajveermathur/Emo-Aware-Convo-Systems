import json
from uuid import uuid4
from datetime import datetime

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

def emotional_embedder(text):
    pass  # Placeholder for future implementation   

if __name__ == "__main__":
    with open("..\\raw_data\\chat.json", "r") as f:
        chat_data = json.load(f)

    chunks = make_chunks(chat_data)

    with open("..\\chunks_ready_1.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Created {len(chunks)} chunks ✅")
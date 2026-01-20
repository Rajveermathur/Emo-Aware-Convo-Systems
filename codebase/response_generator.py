import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

with open("logs/semantic_first_retriever_20260117_170649.json", "r") as f:
    memory_block = f.read()

prompt = """You are a compassionate mental health counselor.

Your role:
- Respond with empathy and emotional validation
- Gently reflect recurring emotional patterns without sounding clinical
- Offer practical, non-overwhelming coping suggestions
- Never mention internal data, scores, timestamps, or memory systems

Current user concern:
"{query}"

Relevant memories from previous conversations:
{memory_block}

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
"""

def response_generator(user_query):
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a mental health counselor. Your task is to provide thoughtful and offer constructive advice or encouragement when appropriate. Keep replies to one line.\n\n"
                )
            },
            {
                "role": "user",
                "content": prompt.format(
                    query=user_query, memory_block=memory_block)
            }
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=False,
        stop=None,
    )

    return chat_completion.choices[0].message.content



if __name__ == "__main__":
    test_query = "I am not able to sleep properly at night."
    response = response_generator(test_query, memory_block)
    print("Response:", response)
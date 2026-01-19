import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=10)

def response_generator(user_query):
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

    return chat_completion.choices[0].message.content


if __name__ == "__main__":
    test_query = "I just got a promotion at work, but I'm worried about balancing my new responsibilities with my personal life."
    response = response_generator(test_query)
    print("Response:", response)
import os
from openai import OpenAI

try:
    GROQ_API_KEY = "your-api-key-here"

    if GROQ_API_KEY == "your-api-key-here" or not GROQ_API_KEY.strip():
        raise ValueError("Groq API key is missing. Please replace 'your-api-key-here' with your actual API key.")

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

except Exception as e:
    print(f"❌ Initialization error: {e}")
    exit(1)

model_name = "llama3-8b-8192"

def ask_groq(query, context=None):
    if not query:
        return "⚠️ No query provided."

    if context:
        # There are two scenarios where the api gets called. These system prompts can be changed to fit a particular role
        # if context is given, it gives the answer to the user.
        system_prompt = (
            "You are a helpful assistant answering questions based on multiple PDF documents. "
            "Use the provided context to answer concisely. Cite filenames and page numbers if relevant. Also ask if the user requires any other information or not."
            "Try to keep each conversation long but still to the point."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    else:
        # Otherwise act as a general ChatBot
        system_prompt = (
            "You are a helpful assistant that answers general knowledge questions."
            "Answers can be longer if need be."
        )
        user_prompt = query

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error while calling Groq API: {e}"

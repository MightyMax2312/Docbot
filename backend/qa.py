from groq import Groq
import os
import streamlit as st

# It's recommended to set the API key via environment variables
# For Streamlit sharing, you would set this in the secrets manager
try:
    # This will read the secret you set in the Streamlit Community Cloud settings
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    # Fallback for local development if you use a .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        st.error(f"Groq API key not found. Please set it in your Streamlit secrets or a local .env file. Error: {e}")
        client = None


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

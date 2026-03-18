import os
from google import genai
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
# Read API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

# Configure client
client = genai.Client(api_key=api_key)


SYSTEM_PROMPT = """
You are a helpful customer support chatbot for an online store.

Policies:
- Refund can be done in the 7 days of delivery.
- Shipping takes 3-5 business days.
- We accept credit cards, debit cards, and PayPal.

Rules:
- Answer politely.
- Keep responses short.
- If the question is unrelated to the store, say you can only help with store questions.
"""

def ask_gemini():

    contents = []

    # add system prompt
    contents.append({
        "role": "user",
        "parts": [{"text": SYSTEM_PROMPT}]
    })

    # add conversation history
    for msg in st.session_state.messages:
        role = msg["role"]
        text = msg["content"]

        if role == "assistant":
            role = "model"

        contents.append({
            "role": role,
            "parts": [{"text": text}]
        })

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=contents
    )

    return response.text


#streamlit ui

st.title("Gemini Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Gemini response
    reply = ask_gemini()

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
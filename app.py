import streamlit as st
import openai
import pandas as pd
import os

st.title("Omniscience: Sports Betting Chatbot")

# Load your OpenAI API key from Streamlit secrets or environment variable
api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OpenAI API key not found! Please set it in Streamlit secrets or as an environment variable.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Upload data
uploaded_file = st.file_uploader("Upload your matchup data (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
else:
    data = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask Omniscience anything about today's games, matchups, or props:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Simple prediction logic (replace with your model)
    if data is not None:
        prediction = data.iloc[0].to_dict()
        insight = f"Here's a sample prediction: {prediction}"
    else:
        insight = "Please upload data for predictions."

    # Prepare messages for OpenAI
    messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    messages.append({"role": "assistant", "content": insight})

    # Call OpenAI Chat API (new syntax)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

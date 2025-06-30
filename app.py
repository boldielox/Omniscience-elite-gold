import streamlit as st
import openai  # pip install openai
import pandas as pd

st.title("Omniscience: Sports Betting Chatbot")

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

# Upload data
uploaded_file = st.file_uploader("Upload your matchup data (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
else:
    data = None

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask Omniscience anything about today's games, matchups, or props:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Simple prediction logic (replace with your model)
    if data is not None:
        # Example: Just echo the first row as a "prediction"
        prediction = data.iloc[0].to_dict()
        insight = f"Here's a sample prediction: {prediction}"
    else:
        insight = "Please upload data for predictions."

    # Use OpenAI to generate a conversational response
    prompt = f"You are Omniscience, an elite sports betting AI. {insight} User asked: {user_input}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages] +
                 [{"role": "assistant", "content": insight}]
    )
    answer = response.choices[0].message["content"]
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state.messages:
    st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

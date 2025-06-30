import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None

if not api_key:
    st.error("API key not found!")
else:
    st.success("API key loaded successfully!")

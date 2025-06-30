import streamlit as st

st.title("Test Streamlit App")
st.write("If you see this message, the main area is working fine!")

if st.button("Click me"):
    st.success("You clicked the button!")

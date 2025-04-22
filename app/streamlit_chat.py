# app/streamlit_chat.py
import streamlit as st
import requests

st.title("Amazon Automotive Chatbot")

query = st.text_area("Ask a question about Automotive products:")
if st.button("Send"):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        resp = requests.post("http://localhost:5001/chat", json={"query": query})
        if resp.ok:
            ans = resp.json().get("answer", "(no answer)")
            st.markdown(f"**AI:** {ans}")
        else:
            st.error(f"Error: {resp.status_code} {resp.text}")
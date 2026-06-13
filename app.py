import streamlit as st

st.title("Smart AI Study Assistant")

query = st.chat_input("Ask me anything")

if query:
    st.write("You:", query)
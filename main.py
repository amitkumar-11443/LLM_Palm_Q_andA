import streamlit as st
from langchain_helper import create_vector_db,get_qa_chain

st.title("logicfirst Demo QA 🌱")

btn = st.button("Update Database")
if btn:
    pass

question=st.text_input("Question: ")

if question:
    chain = get_qa_chain()  # initialize the function
    response = chain(question) #

    st.header("Answer")
    st.write(response["result"])  # syntax to put the answer below question


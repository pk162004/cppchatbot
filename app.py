import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="My GPT",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 My GPT")
st.write("Powered by LangChain + Ollama + Gemma 2")

# ----------------------------------------------------
# Load LLM
# ----------------------------------------------------

llm = ChatOllama(
    model="gemma2",      # You have gemma2:latest installed
    temperature=0
)

# ----------------------------------------------------
# Prompt Template
# ----------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the user's questions accurately,
            clearly, and concisely.
            """
        ),
        (
            "human",
            "{question}"
        )
    ]
)

# ----------------------------------------------------
# Output Parser
# ----------------------------------------------------

output_parser = StrOutputParser()

# ----------------------------------------------------
# Create Chain
# ----------------------------------------------------

chain = prompt | llm | output_parser

# ----------------------------------------------------
# User Input
# ----------------------------------------------------

question = st.text_input(
    "What question do you have in mind?"
)

# ----------------------------------------------------
# Ask Button
# ----------------------------------------------------

if st.button("Ask"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner("Thinking..."):

                response = chain.invoke(
                    {
                        "question": question
                    }
                )

            st.success("Response")
            st.write(response)

        except Exception as e:
            st.error(f"Error: {e}")
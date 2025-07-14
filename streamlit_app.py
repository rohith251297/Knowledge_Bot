import streamlit as st
import os
from configure import OPENAI_API_KEY, CHROMA_DB_DIR, COLLECTION_NAME
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Set API Key from configure.py
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load persisted Chroma vectorstore
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(),
    persist_directory=CHROMA_DB_DIR
)

retriever = vectorstore.as_retriever()

# Custom Prompt Template with memory support
template = """
Use the following context (between <ctx></ctx>) and the chat history (between <hs></hs>) to answer the question.

------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template
)

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    return_messages=True
)

# QA chain using RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt,
        "memory": memory
    },
    verbose=True
)

# Streamlit UI
st.title("📄 Chat with PDFs")

# Initialize session state
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat input box
question = st.chat_input("Ask your question here...")

# Process question
if question:
    with st.spinner("Thinking..."):
        response = qa_chain.run(question)
        st.session_state["user_prompt_history"].append(question)
        st.session_state["chat_answers_history"].append(response)
        st.session_state["chat_history"].append((question, response))

# Display chat history
if st.session_state["chat_answers_history"]:
    for user_msg, bot_msg in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]
    ):
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(bot_msg)

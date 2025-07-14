import streamlit as st
import os
from configure import OPENAI_API_KEY, CHROMA_DB_DIR, COLLECTION_NAME
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# -------------------- Setup --------------------

# Set API Key from configure.py
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Load Chroma Vectorstore
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(),
    persist_directory=CHROMA_DB_DIR
)

retriever = vectorstore.as_retriever()

# Clean Prompt Template (No fallback instruction)
template = """
Answer the question based only on the following context, which can include text, images and tables.

Context:
{context}

Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template
)

# LLM initialization
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Conversation Memory
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="question",
    return_messages=True
)

# Retrieval QA chain
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

# -------------------- Streamlit App --------------------

st.title("📄 Chat with PDFs")

# Initialize session state for chat history
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Chat input
question = st.chat_input("Ask your question here...")

# -------------------- Process question --------------------
if question:
    with st.spinner("Thinking..."):
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(question)

        # Collect combined retrieved content
        combined_context = " ".join(doc.page_content.lower() for doc in retrieved_docs)

        # Extract words from question as crude keywords
        question_keywords = [word.strip("?.,").lower() for word in question.split() if len(word) > 2]

        # Check if any keyword is in the retrieved content
        has_relevant_info = any(keyword in combined_context for keyword in question_keywords)

        # Fallback if no relevant information found
        if not retrieved_docs or not has_relevant_info:
            response = "I couldn’t find that information in the provided documents."
        else:
            # Run QA chain with clean prompt
            response = qa_chain.run(question)

        # Store chat history
        st.session_state["user_prompt_history"].append(question)
        st.session_state["chat_answers_history"].append(response)
        st.session_state["chat_history"].append((question, response))

# -------------------- Display Chat History --------------------
if st.session_state["chat_answers_history"]:
    for user_msg, bot_msg in zip(
        st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]
    ):
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(bot_msg)



# pdf_processing.py
import os
import uuid
import base64
import time
import openai
import configure as cfg
from unstructured.partition.pdf import partition_pdf
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage


# ---------------------------
# Vector Store Setup
# ---------------------------
vectorstore = Chroma(
    collection_name=cfg.COLLECTION_NAME,
    embedding_function=OpenAIEmbeddings(),
    persist_directory=cfg.CHROMA_DB_DIR
)
store = InMemoryStore()

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id"
)

# GPT Model
gpt = ChatOpenAI(model="gpt-4o-mini", max_tokens=1024)


# ---------------------------
# Utility Functions
# ---------------------------
def encode_image(image_path: str) -> str:
    """Convert image to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def safe_gpt_invoke(messages):
    """Safely invoke GPT model with rate limit handling."""
    wait_time = 60
    while True:
        try:
            return gpt.invoke(messages).content
        except openai.RateLimitError:
            print(f"Rate limit hit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)


def summarize_text(text: str) -> str:
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    return safe_gpt_invoke([HumanMessage(content=prompt)])


def summarize_table(table: str) -> str:
    prompt = f"Summarize the following table:\n\n{table}\n\nSummary:"
    return safe_gpt_invoke([HumanMessage(content=prompt)])


def summarize_image(encoded_image: str) -> str:
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(content=[
            {"type": "text", "text": "Describe the contents of this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ])
    ]
    return safe_gpt_invoke(prompt)


def add_to_retriever(summaries, originals, filename, content_type, pdf_path):
    """Store summaries and originals with metadata in retriever."""
    doc_ids = [str(uuid.uuid4()) for _ in summaries]
    docs = [
        Document(
            page_content=s,
            metadata={
                "doc_id": doc_ids[i],
                "filename": filename,
                "content_type": content_type,
                "source": pdf_path
            }
        )
        for i, s in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(docs)
    retriever.docstore.mset(list(zip(doc_ids, originals)))


# ---------------------------
# Process PDFs
# ---------------------------
os.makedirs(cfg.OUTPUT_BASE, exist_ok=True)

for filename in os.listdir(cfg.INPUT_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(cfg.INPUT_FOLDER, filename)
        pdf_name = os.path.splitext(filename)[0]
        output_path = os.path.join(cfg.OUTPUT_BASE, pdf_name)

        os.makedirs(output_path, exist_ok=True)

        # Extract PDF contents
        raw_elements = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=output_path
        )

        text_elements, table_elements, image_elements = [], [], []

        for element in raw_elements:
            if 'CompositeElement' in str(type(element)):
                text_elements.append(element.text)
            elif 'Table' in str(type(element)):
                table_elements.append(element.text)

        for image_file in sorted(os.listdir(output_path)):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(output_path, image_file)
                image_elements.append(encode_image(image_path))

        # Summarize and store
        if text_elements:
            summaries = [summarize_text(t) for t in text_elements]
            add_to_retriever(summaries, text_elements, filename, "text", pdf_path)

        if table_elements:
            summaries = [summarize_table(t) for t in table_elements]
            add_to_retriever(summaries, table_elements, filename, "table", pdf_path)

        if image_elements:
            summaries = [summarize_image(i) for i in image_elements]
            add_to_retriever(summaries, image_elements, filename, "image", pdf_path)

        print(f"✅ Processed: {filename}")

# Persist vectorstore after processing all PDFs
vectorstore.persist()
print("✅ All PDFs processed and Chroma DB saved.")

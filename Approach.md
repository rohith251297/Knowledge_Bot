
# Approach and Design for Knowledge Bot

## Tech Stack:
- Python
- LangChain
- OpenAI GPT-4o-mini
- ChromaDB
- Streamlit for UI

## Process:
1. **PDF Ingestion:** Using `partition_pdf` from unstructured library.
2. **Text, Tables, Images:** Extracted separately.
3. **Summarization:** GPT-4o-mini for text, tables, images.
4. **Vector Store:** Chroma + OpenAI Embeddings.
5. **Retriever:** MultiVectorRetriever.
6. **LLM Chat:** GPT-4o-mini with conversation memory.
7. **Fallback:** Explicit fallback logic if retrieval fails.

## Limitations:
- Only PDFs supported.
- Dependent on OpenAI API rate limits.
- Simple memory (no advanced RAG chaining).

## Architecture Overview:
```
PDFs --> Chunking --> Summarization (GPT-4o) --> Chroma Vector DB  
|
User ---> Streamlit Chat Interface ---> Retrieval + LLM ---> Answer
```


# Knowledge Bot - Document QA System

A prototype chatbot that answers questions strictly based on the provided documents using embeddings, vector store (Chroma), and OpenAI GPT-4o-mini.

## 💡 Features
- Ingests PDFs and extracts text, tables, and images.
- Stores summaries in a Chroma vector database.
- Retrieval-based QA with GPT-4o-mini.
- Memory-supported follow-up questions.
- Fallback response if the answer is not found.

## 🚀 How to Run
### 1️⃣ Clone Repository
```bash
git clone https://github.com/rohith251297/Knowledge_Bot.git
cd Knowledge_Bot
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Environment Variables
Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
```

### 4️⃣ Run the Pipeline (PDF Processing)
```bash
python process_pdf.py
```

### 5️⃣ Launch Chat App
```bash
streamlit run streamlit_app.py
```

## 🧪 Testing
- Open `localhost:8501`
- Ask questions related to the PDFs.
- Try follow-up questions.
- Ask something outside the documents; bot should say:
“I couldn’t find that information in the provided documents.”

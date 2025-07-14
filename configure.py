# configure.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key (from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Input PDF directory (optional if you need later)
INPUT_FOLDER = r"R:\Knowledge_Bot\documents"

# Output base directory for images extracted from PDFs (optional if you need later)
OUTPUT_BASE = os.path.join(os.getcwd(), "figures")

# Chroma Vector Store persistent directory
CHROMA_DB_DIR = r"R:\New folder (11)\chroma_db"

# Chroma collection name
COLLECTION_NAME = "summaries"

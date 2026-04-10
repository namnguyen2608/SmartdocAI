"""
SmartDocAI - Configuration
Cấu hình hệ thống cho SmartDocAI RAG Chatbot
"""

import os

# ============================================================
# Đường dẫn thư mục
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ============================================================
# Cấu hình Ollama LLM
# ============================================================
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.7
OLLAMA_TOP_P = 0.9
OLLAMA_REPEAT_PENALTY = 1.1

# ============================================================
# Cấu hình Embedding Model
# ============================================================
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DEVICE = "cuda"
EMBEDDING_NORMALIZE = True
# ============================================================
# Cấu hình Text Splitter (Chunking)
# ============================================================
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ============================================================
# Cấu hình Retrieval
# ============================================================
RETRIEVAL_SEARCH_TYPE = "mmr"
RETRIEVAL_TOP_K = 3       
RETRIEVAL_FETCH_K = 30
RETRIEVAL_LAMBDA_MULT = 0.7     

# ============================================================
# Cấu hình FAISS Index
# ============================================================
FAISS_INDEX_NAME = "smartdoc_index"

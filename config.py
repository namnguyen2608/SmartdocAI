# -*- coding: utf-8 -*-
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
EMBEDDING_MODEL = "sentence-transformers/LaBSE"
EMBEDDING_DEVICE = "cpu"
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
RETRIEVAL_TOP_K = 8
RETRIEVAL_FETCH_K = 50
RETRIEVAL_LAMBDA_MULT = 0.7     

# ============================================================
# Cấu hình FAISS Index
# ============================================================
FAISS_INDEX_NAME = "smartdoc_index"

# ============================================================
# Q7 — Hybrid Search (BM25 + Vector Ensemble)
# ============================================================
HYBRID_VECTOR_WEIGHT = 0.6   # trọng số cho FAISS semantic search
HYBRID_BM25_WEIGHT   = 0.4   # trọng số cho BM25 keyword search
HYBRID_TOP_K         = 5     # số kết quả lấy từ mỗi retriever trước khi ensemble

# ============================================================
# Q8 — Multi-document Metadata Filtering
# ============================================================
METADATA_FILTER_FIELD = "source"   # field trong Document.metadata dùng để filter

# ============================================================
# Co-RAG — Collaborative RAG (Multi-Agent)
# ============================================================
CO_RAG_TOP_K_PER_AGENT = 5   # số docs mỗi agent truy xuất
CO_RAG_MIN_VOTES = 2          # số agents tối thiểu phải đồng ý (dùng với strategy "voting")
CO_RAG_MERGE_STRATEGY = "voting"  # "voting" | "union" | "intersection"

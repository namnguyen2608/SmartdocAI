# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.7
OLLAMA_TOP_P = 0.9
OLLAMA_REPEAT_PENALTY = 1.1
OLLAMA_NUM_CTX = 4096

EMBEDDING_MODEL = "sentence-transformers/LaBSE"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_NORMALIZE = True

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

RETRIEVAL_SEARCH_TYPE = "mmr"
RETRIEVAL_TOP_K = 6
RETRIEVAL_FETCH_K = 30
RETRIEVAL_LAMBDA_MULT = 0.7

FAISS_INDEX_NAME = "smartdoc_index"

HYBRID_VECTOR_WEIGHT = 0.6
HYBRID_BM25_WEIGHT   = 0.4
HYBRID_TOP_K         = 3

METADATA_FILTER_FIELD = "source"

CO_RAG_TOP_K_PER_AGENT = 5
CO_RAG_MIN_VOTES = 2
CO_RAG_MERGE_STRATEGY = "voting"

import os
import sys
import json
import logging
import tempfile
import time
import shutil
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Thêm root directory vào path để import các modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.document_processor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    split_documents,
    SUPPORTED_EXTENSIONS,
)
from modules.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    add_documents_to_store,
    clear_vector_store,
    create_bm25_retriever,
    create_ensemble_retriever,
    get_cached_bm25_retriever,
)
from modules.rag_chain import ask_question, check_ollama_connection, get_llm, _compute_rrf_scores
from modules.reranker import rerank_with_cross_encoder, CROSS_ENCODER_MODEL
from modules.self_rag import self_rag_pipeline
from modules.co_rag import co_rag_pipeline

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SmartDocAI_API")

app = FastAPI(title="SmartDocAI API", version="1.0.0")

# Cấu hình CORS để hỗ trợ React chạy trên môi trường dev local khác port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục lưu trữ
_PERSIST_DIR = config.VECTORSTORE_DIR
_FILES_PATH = os.path.join(_PERSIST_DIR, "processed_files.json")
_HISTORY_PATH = os.path.join(_PERSIST_DIR, "chat_history.json")
_CONFIG_PATH = os.path.join(_PERSIST_DIR, "search_config.json")

# Trạng thái trong bộ nhớ (Global State)
state = {
    "vector_store": None,
    "processed_files": [],
    "total_chunks": 0,
    "raw_documents": [],
    "chat_history": [],
    "ollama_status": None,
    
    # Settings mặc định
    "chunk_size": config.CHUNK_SIZE,
    "chunk_overlap": config.CHUNK_OVERLAP,
    "hybrid_enabled": False,
    "reranker_enabled": False,
    "self_rag_enabled": False,
    "self_rag_query_rewrite": True,
    "self_rag_relevance_filter": True,
    "self_rag_answer_grading": True,
    "co_rag_enabled": False,
    "co_rag_agent_semantic": True,
    "co_rag_agent_keyword": True,
    "co_rag_agent_conceptual": True,
    "co_rag_merge_strategy": "voting",
    "active_file_filter": []
}

# --- PERSISTENCE HELPERS ---

def save_processed_files(files_info: list):
    try:
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        with open(_FILES_PATH, "w", encoding="utf-8") as f:
            json.dump(files_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Lỗi khi lưu processed_files: {e}")

def load_processed_files() -> list:
    if not os.path.exists(_FILES_PATH):
        return []
    try:
        with open(_FILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi tải processed_files: {e}")
        return []

def save_chat_history(history: list):
    try:
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        safe_history = []
        for msg in history:
            safe_msg = {
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "sources": msg.get("sources", []),
                "question_ctx": msg.get("question_ctx", ""),
                "answer_ctx": msg.get("answer_ctx", ""),
                "self_rag_meta": msg.get("self_rag_meta"),
                "co_rag_meta": msg.get("co_rag_meta"),
                "timestamp": msg.get("timestamp", time.time())
            }
            safe_history.append(safe_msg)
        with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(safe_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Lỗi khi lưu chat_history: {e}")

def load_chat_history() -> list:
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi tải chat_history: {e}")
        return []

def save_app_config():
    try:
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        config_data = {
            "chunk_size": state["chunk_size"],
            "chunk_overlap": state["chunk_overlap"],
            "hybrid_enabled": state["hybrid_enabled"],
            "reranker_enabled": state["reranker_enabled"],
            "self_rag_enabled": state["self_rag_enabled"],
            "self_rag_query_rewrite": state["self_rag_query_rewrite"],
            "self_rag_relevance_filter": state["self_rag_relevance_filter"],
            "self_rag_answer_grading": state["self_rag_answer_grading"],
            "co_rag_enabled": state["co_rag_enabled"],
            "co_rag_agent_semantic": state["co_rag_agent_semantic"],
            "co_rag_agent_keyword": state["co_rag_agent_keyword"],
            "co_rag_agent_conceptual": state["co_rag_agent_conceptual"],
            "co_rag_merge_strategy": state["co_rag_merge_strategy"],
            "active_file_filter": state["active_file_filter"]
        }
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Lỗi khi lưu config: {e}")

def load_app_config():
    if not os.path.exists(_CONFIG_PATH):
        return
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            for k, v in config_data.items():
                if k in state:
                    state[k] = v
    except Exception as e:
        logger.error(f"Lỗi khi tải config: {e}")

# --- INIT STATE ---

@app.on_event("startup")
def startup_event():
    logger.info("Khởi động API server SmartDocAI...")
    # Tải cấu hình cài đặt
    load_app_config()
    
    # Kiểm tra Ollama
    state["ollama_status"] = check_ollama_connection()
    logger.info(f"Trạng thái kết nối Ollama: {state['ollama_status']}")

    # Tải Vector Store từ disk
    saved_store = load_vector_store()
    if saved_store is not None:
        state["vector_store"] = saved_store
        logger.info("Đã tải vector store từ disk.")

        # Khôi phục raw_documents từ FAISS docstore
        try:
            docstore_dict = saved_store.docstore._dict
            if docstore_dict:
                state["raw_documents"] = list(docstore_dict.values())
                logger.info(f"Đã khôi phục {len(state['raw_documents'])} chunks từ FAISS docstore.")
        except Exception as e:
            logger.warning(f"Không thể khôi phục raw_documents từ docstore: {e}")

        # Tải danh sách file đã xử lý
        saved_files = load_processed_files()
        if saved_files:
            state["processed_files"] = saved_files
            state["total_chunks"] = sum(f.get("chunks", 0) for f in saved_files)
            logger.info(f"Đã khôi phục {len(saved_files)} file, {state['total_chunks']} chunks từ disk.")

    # Tải lịch sử chat
    saved_history = load_chat_history()
    if saved_history:
        state["chat_history"] = saved_history
        logger.info(f"Đã khôi phục {len(saved_history)} tin nhắn từ disk.")

# --- UTILITIES ---

def scan_docs_by_question_numbers(query: str, raw_documents: list) -> list:
    import re as _re
    if not raw_documents:
        return []

    STRUCT_KEYWORDS = [
        'câu', 'chương', 'mục', 'tiểu mục', 'phần', 'bài', 'đề', 'ví dụ',
        'chapter', 'section', 'part', 'exercise', 'question',
    ]
    patterns_to_search = []

    range_pat = _re.compile(
        r'(' + '|'.join(STRUCT_KEYWORDS) + r')\s+([\d\.]+)\s*(?:đến|tới|to|-)\s*([\d\.]+)',
        _re.IGNORECASE
    )
    for m in range_pat.finditer(query):
        keyword = m.group(1)
        try:
            start, end = int(float(m.group(2))), int(float(m.group(3)))
            for n in range(start, end + 1):
                patterns_to_search.append(_re.compile(
                    rf'(?:{_re.escape(keyword)}\s+{n}\b)',
                    _re.IGNORECASE
                ))
        except ValueError:
            pass

    single_pat = _re.compile(
        r'(' + '|'.join(STRUCT_KEYWORDS) + r')\s+([\dIVXivx\.]+[a-zA-Z]?)',
        _re.IGNORECASE
    )
    for m in single_pat.finditer(query):
        keyword, val = m.group(1), m.group(2)
        patterns_to_search.append(_re.compile(
            rf'(?:{_re.escape(keyword)}\s+{_re.escape(val)}\b)',
            _re.IGNORECASE
        ))

    if not patterns_to_search:
        return []

    matched = []
    seen = set()
    for doc in raw_documents:
        text = doc.page_content
        for pat in patterns_to_search:
            if pat.search(text):
                key = text[:120]
                if key not in seen:
                    seen.add(key)
                    matched.append(doc)
                break

    return matched[:10]

# --- REQUEST SCHEMAS ---

class ConfigUpdate(BaseModel):
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    hybrid_enabled: Optional[bool] = None
    reranker_enabled: Optional[bool] = None
    self_rag_enabled: Optional[bool] = None
    self_rag_query_rewrite: Optional[bool] = None
    self_rag_relevance_filter: Optional[bool] = None
    self_rag_answer_grading: Optional[bool] = None
    co_rag_enabled: Optional[bool] = None
    co_rag_agent_semantic: Optional[bool] = None
    co_rag_agent_keyword: Optional[bool] = None
    co_rag_agent_conceptual: Optional[bool] = None
    co_rag_merge_strategy: Optional[str] = None
    active_file_filter: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str

# --- API ENDPOINTS ---

@app.get("/api/status")
def get_status():
    state["ollama_status"] = check_ollama_connection()
    return {
        "ollama_status": state["ollama_status"],
        "ollama_model": config.OLLAMA_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "total_files": len(state["processed_files"]),
        "total_pages": sum(f.get("pages", 0) for f in state["processed_files"]),
        "total_chunks": state["total_chunks"]
    }

@app.get("/api/config")
def get_config():
    return {
        "chunk_size": state["chunk_size"],
        "chunk_overlap": state["chunk_overlap"],
        "hybrid_enabled": state["hybrid_enabled"],
        "reranker_enabled": state["reranker_enabled"],
        "self_rag_enabled": state["self_rag_enabled"],
        "self_rag_query_rewrite": state["self_rag_query_rewrite"],
        "self_rag_relevance_filter": state["self_rag_relevance_filter"],
        "self_rag_answer_grading": state["self_rag_answer_grading"],
        "co_rag_enabled": state["co_rag_enabled"],
        "co_rag_agent_semantic": state["co_rag_agent_semantic"],
        "co_rag_agent_keyword": state["co_rag_agent_keyword"],
        "co_rag_agent_conceptual": state["co_rag_agent_conceptual"],
        "co_rag_merge_strategy": state["co_rag_merge_strategy"],
        "active_file_filter": state["active_file_filter"]
    }

@app.post("/api/config")
def update_config(data: ConfigUpdate):
    for key, value in data.dict(exclude_unset=True).items():
        if key in state:
            state[key] = value
    save_app_config()
    return {"status": "success", "config": get_config()}

@app.get("/api/files")
def get_files():
    return state["processed_files"]

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Không nhận được file nào.")

    # Reset vector store và cache nếu upload đợt mới (Giống Streamlit app.py)
    clear_vector_store()
    state["vector_store"] = None
    state["raw_documents"] = []
    state["processed_files"] = []
    state["total_chunks"] = 0

    all_chunks = []
    new_files_info = []

    for uf in files:
        file_name = uf.filename
        _, file_ext = os.path.splitext(file_name)
        file_ext = file_ext.lower() if file_ext else ".pdf"

        if file_ext not in [".pdf", ".docx"]:
            continue

        try:
            # Lưu file tạm với đúng phần mở rộng
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                shutil.copyfileobj(uf.file, tmp)
                tmp_path = tmp.name

            # Trích xuất văn bản
            if file_ext == ".docx":
                raw_docs = extract_text_from_docx(tmp_path, source_name=file_name)
            else:
                raw_docs = extract_text_from_pdf(tmp_path, source_name=file_name)
            num_pages = len(raw_docs)

            # Chia nhỏ
            chunks = split_documents(
                raw_docs,
                chunk_size=state["chunk_size"],
                chunk_overlap=state["chunk_overlap"],
            )

            if chunks:
                all_chunks.extend(chunks)
                new_files_info.append(
                    {"name": file_name, "chunks": len(chunks), "pages": num_pages}
                )
                logger.info(f"Đã xử lý '{file_name}': {num_pages} trang -> {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý '{file_name}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi xử lý file {file_name}: {str(e)}")
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if all_chunks:
        try:
            # Tạo Vector Store
            state["vector_store"] = create_vector_store(all_chunks)
            save_vector_store(state["vector_store"])

            # Cập nhật state
            state["processed_files"].extend(new_files_info)
            state["total_chunks"] += len(all_chunks)
            state["raw_documents"].extend(all_chunks)

            # Lưu processed_files
            save_processed_files(state["processed_files"])

        except Exception as e:
            logger.error(f"Lỗi khi tạo vector store: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Lỗi tạo vector index: {str(e)}")

    return {
        "status": "success",
        "processed_files": state["processed_files"],
        "total_chunks": state["total_chunks"]
    }

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Nội dung tin nhắn trống.")

    # Lưu câu hỏi của người dùng vào history
    user_msg = {"role": "user", "content": user_input, "timestamp": time.time()}
    state["chat_history"].append(user_msg)
    save_chat_history(state["chat_history"])

    # Bắt đầu xử lý phản hồi từ AI
    self_rag_meta = None
    co_rag_meta = None
    result = None

    try:
        # 1. Pipeline: Self-RAG
        if state["self_rag_enabled"] and state["vector_store"] is not None:
            llm = get_llm()
            result = self_rag_pipeline(
                question=user_input,
                vector_store=state["vector_store"],
                llm=llm,
                enable_query_rewrite=state["self_rag_query_rewrite"],
                enable_relevance_filter=state["self_rag_relevance_filter"],
                enable_answer_grading=state["self_rag_answer_grading"],
            )
            result["search_mode"] = "self_rag"
            result["active_filter"] = state["active_file_filter"]
            self_rag_meta = result

        # 2. Pipeline: Co-RAG (Multi-Agent)
        elif state["co_rag_enabled"] and state["vector_store"] is not None:
            llm = get_llm()
            result = co_rag_pipeline(
                question=user_input,
                vector_store=state["vector_store"],
                raw_documents=state["raw_documents"],
                llm=llm,
                min_votes=config.CO_RAG_MIN_VOTES,
                merge_strategy=state["co_rag_merge_strategy"],
                enable_agent_semantic=state["co_rag_agent_semantic"],
                enable_agent_keyword=state["co_rag_agent_keyword"],
                enable_agent_conceptual=state["co_rag_agent_conceptual"],
            )
            result["search_mode"] = "co_rag"
            result["active_filter"] = state["active_file_filter"]
            co_rag_meta = result

        # 3. Pipeline: RAG chuẩn / Hybrid / Reranked
        else:
            retriever = None
            if state["hybrid_enabled"] and state["vector_store"] is not None:
                bm25 = get_cached_bm25_retriever()
                if bm25 is None and state["raw_documents"]:
                    bm25 = create_bm25_retriever(state["raw_documents"])
                if bm25 is not None:
                    retriever = create_ensemble_retriever(state["vector_store"], bm25)

            forced_docs = scan_docs_by_question_numbers(
                user_input, state["raw_documents"]
            )

            result = ask_question(
                question=user_input,
                vector_store=state["vector_store"],
                chat_history=state["chat_history"],
                retriever=retriever,
                file_filter=state["active_file_filter"],
                forced_docs=forced_docs if forced_docs else None,
            )

            # Reranking
            if state["reranker_enabled"] and state["vector_store"] is not None:
                if retriever is not None:
                    doc_score_pairs = _compute_rrf_scores(retriever, user_input)
                else:
                    doc_score_pairs = state["vector_store"].similarity_search_with_score(user_input)
                
                if doc_score_pairs:
                    reranked = rerank_with_cross_encoder(user_input, doc_score_pairs, top_k=3)
                    reranked_sources = []
                    for idx, (doc, bi_score, ce_score) in enumerate(reranked):
                        fname = os.path.basename(str(doc.metadata.get("source", "N/A")))
                        reranked_sources.append({
                            "file": fname,
                            "page": doc.metadata.get("page", "N/A"),
                            "total_pages": doc.metadata.get("total_pages"),
                            "file_type": doc.metadata.get("file_type", "pdf"),
                            "content": doc.page_content,
                            "chunk_index": idx + 1,
                            "score": float(ce_score),
                            "bi_encoder_score": float(bi_score),
                        })
                    result["sources"] = reranked_sources
                    result["search_mode"] = result.get("search_mode", "vector") + "+reranked"

        # Định dạng cấu trúc trả về
        mode = result.get("search_mode", "vector")
        active_filter = result.get("active_filter", [])
        
        # Làm sạch sources để an toàn khi chuyển json
        sources_clean = []
        for s in result.get("sources", []):
            sources_clean.append({
                "file": s.get("file", ""),
                "page": s.get("page", "?"),
                "total_pages": s.get("total_pages"),
                "file_type": s.get("file_type", "pdf"),
                "content": s.get("content", ""),
                "chunk_index": s.get("chunk_index", ""),
                "score": float(s.get("score", 0.0))
            })

        # Chuẩn hóa Self-RAG Metadata sang JSON-serializable
        self_rag_data = None
        if self_rag_meta:
            self_rag_data = {
                "confidence_score": float(self_rag_meta.get("confidence_score", 0.5)),
                "is_grounded": bool(self_rag_meta.get("is_grounded", True)),
                "has_hallucination": bool(self_rag_meta.get("has_hallucination", False)),
                "grading_feedback": str(self_rag_meta.get("grading_feedback", "")),
                "rewritten_queries": self_rag_meta.get("rewritten_queries", []),
                "docs_before_filter": int(self_rag_meta.get("docs_before_filter", 0)),
                "docs_after_filter": int(self_rag_meta.get("docs_after_filter", 0)),
                "sub_questions": self_rag_meta.get("sub_questions", []),
                "used_multihop": bool(self_rag_meta.get("used_multihop", False))
            }

        # Chuẩn hóa Co-RAG Metadata
        co_rag_data = None
        if co_rag_meta:
            co_rag_data = {
                "co_rag_agent_counts": {str(k): int(v) for k, v in co_rag_meta.get("co_rag_agent_counts", {}).items()},
                "co_rag_total_before_merge": int(co_rag_meta.get("co_rag_total_before_merge", 0)),
                "co_rag_total_after_merge": int(co_rag_meta.get("co_rag_total_after_merge", 0)),
                "co_rag_merge_strategy": str(co_rag_meta.get("co_rag_merge_strategy", "voting"))
            }

        assistant_msg = {
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": sources_clean,
            "question_ctx": user_input,
            "answer_ctx": result.get("answer", ""),
            "self_rag_meta": self_rag_data,
            "co_rag_meta": co_rag_data,
            "timestamp": time.time(),
            "search_mode": mode,
            "active_filter": active_filter,
            "error": result.get("error"),
            "used_fallback": result.get("used_fallback", False)
        }

        state["chat_history"].append(assistant_msg)
        save_chat_history(state["chat_history"])

        return assistant_msg

    except Exception as e:
        logger.error(f"Lỗi trong endpoint chat: {e}", exc_info=True)
        # Rollback message của user nếu lỗi để tránh mất đồng bộ
        if state["chat_history"] and state["chat_history"][-1]["role"] == "user":
            state["chat_history"].pop()
            save_chat_history(state["chat_history"])
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý câu hỏi: {str(e)}")

@app.get("/api/history")
def get_history():
    return state["chat_history"]

@app.post("/api/clear-history")
def clear_history():
    state["chat_history"] = []
    try:
        if os.path.exists(_HISTORY_PATH):
            os.remove(_HISTORY_PATH)
    except Exception as e:
        logger.error(f"Lỗi khi xóa chat_history: {e}")
        raise HTTPException(status_code=500, detail=f"Không thể xóa file lịch sử: {e}")
    return {"status": "success"}

@app.post("/api/clear-documents")
def clear_documents():
    clear_vector_store()
    
    # Xóa file persist
    for path in [_FILES_PATH, _HISTORY_PATH, _CONFIG_PATH]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.error(f"Lỗi khi xóa {path}: {e}")

    state["vector_store"] = None
    state["processed_files"] = []
    state["total_chunks"] = 0
    state["raw_documents"] = []
    state["chat_history"] = []
    state["active_file_filter"] = []
    
    return {"status": "success"}

# Phục vụ file tĩnh của frontend React ở root
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_api.py:app", host="127.0.0.1", port=8000, reload=True)

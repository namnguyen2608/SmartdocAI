# -*- coding: utf-8 -*-
"""
SmartDocAI - Streamlit Application
Giao diện chatbot hỏi đáp tài liệu PDF với RAG
Phiên bản 2.0 — Thiết kế lại hoàn toàn
"""

import os
import sys
import json
import time
import logging
import tempfile
import streamlit as st

# Thêm root directory vào path
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
    create_bm25_retriever,       # Q7
    create_ensemble_retriever,   # Q7
    get_cached_bm25_retriever,   # Q7
)
from modules.rag_chain import ask_question, check_ollama_connection, get_llm
from modules.reranker import rerank_with_cross_encoder
from modules.self_rag import self_rag_pipeline
from modules.co_rag import co_rag_pipeline

# ============================================================
# Cấu hình logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Cấu hình trang Streamlit
# ============================================================
st.set_page_config(
    page_title="SmartDocAI — Trợ lý Tài liệu Thông minh",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Thiết kế hiện đại
# ============================================================
st.markdown(
    """
<style>
    /* ── Google Fonts — Clean Design System ── */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&family=Poppins:wght@500;600;700;800&family=Inconsolata:wght@400;600&display=swap');

    /* ── CSS Variables — Clean Light Theme ── */
    :root {
        --bg-primary:    #FFFFFF;
        --bg-secondary:  #F9FAFB;
        --bg-surface:    #F3F4F6;
        --bg-elevated:   #FFFFFF;
        --bg-hover:      #F3F4F6;
        --border-subtle: #E5E7EB;
        --border-default:#D1D5DB;
        --text-primary:  #111827;
        --text-secondary:#4B5563;
        --text-muted:    #9CA3AF;
        --accent:        #3B82F6;
        --accent-hover:  #2563EB;
        --accent-soft:   rgba(59, 130, 246, 0.08);
        --accent-border: rgba(59, 130, 246, 0.20);
        --success:       #16A34A;
        --success-soft:  rgba(22, 163, 74, 0.08);
        --success-border:rgba(22, 163, 74, 0.20);
        --error:         #DC2626;
        --error-soft:    rgba(220, 38, 38, 0.08);
        --error-border:  rgba(220, 38, 38, 0.20);
        --warning:       #D97706;
        --warning-soft:  rgba(217, 119, 6, 0.08);
        --radius-sm:     6px;
        --radius-md:     8px;
        --radius-lg:     12px;
        --radius-xl:     16px;
        --shadow-sm:     0 1px 2px rgba(0,0,0,0.05);
        --shadow-md:     0 2px 8px rgba(0,0,0,0.08);
        --shadow-lg:     0 4px 16px rgba(0,0,0,0.10);
        --transition:    all 0.2s ease;
    }

    /* ── Global ── */
    .stApp {
        font-family: 'Roboto', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    .main .block-container {
        padding: 2rem 2.5rem 8rem 2.5rem;
        max-width: 100%;
    }
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.5rem 1.25rem;
    }

    /* ── Sidebar Brand ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
        border-radius: var(--radius-lg);
        margin-bottom: 20px;
    }
    .sidebar-brand .brand-icon {
        width: 40px; height: 40px;
        border-radius: var(--radius-md);
        background: var(--accent);
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #FFFFFF;
    }
    .sidebar-brand .brand-text h3 {
        margin: 0; font-size: 1rem; font-weight: 700;
        font-family: 'Poppins', sans-serif;
        color: var(--text-primary); line-height: 1.2;
    }
    .sidebar-brand .brand-text p {
        margin: 2px 0 0; font-size: 0.75rem;
        color: var(--text-muted); line-height: 1.3;
    }

    /* ── Status Badge ── */
    .status-badge {
        display: flex; align-items: center; gap: 8px;
        padding: 10px 14px; border-radius: var(--radius-md);
        font-size: 0.8rem; font-weight: 600;
        margin-bottom: 16px;
        transition: var(--transition);
    }
    .status-online {
        background: var(--success-soft);
        color: var(--success);
        border: 1px solid var(--success-border);
    }
    .status-online .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 6px var(--success);
    }
    .status-offline {
        background: var(--error-soft);
        color: var(--error);
        border: 1px solid var(--error-border);
    }
    .status-offline .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--error);
    }



    /* ── Section Headers ── */
    .section-header {
        font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: var(--text-muted);
        padding: 0 4px;
        margin: 18px 0 8px;
    }

    /* ── KPI Row ── */
    .kpi-row {
        display: flex; gap: 6px;
        margin-bottom: 16px; flex-wrap: wrap;
    }
    .kpi-item {
        flex: 1; min-width: 70px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 8px;
        text-align: center;
        transition: var(--transition);
    }
    .kpi-item:hover { border-color: var(--accent-border); }
    .kpi-item .kpi-value {
        font-size: 1.15rem; font-weight: 700;
        color: var(--accent);
        line-height: 1;
    }
    .kpi-item .kpi-label {
        font-size: 0.65rem; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* ── File List ── */
    .file-card {
        display: flex; align-items: center; gap: 10px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 12px;
        margin-bottom: 6px;
        transition: var(--transition);
    }
    .file-card:hover {
        background: var(--bg-hover);
        border-color: var(--border-default);
    }
    .file-card .file-icon {
        width: 34px; height: 34px;
        border-radius: var(--radius-sm);
        background: var(--accent-soft);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.9rem; flex-shrink: 0;
    }
    .file-card .file-info { flex: 1; min-width: 0; }
    .file-card .file-name {
        font-size: 0.8rem; font-weight: 600;
        color: var(--text-primary);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .file-card .file-meta {
        font-size: 0.68rem; color: var(--text-muted); margin-top: 2px;
    }
    .file-card .file-badge {
        font-size: 0.65rem; font-weight: 700;
        color: var(--accent);
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
        border-radius: 999px;
        padding: 2px 8px; white-space: nowrap;
    }

    /* ── Upload Area ── */
    [data-testid="stFileUploader"] > div:first-child {
        background: var(--bg-elevated) !important;
        border: 2px dashed var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        transition: var(--transition) !important;
    }
    [data-testid="stFileUploader"] > div:first-child:hover {
        border-color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }
    [data-testid="stFileUploader"] label { display: none !important; }

    /* ── Progress Bar ── */
    [data-testid="stProgress"] > div > div {
        background: var(--accent) !important;
        border-radius: 5px !important;
    }

    /* ── Welcome Hero ── */
    .welcome-hero {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        text-align: center;
        padding: 1.5rem 2rem 1rem;
        min-height: calc(100vh - 380px);
    }
    .welcome-icon {
        width: 64px; height: 64px;
        border-radius: 20px;
        background: var(--accent);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 14px;
    }

    .welcome-hero h1 {
        font-size: 1.5rem; font-weight: 700;
        font-family: 'Poppins', sans-serif;
        margin: 0 0 6px;
        color: var(--text-primary);
    }
    .welcome-hero p {
        font-size: 0.88rem; color: var(--text-secondary);
        max-width: 480px; line-height: 1.5;
        margin: 0 0 16px;
    }
    .welcome-steps {
        display: flex; gap: 8px; flex-wrap: wrap;
        justify-content: center; margin-bottom: 16px;
    }
    .welcome-step {
        display: flex; align-items: center; gap: 8px;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 16px;
        font-size: 0.82rem;
        color: var(--text-secondary);
        transition: var(--transition);
    }
    .welcome-step:hover {
        border-color: var(--accent-border);
        background: var(--bg-hover);
    }
    .welcome-step .step-num {
        width: 22px; height: 22px;
        border-radius: 50%;
        background: var(--accent-soft);
        color: var(--accent);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700;
        flex-shrink: 0;
    }

    /* ── Chat Messages ── */
    [data-testid="stChatMessage"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: var(--shadow-sm) !important;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Source Tags ── */
    .source-tag {
        display: inline-flex; align-items: center; gap: 4px;
        background: var(--accent-soft);
        color: var(--accent-hover);
        border: 1px solid var(--accent-border);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem; font-weight: 600;
        margin: 3px 4px 3px 0;
        transition: var(--transition);
    }
    .source-tag:hover { background: rgba(108,140,255,0.20); }

    /* ── Processing Toast ── */
    .processing-toast {
        display: flex; align-items: center; gap: 10px;
        padding: 12px 16px;
        background: var(--accent-soft);
        border: 1px solid var(--accent-border);
        border-radius: var(--radius-md);
        color: var(--accent-hover);
        font-size: 0.85rem; font-weight: 600;
        margin: 8px 0;
        animation: slideIn 0.3s ease-out;
    }
    .success-toast {
        display: flex; align-items: center; gap: 10px;
        padding: 12px 16px;
        background: var(--success-soft);
        border: 1px solid var(--success-border);
        border-radius: var(--radius-md);
        color: var(--success);
        font-size: 0.85rem; font-weight: 600;
    }

    /* ── stBottom: compact padding ── */
    [data-testid="stBottom"] {
        padding-top: 0 !important;
        padding-bottom: 0.5rem !important;
    }
    [data-testid="stBottom"] > div {
        padding-top: 0 !important;
    }
    [data-testid="stChatInput"] {
        border-radius: px !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        background: var(--bg-elevated) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        font-family: 'Roboto', sans-serif !important;
        box-shadow: none !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        font-family: 'Roboto', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        transition: var(--transition) !important;
        padding: 8px 16px !important;
    }
    .stButton > button:hover {
        background: var(--bg-hover) !important;
        border-color: var(--accent) !important;
        color: var(--accent-hover) !important;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* ── Toggle ── */
    [data-testid="stToggle"] label span {
        font-size: 0.82rem !important;
        color: var(--text-secondary) !important;
    }

    /* ── Divider ── */
    .sidebar-divider {
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 8px 0;
    }

    /* ── Empty file list ── */
    .empty-files {
        text-align: center;
        padding: 16px;
        color: var(--text-muted);
        font-size: 0.8rem;
        border: 1px dashed var(--border-default);
        border-radius: var(--radius-md);
    }

    /* ── Chat History Sidebar ── */
    .history-list {
        max-height: 280px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding-right: 2px;
    }
    .history-item {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 12px;
        cursor: default;
        transition: var(--transition);
    }
    .history-item:hover {
        background: var(--bg-hover);
        border-color: var(--border-default);
    }
    .history-question {
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-primary);
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.4;
    }
    .history-answer-preview {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-top: 4px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        line-height: 1.35;
    }
    .history-time {
        font-size: 0.62rem;
        color: var(--text-muted);
        margin-top: 4px;
        opacity: 0.7;
    }
    .empty-history {
        text-align: center;
        padding: 16px;
        color: var(--text-muted);
        font-size: 0.8rem;
        border: 1px dashed var(--border-default);
        border-radius: var(--radius-md);
    }

    /* ── History Button items ── */
    .history-btn-item {
        width: 100%;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 10px 12px;
        margin-bottom: 5px;
        cursor: pointer;
        text-align: left;
        transition: var(--transition);
    }
    .history-btn-item:hover {
        background: var(--accent-soft);
        border-color: var(--accent-border);
    }

    /* ── Confirmation Dialog ── */
    .confirm-dialog {
        background: var(--bg-elevated);
        border: 1px solid var(--error-border);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        margin: 8px 0;
        animation: slideIn 0.25s ease-out;
    }
    .confirm-dialog .confirm-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--error);
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .confirm-dialog .confirm-message {
        font-size: 0.75rem;
        color: var(--text-secondary);
        line-height: 1.45;
        margin-bottom: 10px;
    }

    /* ── Danger Button ── */
    .danger-btn button {
        background: var(--error-soft) !important;
        border: 1px solid var(--error-border) !important;
        color: var(--error) !important;
    }
    .danger-btn button:hover {
        background: rgba(240, 110, 110, 0.25) !important;
        border-color: var(--error) !important;
    }

    /* ── Citation Panel ── */
    .citation-panel {
        margin-top: 14px;
        border-top: 1px solid var(--border-subtle);
        padding-top: 12px;
    }
    .citation-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 8px;
    }
    .citation-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: var(--accent-soft);
        color: var(--accent-hover);
        border: 1px solid var(--accent-border);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        transition: var(--transition);
        white-space: nowrap;
    }
    .citation-badge:hover { background: rgba(108,140,255,0.22); }
    .citation-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    .citation-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: var(--accent);
        border-radius: 999px 0 0 999px;
    }
    .citation-card:hover {
        border-color: var(--accent-border);
        background: var(--bg-hover);
    }
    .citation-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        flex-wrap: wrap;
        gap: 6px;
    }
    .citation-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .citation-meta {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .citation-page-badge {
        font-size: 0.68rem;
        font-weight: 700;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--accent-border);
        border-radius: var(--radius-sm);
        padding: 2px 8px;
    }
    .citation-type-badge {
        font-size: 0.65rem;
        font-weight: 700;
        background: var(--bg-surface);
        color: var(--text-muted);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 2px 7px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .citation-score-wrap {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 10px;
    }
    .citation-score-label {
        font-size: 0.67rem;
        color: var(--text-muted);
        white-space: nowrap;
        min-width: 72px;
    }
    .citation-score-bar-bg {
        flex: 1;
        height: 5px;
        background: var(--border-subtle);
        border-radius: 999px;
        overflow: hidden;
    }
    .citation-score-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--accent);
        transition: width 0.5s ease;
    }
    .citation-score-value {
        font-size: 0.67rem;
        font-weight: 700;
        color: var(--accent);
        min-width: 32px;
        text-align: right;
    }
    .citation-content {
        font-size: 0.78rem;
        color: var(--text-secondary);
        line-height: 1.65;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 10px 12px;
        max-height: 160px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .citation-content mark {
        background: rgba(59, 130, 246, 0.15);
        color: var(--accent-hover);
        border-radius: 3px;
        padding: 1px 2px;
        font-weight: 600;
    }
    .citation-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px; height: 20px;
        background: var(--accent-soft);
        color: var(--accent);
        border-radius: 50%;
        font-size: 0.65rem;
        font-weight: 800;
        flex-shrink: 0;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Persistence Helpers — lưu/tải state ra disk
# ============================================================
_PERSIST_DIR  = config.VECTORSTORE_DIR
_FILES_PATH   = os.path.join(_PERSIST_DIR, "processed_files.json")
_HISTORY_PATH = os.path.join(_PERSIST_DIR, "chat_history.json")


def save_processed_files(files_info: list):
    """Lưu danh sách file đã xử lý ra disk."""
    try:
        with open(_FILES_PATH, "w", encoding="utf-8") as f:
            json.dump(files_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Lỗi khi lưu processed_files: {e}")


def load_processed_files() -> list:
    """Tải danh sách file đã xử lý từ disk."""
    if not os.path.exists(_FILES_PATH):
        return []
    try:
        with open(_FILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi tải processed_files: {e}")
        return []


def save_chat_history(history: list):
    """Lưu lịch sử chat ra disk (chỉ giữ role/content/sources — bỏ objects không JSON-able)."""
    try:
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
            }
            safe_history.append(safe_msg)
        with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(safe_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Lỗi khi lưu chat_history: {e}")


def load_chat_history() -> list:
    """Tải lịch sử chat từ disk."""
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Lỗi khi tải chat_history: {e}")
        return []


def clear_persist_data():
    """Xóa toàn bộ dữ liệu persist (files + history)."""
    for path in [_FILES_PATH, _HISTORY_PATH]:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.error(f"Lỗi khi xóa {path}: {e}")


def clear_history_persist():
    """Xóa chỉ file lịch sử chat."""
    try:
        if os.path.exists(_HISTORY_PATH):
            os.remove(_HISTORY_PATH)
    except Exception as e:
        logger.error(f"Lỗi khi xóa chat_history.json: {e}")


# ============================================================
# Session State Initialization
# ============================================================
def init_session_state():
    """Khởi tạo session state cho Streamlit."""
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "processed_files": [],
        "total_chunks": 0,
        "ollama_status": None,
        "is_processing": False,
        "auto_process_upload": True,
        "last_processed_upload_signature": "",
        # Q4 — Chunk Parameters
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        # History viewer
        "selected_history_idx": None,
        # Q7 — Hybrid Search
        "hybrid_enabled": False,
        "raw_documents": [],        # toàn bộ chunks để xây BM25
        # Q8 — Metadata Filtering
        "active_file_filter": [],   # danh sách file đang được lọc
        "reranker_enabled": False,
        "self_rag_enabled": False,
        "self_rag_query_rewrite": True,
        "self_rag_relevance_filter": True,
        "self_rag_answer_grading": True,
        # Co-RAG
        "co_rag_enabled": False,
        "co_rag_agent_semantic": True,
        "co_rag_agent_keyword": True,
        "co_rag_agent_conceptual": True,
        "co_rag_merge_strategy": "voting",
        # Flag: đã restore từ disk chưa (tránh load nhiều lần)
        "_state_restored": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """Render sidebar với đầy đủ chức năng: brand, status, upload, files, actions."""
    with st.sidebar:
        # ── Brand ──
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">S</div>
                <div class="brand-text">
                    <h3>SmartDocAI</h3>
                    <p>Trợ lý Tài liệu Thông minh</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Trạng thái hệ thống ──
        render_system_status()

        # ── KPI ──
        total_files = len(st.session_state.processed_files)
        total_pages = sum(f.get("pages", 0) for f in st.session_state.processed_files)
        total_chunks = st.session_state.total_chunks

        st.markdown(
            f"""
            <div class="kpi-row">
                <div class="kpi-item">
                    <div class="kpi-value">{total_files}</div>
                    <div class="kpi-label">Tài liệu</div>
                </div>
                <div class="kpi-item">
                    <div class="kpi-value">{total_pages}</div>
                    <div class="kpi-label">Trang</div>
                </div>
                <div class="kpi-item">
                    <div class="kpi-value">{total_chunks}</div>
                    <div class="kpi-label">Chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Chunk Parameters ──
        st.markdown('<div class="section-header">Cài đặt Chunking (Q4)</div>', unsafe_allow_html=True)
        render_chunk_settings()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Upload ──
        st.markdown('<div class="section-header">Tải tài liệu lên</div>', unsafe_allow_html=True)
        render_upload_section()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Danh sách file ──
        st.markdown('<div class="section-header">Tài liệu đã xử lý</div>', unsafe_allow_html=True)
        render_file_list()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Actions ──
        st.markdown('<div class="section-header">Thao tác</div>', unsafe_allow_html=True)
        render_action_buttons()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Lịch sử hội thoại ──
        st.markdown('<div class="section-header">Lịch sử hội thoại</div>', unsafe_allow_html=True)
        render_chat_history_sidebar()


# ── Sidebar helpers ──────────────────────────────────────────────────────────

CHUNK_SIZE_OPTIONS   = [500, 1000, 1500, 2000]
CHUNK_OVERLAP_OPTIONS = [50, 100, 200]


def render_chunk_settings():
    """Q4 — Cho phép người dùng tùy chỉnh chunk_size và chunk_overlap."""
    current_size    = st.session_state.chunk_size
    current_overlap = st.session_state.chunk_overlap

    # Đảm bảo giá trị hiện tại nằm trong danh sách option
    size_idx    = CHUNK_SIZE_OPTIONS.index(current_size) if current_size in CHUNK_SIZE_OPTIONS else 2
    overlap_idx = CHUNK_OVERLAP_OPTIONS.index(current_overlap) if current_overlap in CHUNK_OVERLAP_OPTIONS else 1

    new_size = st.selectbox(
        "Chunk Size (ký tự)",
        options=CHUNK_SIZE_OPTIONS,
        index=size_idx,
        key="chunk_size_select",
        help=(
            "Số ký tự tối đa trong mỗi chunk.\n"
            "• 500 — chi tiết, nhiều chunk hơn\n"
            "• 1000 — cân bằng\n"
            "• 1500 — mặc định\n"
            "• 2000 — ngữ cảnh rộng hơn"
        ),
    )

    new_overlap = st.selectbox(
        "Chunk Overlap (ký tự)",
        options=CHUNK_OVERLAP_OPTIONS,
        index=overlap_idx,
        key="chunk_overlap_select",
        help=(
            "Số ký tự chồng lấp giữa các chunk liền kề.\n"
            "Overlap lớn hơn giúp không mất ngữ cảnh ở ranh giới chunk."
        ),
    )

    # Phát hiện thay đổi
    changed = (new_size != current_size) or (new_overlap != current_overlap)
    if changed:
        st.session_state.chunk_size    = new_size
        st.session_state.chunk_overlap = new_overlap

    # Badge hiển thị cấu hình hiện tại
    st.markdown(
        f"""
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;">
            <span style="font-size:0.7rem;font-weight:700;
                background:var(--accent-soft);color:var(--accent);
                border:1px solid var(--accent-border);border-radius:999px;
                padding:2px 10px;">
                Size: {st.session_state.chunk_size}
            </span>
            <span style="font-size:0.7rem;font-weight:700;
                background:var(--accent-soft);color:var(--accent);
                border:1px solid var(--accent-border);border-radius:999px;
                padding:2px 10px;">
                Overlap: {st.session_state.chunk_overlap}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if changed and st.session_state.processed_files:
        st.caption("⚠️ Tải lại tài liệu để áp dụng cài đặt mới.")


def render_system_status():
    """Kiểm tra và hiển thị trạng thái Ollama."""
    if st.session_state.ollama_status is None:
        st.session_state.ollama_status = check_ollama_connection()

    if st.session_state.ollama_status:
        st.markdown(
            f"""
            <div class="status-badge status-online">
                <div class="dot"></div>
                Ollama đang hoạt động — {config.OLLAMA_MODEL}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-badge status-offline">
                <div class="dot"></div>
                Ollama không khả dụng
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            f"Không thể kết nối tới Ollama!\n\n"
            f"1. Cài đặt Ollama: https://ollama.com\n"
            f"2. Chạy: `ollama pull {config.OLLAMA_MODEL}`\n"
            f"3. Đảm bảo Ollama đang chạy trên cổng 11434"
        )

    if st.button("Kiểm tra kết nối", key="recheck_ollama", use_container_width=True):
        st.session_state.ollama_status = check_ollama_connection()
        st.rerun()


def render_upload_section():
    """Render phần upload PDF/DOCX trong sidebar."""
    uploaded_files = st.file_uploader(
        "Chọn file PDF hoặc DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Kéo-thả hoặc nhấn để chọn nhiều file PDF / DOCX cùng lúc.",
    )
    st.caption("Hỗ trợ kéo-thả nhiều file PDF & DOCX cùng lúc")

    # Toggle auto-process
    st.toggle(
        "Tự động xử lý sau khi chọn file",
        key="auto_process_upload",
        help="Bật để tự động xử lý ngay sau khi chọn file, không cần bấm nút.",
    )

    # Tính upload signature
    upload_signature = ""
    if uploaded_files:
        signature_parts = []
        total_size_mb = 0.0
        for uf in uploaded_files:
            size_bytes = len(uf.getbuffer())
            signature_parts.append(f"{uf.name}:{size_bytes}")
            total_size_mb += size_bytes / (1024 * 1024)
        upload_signature = "|".join(signature_parts)

        # Hiển thị preview files
        for uf in uploaded_files:
            file_size = len(uf.getbuffer())
            if file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.0f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            file_icon = "DOCX" if uf.name.lower().endswith(".docx") else "PDF"

            st.markdown(
                f"""
                <div class="file-card">
                    <div class="file-icon">{file_icon}</div>
                    <div class="file-info">
                        <div class="file-name">{uf.name}</div>
                        <div class="file-meta">{size_str}</div>
                    </div>
                    <div class="file-badge">Sẵn sàng</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Nút xử lý thủ công
        btn_label = "Đang xử lý..." if st.session_state.is_processing else "Xử lý tài liệu"
        already_processed = (
            upload_signature == st.session_state.last_processed_upload_signature
        )
        if st.button(
            btn_label,
            use_container_width=True,
            type="primary",
            disabled=st.session_state.is_processing or already_processed,
            key="process_btn",
            help="Tài liệu đã được xử lý" if already_processed else None,
        ):
            if not already_processed:
                process_documents(uploaded_files, upload_signature=upload_signature)

        # Auto-process
        if (
            st.session_state.auto_process_upload
            and not st.session_state.is_processing
            and upload_signature
            and upload_signature != st.session_state.last_processed_upload_signature
        ):
            process_documents(uploaded_files, upload_signature=upload_signature)


def render_file_list():
    """Render danh sách file đã xử lý."""
    if not st.session_state.processed_files:
        st.markdown(
            """
            <div class="empty-files">
                Chưa có tài liệu nào được xử lý.<br>
                Hãy tải file PDF hoặc DOCX lên ở phía trên.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for f in st.session_state.processed_files[::-1]:
        pages = f.get("pages", "?")
        chunks = f.get("chunks", "?")
        file_icon = "DOCX" if f["name"].lower().endswith(".docx") else "PDF"
        st.markdown(
            f"""
            <div class="file-card">
                <div class="file-icon">{file_icon}</div>
                <div class="file-info">
                    <div class="file-name">{f['name']}</div>
                    <div class="file-meta">{pages} trang</div>
                </div>
                <div class="file-badge">{chunks} chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_history_sidebar():
    """Render lịch sử hội thoại trong sidebar — mỗi item là nút bấm được."""
    history = st.session_state.chat_history

    # Tích các cặp Q&A
    qa_pairs = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            question = history[i]["content"]
            answer = ""
            sources = []
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                answer = history[i + 1]["content"]
                sources = history[i + 1].get("sources", [])
                i += 2
            else:
                i += 1
            qa_pairs.append({"question": question, "answer": answer, "sources": sources})
        else:
            i += 1

    if not qa_pairs:
        st.markdown(
            """
            <div class="empty-history">
                Chưa có cuộc hội thoại nào.<br>
                Hãy đặt câu hỏi để bắt đầu.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(qa_pairs)} câu hỏi — nhấn để xem chi tiết")

    # Hiển thị danh sách (mới nhất lên đầu)
    for display_idx, pair in enumerate(reversed(qa_pairs)):
        real_idx = len(qa_pairs) - 1 - display_idx   # index gốc trong qa_pairs
        q_short = pair["question"][:60] + ("…" if len(pair["question"]) > 60 else "")
        a_short = pair["answer"][:80] + ("…" if len(pair["answer"]) > 80 else "")

        # Nút bấm — Streamlit button mới nhất ủng hộ use_container_width
        clicked = st.button(
            f"  {q_short}",
            key=f"hist_btn_{real_idx}",
            use_container_width=True,
            help=a_short,
        )
        if clicked:
            st.session_state.selected_history_idx = real_idx
            show_history_detail_dialog(qa_pairs, real_idx)


@st.dialog(" Lịch sử hội thoại", width="large")
def show_history_detail_dialog(qa_pairs: list, idx: int):
    """Dialog hiển thị chi tiết một cặp Q&A trong lịch sử."""
    pair = qa_pairs[idx]
    total = len(qa_pairs)

    # Header: số thứ tự
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:12px;">
            <span style="font-size:0.75rem;color:var(--text-muted);font-weight:600;
                         text-transform:uppercase;letter-spacing:0.08em;">
                Câu hỏi {idx + 1} / {total}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Câu hỏi
    st.markdown(
        f"""
        <div style="background:var(--accent-soft);border:1px solid var(--accent-border);
                    border-radius:var(--radius-md);padding:12px 16px;margin-bottom:12px;">
            <div style="font-size:0.68rem;font-weight:700;color:var(--accent);
                        text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">Người dùng</div>
            <div style="font-size:0.9rem;color:var(--text-primary);line-height:1.55;">
                {pair['question'].replace(chr(10), '<br>')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Câu trả lời
    st.markdown(
        """
        <div style="font-size:0.68rem;font-weight:700;color:var(--text-muted);
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">
            Trợ lý AI
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(pair["answer"] if pair["answer"] else "_Chưa có câu trả lời._")

    # Nguồn trích dẫn (nếu có)
    if pair.get("sources"):
        st.markdown("---")
        src_lines = []
        seen = set()
        for s in pair["sources"]:
            label = f"📎 **{s['file']}** — Trang {s.get('page', '?')}"
            if label not in seen:
                src_lines.append(label)
                seen.add(label)
        st.markdown("**Nguồn tham khảo:**")
        for line in src_lines:
            st.markdown(f"- {line}")

    # Nút điếu hướng giữa các câu hỏi
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
    with nav_col1:
        if idx > 0:
            if st.button("← Trước", use_container_width=True, key="hist_prev"):
                st.session_state.selected_history_idx = idx - 1
                show_history_detail_dialog(qa_pairs, idx - 1)
    with nav_col2:
        st.markdown(
            f"<div style='text-align:center;font-size:0.75rem;color:var(--text-muted);"
            f"padding-top:8px;'>{idx + 1} / {total}</div>",
            unsafe_allow_html=True,
        )
    with nav_col3:
        if idx < total - 1:
            if st.button("Tiếp →", use_container_width=True, key="hist_next"):
                st.session_state.selected_history_idx = idx + 1
                show_history_detail_dialog(qa_pairs, idx + 1)


@st.dialog("Xác nhận xóa lịch sử")
def confirm_clear_history_dialog():
    """Dialog xác nhận xóa lịch sử chat — hiển thị ở giữa màn hình."""
    st.markdown(
        """
        <div class="confirm-dialog">
            <div class="confirm-title">Xác nhận xóa lịch sử</div>
            <div class="confirm-message">
                Bạn có chắc chắn muốn xóa <strong>toàn bộ lịch sử chat</strong>?
                Hành động này không thể hoàn tác.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        if st.button("Xác nhận xóa", use_container_width=True, key="confirm_clear_history_yes", type="primary"):
            st.session_state.chat_history = []
            clear_history_persist()
            st.rerun()
    with confirm_col2:
        if st.button("Hủy bỏ", use_container_width=True, key="confirm_clear_history_no"):
            st.rerun()


@st.dialog("Xác nhận xóa tài liệu")
def confirm_clear_vectorstore_dialog():
    """Dialog xác nhận xóa vector store — hiển thị ở giữa màn hình."""
    st.markdown(
        """
        <div class="confirm-dialog">
            <div class="confirm-title">Xác nhận xóa tài liệu</div>
            <div class="confirm-message">
                Bạn có chắc chắn muốn xóa <strong>toàn bộ tài liệu đã upload</strong>
                và vector store? Hành động này không thể hoàn tác.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        if st.button("Xác nhận xóa", use_container_width=True, key="confirm_clear_vs_yes", type="primary"):
            clear_vector_store()
            clear_persist_data()
            st.session_state.vector_store = None
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.session_state.last_processed_upload_signature = ""
            st.session_state.raw_documents = []        # Q7: reset BM25 data
            st.session_state.active_file_filter = []   # Q8: reset filter
            st.rerun()
    with confirm_col2:
        if st.button("Hủy bỏ", use_container_width=True, key="confirm_clear_vs_no"):
            st.rerun()


def render_metadata_filter():
    """
    Q8 — Metadata Filtering: cho phép người dùng chọn file để giới hạn tìm kiếm.
    Chỉ hiển thị khi có ít nhất 1 file đã xử lý.
    """
    files = st.session_state.processed_files
    if not files:
        st.markdown(
            '<div class="empty-files">Chưa có tài liệu nào để lọc.</div>',
            unsafe_allow_html=True,
        )
        return

    file_names = [f["name"] for f in files]
    selected = st.multiselect(
        "Chỉ tìm kiếm trong:",
        options=file_names,
        default=st.session_state.active_file_filter,
        placeholder="Chọn file (bỏ trống = tất cả)",
        key="file_filter_select",
        help="Lọc câu trả lời chỉ từ các file được chọn. Bỏ chọn để tìm toàn bộ.",
    )
    st.session_state.active_file_filter = selected

    if selected:
        st.caption(f"Đang lọc: {len(selected)}/{len(file_names)} file")
    else:
        st.caption("Tìm kiếm toàn bộ tài liệu")


def render_hybrid_toggle():
    """
    Q7 — Hybrid Search: toggle BM25 + Vector Ensemble.
    Hiển thị trọng số hiện tại và thông báo khi chưa có tài liệu.
    """
    has_docs = bool(st.session_state.raw_documents)

    hybrid_on = st.toggle(
        "Bật Hybrid Search (BM25 + Vector)",
        value=st.session_state.hybrid_enabled,
        disabled=not has_docs,
        key="hybrid_toggle",
        help=(
            "Kết hợp tìm kiếm ngữ nghĩa (FAISS) với tìm kiếm từ khoá (BM25). "
            "Hiệu quả hơn với câu hỏi chứa tên riêng, mã số, hoặc từ khoá chuyên ngành."
        ),
    )
    st.session_state.hybrid_enabled = hybrid_on

    if not has_docs:
        st.caption("Tải tài liệu lên để kích hoạt Hybrid Search")
    elif hybrid_on:
        import config as _cfg
        st.caption(
            f"Đang dùng: Vector {int(_cfg.HYBRID_VECTOR_WEIGHT*100)}% "
            f"+ BM25 {int(_cfg.HYBRID_BM25_WEIGHT*100)}%"
        )
    else:
        st.caption("Đang dùng: Pure Vector Search (FAISS)")


def render_reranker_toggle():
    """Q9 — Cross-Encoder Re-ranking."""
    has_docs = st.session_state.vector_store is not None
    reranker_on = st.toggle(
        "Bật Re-ranking (Cross-Encoder)",
        value=st.session_state.reranker_enabled,
        disabled=not has_docs,
        key="reranker_toggle",
        help=(
            "Sau khi FAISS/BM25 lấy candidates, Cross-Encoder đánh giá lại "
            "từng cặp (query, passage) để xếp hạng chính xác hơn."
        ),
    )
    st.session_state.reranker_enabled = reranker_on
    if not has_docs:
        st.caption("Tải tài liệu để kích hoạt Re-ranking")
    elif reranker_on:
        st.caption("Cross-Encoder: ms-marco-MiniLM-L-6-v2")
    else:
        st.caption("Dùng Bi-Encoder scores (FAISS)")


def render_self_rag_toggle():
    """Q10 — Self-RAG."""
    has_docs = st.session_state.vector_store is not None
    self_rag_on = st.toggle(
        "Bật Self-RAG (AI tự đánh giá)",
        value=st.session_state.self_rag_enabled,
        disabled=not has_docs,
        key="self_rag_toggle",
        help="LLM tự viết lại query, lọc docs, sinh câu trả lời rồi tự đánh giá.",
    )
    st.session_state.self_rag_enabled = self_rag_on
    if self_rag_on and has_docs:
        st.session_state.self_rag_query_rewrite = st.checkbox(
            "Query Rewriting", value=st.session_state.self_rag_query_rewrite,
            key="self_rag_qr", help="Tự động viết lại câu hỏi thành 3 variants",
        )
        st.session_state.self_rag_relevance_filter = st.checkbox(
            "Relevance Filtering", value=st.session_state.self_rag_relevance_filter,
            key="self_rag_rf", help="Lọc docs không liên quan",
        )
        st.session_state.self_rag_answer_grading = st.checkbox(
            "Answer Grading", value=st.session_state.self_rag_answer_grading,
            key="self_rag_ag", help="Tự đánh giá chất lượng câu trả lời",
        )
        st.caption("Multi-hop reasoning: tự động bật khi cần")
    elif not has_docs:
        st.caption("Tải tài liệu để kích hoạt Self-RAG")
    else:
        st.caption("Dùng RAG thông thường")


def render_self_rag_metadata(result: dict):
    """Hiển thị panel Self-RAG Analysis sau câu trả lời."""
    confidence = result.get("confidence_score", 0.5)
    is_grounded = result.get("is_grounded", True)
    has_hallucination = result.get("has_hallucination", False)
    feedback = result.get("grading_feedback", "")
    rewritten = result.get("rewritten_queries", [])
    docs_before = result.get("docs_before_filter", 0)
    docs_after = result.get("docs_after_filter", 0)
    sub_questions = result.get("sub_questions", [])
    used_multihop = result.get("used_multihop", False)

    conf_pct = int(confidence * 100)
    conf_icon = "[OK]" if conf_pct >= 70 else ("[!]" if conf_pct >= 40 else "[X]")
    conf_label = "Cao" if conf_pct >= 70 else ("Trung bình" if conf_pct >= 40 else "Thấp")
    hallucination_icon = "Có thể có hallucination" if has_hallucination else "Không phát hiện hallucination"
    grounded_icon = "Dựa trên tài liệu" if is_grounded else "Có thể tự bịa"
    multihop_str = f"| Multi-hop: {len(sub_questions)} sub-questions" if used_multihop else ""
    feedback_html = f"<div style='color:#4B5563;margin-bottom:6px;'><em>{feedback}</em></div>" if feedback else ""

    st.markdown(
        f"""<div style="background:rgba(108,140,255,0.08);border:1px solid rgba(108,140,255,0.25);
        border-radius:12px;padding:14px 16px;margin-top:10px;font-size:0.82rem;">
            <div style="font-weight:700;color:#9aa5bc;margin-bottom:10px;font-size:0.7rem;
            text-transform:uppercase;letter-spacing:0.08em;">Self-RAG Analysis</div>
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;">
                <span>{conf_icon} Confidence: <strong>{conf_pct}% ({conf_label})</strong></span>
                <span>{grounded_icon}</span>
                <span>{hallucination_icon}</span>
            </div>
            {feedback_html}
            <div style='color:#6b7894;font-size:0.72rem;'>
                Docs: {docs_before} retrieved → {docs_after} sau filter {multihop_str}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    if len(rewritten) > 1:
        with st.expander("Xem Query Rewriting", expanded=False):
            for i, q in enumerate(rewritten):
                label = "Gốc" if i == 0 else f"Variant {i}"
                st.markdown(f"**{label}:** {q}")

    if sub_questions:
        with st.expander("Multi-hop Sub-questions", expanded=False):
            for sq in sub_questions:
                st.markdown(f"• {sq}")


def render_co_rag_toggle():
    """Co-RAG — toggle bật/tắt cùng cấu hình agents."""
    has_docs = st.session_state.vector_store is not None
    co_rag_on = st.toggle(
        "Bật Co-RAG (Multi-Agent)",
        value=st.session_state.co_rag_enabled,
        disabled=not has_docs,
        key="co_rag_toggle",
        help=(
            "Co-RAG chạy 3 agents song song (Semantic, Keyword, Conceptual) và hợp nhất "
            "kết quả qua voting để tăng chất lượng truy xuất."
        ),
    )
    st.session_state.co_rag_enabled = co_rag_on

    if co_rag_on and has_docs:
        strategy = st.selectbox(
            "Chiến lược merge",
            options=["voting", "union", "intersection"],
            index=["voting", "union", "intersection"].index(
                st.session_state.co_rag_merge_strategy
            ),
            key="co_rag_strategy_select",
            help=(
                "voting: chỉ giữ docs được ≥2 agents đồng ý\n"
                "union: giữ tất cả docs từ mọi agents\n"
                "intersection: chỉ giữ docs có trong MỌI agents"
            ),
        )
        st.session_state.co_rag_merge_strategy = strategy

        st.markdown("**Agents:**")
        st.session_state.co_rag_agent_semantic = st.checkbox(
            "Semantic Agent (FAISS)",
            value=st.session_state.co_rag_agent_semantic,
            key="co_rag_sem",
            help="Tìm kiếm ngữ nghĩa bằng vector embedding",
        )
        st.session_state.co_rag_agent_keyword = st.checkbox(
            "Keyword Agent (BM25)",
            value=st.session_state.co_rag_agent_keyword,
            key="co_rag_kw",
            help="Tìm kiếm từ khoá chính xác bằng BM25",
        )
        st.session_state.co_rag_agent_conceptual = st.checkbox(
            "Conceptual Agent (LLM)",
            value=st.session_state.co_rag_agent_conceptual,
            key="co_rag_con",
            help="LLM phân rã câu hỏi → sub-questions → retrieve",
        )
        st.caption("Consensus Merger tổng hợp kết quả từ các agents")
    elif not has_docs:
        st.caption("Tải tài liệu để kích hoạt Co-RAG")
    else:
        st.caption("Dùng RAG thông thường")


def render_co_rag_metadata(result: dict):
    """Hiển thị Co-RAG Analysis panel sau câu trả lời."""
    agent_counts = result.get("co_rag_agent_counts", {})
    total_before = result.get("co_rag_total_before_merge", 0)
    total_after = result.get("co_rag_total_after_merge", 0)
    strategy = result.get("co_rag_merge_strategy", "voting")

    agents_html = "".join(
        f"<span style='margin-right:12px;'>● <strong>{name}</strong>: {cnt} docs</span>"
        for name, cnt in agent_counts.items()
    )
    strategy_color = {
        "voting": "#3B82F6",
        "union": "#16A34A",
        "intersection": "#D97706",
    }.get(strategy, "#6B7280")

    st.markdown(
        f"""<div style="background:rgba(108,140,255,0.07);border:1px solid rgba(108,140,255,0.22);
        border-radius:12px;padding:14px 16px;margin-top:10px;font-size:0.82rem;">
            <div style="font-weight:700;color:#9aa5bc;margin-bottom:10px;font-size:0.7rem;
            text-transform:uppercase;letter-spacing:0.08em;">Co-RAG Analysis</div>
            <div style="margin-bottom:8px;color:#4B5563;">
                {agents_html}
            </div>
            <div style="display:flex;gap:20px;flex-wrap:wrap;font-size:0.78rem;color:#6b7894;">
                <span>Tổng docs (trước merge): <strong>{total_before}</strong></span>
                <span>Sau Consensus Merger: <strong>{total_after}</strong></span>
                <span>Chiến lược: <strong style="color:{strategy_color};">{strategy}</strong></span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_action_buttons():
    """Render các nút thao tác — dialog xác nhận hiển thị ở giữa màn hình."""
    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("Xóa lịch sử", use_container_width=True, key="clear_chat_btn"):
            confirm_clear_history_dialog()

    with action_col2:
        if st.button("Xóa tài liệu", use_container_width=True, key="clear_vs_btn"):
            confirm_clear_vectorstore_dialog()


# ── End sidebar helpers ───────────────────────────────────────────────────────


# ============================================================
# Document Processing
# ============================================================
def process_documents(uploaded_files, upload_signature: str = ""):
    """Xử lý các file PDF / DOCX đã upload với giao diện mượt mà."""
    st.session_state.is_processing = True

    # Reset toàn bộ vector store cũ (kể cả dữ liệu từ session trước trên disk)
    # để đảm bảo chỉ tìm kiếm trong đúng các file đang upload lần này
    clear_vector_store()
    st.session_state.vector_store = None
    st.session_state.raw_documents = []
    st.session_state.processed_files = []
    st.session_state.total_chunks = 0

    total = len(uploaded_files)
    all_chunks = []
    new_files_info = []

    # Tạo UI containers trong main area (sidebar quá nhỏ cho progress)
    progress_bar = st.progress(0, text="Đang chuẩn bị xử lý...")
    status_container = st.empty()

    # === Bước 1 & 2: Đọc tài liệu và Chunking ===
    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        base_progress = idx / total

        status_container.markdown(
            f"""
            <div class="processing-toast">
                Đang đọc <strong>{file_name}</strong> ({idx + 1}/{total})...
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress_bar.progress(base_progress + (0.3 / total), text=f"Đọc {file_name}...")

        # Xác định phần mở rộng để giữ nguyên khi lưu file tạm
        import os as _os
        _, file_ext = _os.path.splitext(file_name)
        file_ext = file_ext.lower() if file_ext else ".pdf"

        try:
            # Lưu file tạm với đúng phần mở rộng
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Trích xuất (tự động phát hiện định dạng)
            if file_ext == ".docx":
                raw_docs = extract_text_from_docx(tmp_path, source_name=file_name)
            else:
                raw_docs = extract_text_from_pdf(tmp_path, source_name=file_name)
            num_pages = len(raw_docs)

            # Chunking
            status_container.markdown(
                f"""
                <div class="processing-toast">
                    Đang chia nhỏ <strong>{file_name}</strong> ({num_pages} trang)...
                </div>
                """,
                unsafe_allow_html=True,
            )
            progress_bar.progress(base_progress + (0.6 / total), text=f"Chia nhỏ {file_name}...")

            chunks = split_documents(
                raw_docs,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            )

            if chunks:
                all_chunks.extend(chunks)
                new_files_info.append(
                    {"name": file_name, "chunks": len(chunks), "pages": num_pages}
                )
                logger.info(
                    f"Đã xử lý '{file_name}': {num_pages} trang → {len(chunks)} chunks"
                )
            else:
                st.warning(f"'{file_name}' không có nội dung văn bản.")

        except Exception as e:
            st.error(f"Lỗi khi xử lý '{file_name}': {str(e)}")
            logger.error(f"Lỗi xử lý '{file_name}': {str(e)}")
        finally:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        progress_bar.progress((idx + 1) / total * 0.7, text="Chuẩn bị tạo vector index...")

    # === Bước 3: Tạo Vector Index ===
    if all_chunks:
        status_container.markdown(
            f"""
            <div class="processing-toast">
                Đang tạo vector index cho <strong>{len(all_chunks)}</strong> chunks...
            </div>
            """,
            unsafe_allow_html=True,
        )
        progress_bar.progress(0.75, text="Tạo vector index...")

        try:
            if st.session_state.vector_store is None:
                st.session_state.vector_store = create_vector_store(all_chunks)
            else:
                st.session_state.vector_store = add_documents_to_store(
                    st.session_state.vector_store, all_chunks
                )

            progress_bar.progress(0.9, text="Lưu vector store...")

            # Lưu vector store (FAISS index)
            save_vector_store(st.session_state.vector_store)

            # Cập nhật thông tin
            st.session_state.processed_files.extend(new_files_info)
            st.session_state.total_chunks += len(all_chunks)

            # Q7: lưu raw docs để xây BM25 retriever (reset khi xóa rồi upload lại)
            st.session_state.raw_documents.extend(all_chunks)

            # Persist metadata file list ra disk để restore khi khởi động lại
            save_processed_files(st.session_state.processed_files)

            progress_bar.progress(1.0, text="Hoàn tất!")

            total_new_chunks = sum(f["chunks"] for f in new_files_info)
            total_new_pages = sum(f["pages"] for f in new_files_info)
            status_container.markdown(
                f"""
                <div class="success-toast">
                    Hoàn tất! {len(new_files_info)} tài liệu · {total_new_pages} trang · {total_new_chunks} chunks
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(2)

        except Exception as e:
            st.error(f"Lỗi khi tạo vector store: {str(e)}")
            logger.error(f"Lỗi tạo vector store: {str(e)}")

    # Cleanup
    status_container.empty()
    progress_bar.empty()

    st.session_state.is_processing = False
    if upload_signature:
        st.session_state.last_processed_upload_signature = upload_signature
    st.rerun()


# ============================================================
# Main Chat Area
# ============================================================
def render_main():
    """Render main chat area — chỉ chứa chat."""

    # Welcome hoặc Chat history
    if not st.session_state.chat_history:
        render_welcome()
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])
            question_ctx = msg.get("question_ctx", "")
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(content)
                    if sources:
                        answer_ctx = msg.get("answer_ctx", "")
                        render_sources(sources, question=question_ctx, answer=answer_ctx)

    # Settings expander — đặt trên chat input, kiểu ChatGPT
    with st.expander("Cài đặt tìm kiếm", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Lọc tài liệu (Q8)</div>', unsafe_allow_html=True)
            render_metadata_filter()
            st.markdown('<div class="section-header">Hybrid Search (Q7)</div>', unsafe_allow_html=True)
            render_hybrid_toggle()
        with col2:
            st.markdown('<div class="section-header">Re-ranking (Q9)</div>', unsafe_allow_html=True)
            render_reranker_toggle()
            st.markdown('<div class="section-header">Self-RAG (Q10)</div>', unsafe_allow_html=True)
            render_self_rag_toggle()
            st.markdown('<div class="section-header">Co-RAG (Multi-Agent)</div>', unsafe_allow_html=True)
            render_co_rag_toggle()

    # Chat input
    if prompt := st.chat_input(
        "Nhập câu hỏi của bạn... (ví dụ: Tóm tắt tài liệu, Trích xuất thông tin quan trọng)",
        key="chat_input",
    ):
        handle_user_input(prompt)


# ── Main area helpers ────────────────────────────────────────────────────────

def render_welcome():
    """Render welcome hero khi chưa có cuộc hội thoại."""
    has_docs = len(st.session_state.processed_files) > 0

    if has_docs:
        st.markdown(
            """
            <div class="welcome-hero">
                <div class="welcome-icon">S</div>
                <h1>Sẵn sàng trò chuyện!</h1>
                <p>
                    Tài liệu đã được xử lý thành công. Hãy đặt câu hỏi về nội dung
                    tài liệu trong khung chat bên dưới.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="welcome-hero">
                <div class="welcome-icon">S</div>
                <h1>Chào mừng đến SmartDocAI</h1>
                <p>
                    Trợ lý AI thông minh giúp bạn phân tích và hỏi đáp nội dung
                    tài liệu PDF & DOCX. Tải tài liệu lên và bắt đầu trò chuyện ngay!
                </p>
                <div class="welcome-steps">
                    <div class="welcome-step">
                        <div class="step-num">1</div>
                        Tải file PDF hoặc DOCX lên ở thanh bên trái
                    </div>
                    <div class="welcome-step">
                        <div class="step-num">2</div>
                        Chờ hệ thống xử lý tài liệu
                    </div>
                    <div class="welcome-step">
                        <div class="step-num">3</div>
                        Đặt câu hỏi trong khung chat
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def highlight_text(text: str, query: str, answer: str = "") -> str:
    """
    Highlight các đoạn văn được sử dụng để trả lời theo 2 lớp ưu tiên:

    Layer 1 (ưu tiên cao) — N-gram từ câu trả lời (answer):
        Tìm các cụm 4-7 từ xuất hiện trong answer và cũng có trong chunk.
        Đây là bằng chứng trực tiếp rằng LLM đã đọc đoạn này để tạo câu trả lời.

    Layer 2 (fallback) — Keyword từ câu hỏi (query):
        Các từ ≥ 3 ký tự từ query chưa được highlight ở layer 1.
    """
    import re
    import html

    safe = html.escape(text)
    candidates = []   # list of (phrase, layer)

    # ── Layer 1: n-gram từ answer ──
    if answer:
        answer_clean = re.sub(r'[\*\_\`\#\>\|]', ' ', answer)   # strip markdown
        answer_words = re.split(r'\s+', answer_clean.strip())
        for n in range(7, 3, -1):   # 7 → 4 words
            for i in range(len(answer_words) - n + 1):
                phrase = ' '.join(answer_words[i : i + n])
                phrase = phrase.strip('.,;:!?()')
                if len(phrase) < 10:
                    continue
                # Kiểm tra phrase có xuất hiện trong nội dung chunk không
                if re.search(re.escape(phrase), safe, re.IGNORECASE):
                    candidates.append((phrase, 1))  # layer 1

    # ── Layer 2: keyword từ query ──
    query_tokens = [
        t for t in re.split(r'[\s\W]+', query)
        if len(t) >= 3
    ]
    for tok in query_tokens:
        candidates.append((tok, 2))  # layer 2

    if not candidates:
        return safe

    # Sắp xếp: layer 1 trước, dài hơn trước (tránh khớp mảnh)
    candidates = sorted(set(candidates), key=lambda x: (-x[1] == 1, -len(x[0])))
    # loại trùng, giữ phrase dài hơn bao gồm phrase ngắn
    unique_phrases = []
    seen_lower = set()
    for phrase, layer in candidates:
        pl = phrase.lower()
        if not any(pl in s for s in seen_lower):
            unique_phrases.append((phrase, layer))
            seen_lower.add(pl)

    # Màu highlight theo layer
    def make_mark(m, layer):
        if layer == 1:
            # layer 1: vàng đẮm — đoạn thực sự được dùng
            return (
                f'<mark style="background:rgba(251,191,36,0.35);'
                f'color:#92400e;border-radius:3px;padding:1px 2px;font-weight:600;">'
                f'{html.escape(m.group(0))}</mark>'
            )
        else:
            # layer 2: xanh nhạt — keyword tìm kiếm
            return (
                f'<mark style="background:rgba(59,130,246,0.15);'
                f'color:#1e40af;border-radius:3px;padding:1px 2px;">'
                f'{html.escape(m.group(0))}</mark>'
            )

    result_text = safe
    for phrase, layer in unique_phrases:
        result_text = re.sub(
            re.escape(phrase),
            lambda m, _layer=layer: make_mark(m, _layer),
            result_text,
            flags=re.IGNORECASE,
        )

    return result_text


def render_sources(sources: list, question: str = "", answer: str = ""):
    """Hiển thị nguồn trích dẫn: badge row nhanh + expander chi tiết."""
    if not sources:
        return

    # ── Badge row ──
    badges_html = ""
    for s in sources:
        file_label = s['file']
        page = s.get('page', '?')
        total = s.get('total_pages')
        page_str = f"Trang {page}/{total}" if total else f"Trang {page}"
        badges_html += (
            f'<span class="citation-badge">'
            f'{file_label} — {page_str}'
            f'</span>'
        )

    st.markdown(
        f'<div class="citation-panel">'
        f'<div class="citation-badge-row">{badges_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Expander chi tiết ──
    with st.expander(f"Xem {len(sources)} nguồn trích dẫn", expanded=False):
        # Chú thích màu sắc
        st.markdown(
            """
            <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:0.72rem;
                        margin-bottom:10px;padding:6px 8px;
                        background:var(--bg-surface);border-radius:6px;
                        border:1px solid var(--border-subtle);">
                <span style="display:flex;align-items:center;gap:5px;">
                    <span style="display:inline-block;width:12px;height:12px;border-radius:2px;
                                 background:rgba(251,191,36,0.45);"></span>
                    <strong>Đoạn được dùng để trả lời</strong>
                </span>
                <span style="display:flex;align-items:center;gap:5px;">
                    <span style="display:inline-block;width:12px;height:12px;border-radius:2px;
                                 background:rgba(59,130,246,0.2);"></span>
                    Từ khóa câu hỏi
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for s in sources:
            file_label = s['file']
            page = s.get('page', '?')
            total = s.get('total_pages')
            file_type = s.get('file_type', 'pdf').upper()
            score = s.get('score', 0.0)
            content = s.get('content', '')
            chunk_idx = s.get('chunk_index', '')

            page_str = f"Trang {page} / {total}" if total else f"Trang {page}"
            score_pct = int(score * 100)
            file_icon = "PDF" if file_type == "PDF" else "DOCX"

            highlighted_content = highlight_text(content, question, answer=answer)

            card_html = f"""
<div class="citation-card">
  <div class="citation-header">
    <div class="citation-title">
      <span class="citation-num">{chunk_idx}</span>
      {file_icon} {__import__('html').escape(file_label)}
    </div>
    <div class="citation-meta">
      <span class="citation-page-badge">{page_str}</span>
      <span class="citation-type-badge">{file_type}</span>
    </div>
  </div>
  <div class="citation-score-wrap">
    <span class="citation-score-label">Độ liên quan</span>
    <div class="citation-score-bar-bg">
      <div class="citation-score-bar-fill" style="width:{score_pct}%;"></div>
    </div>
    <span class="citation-score-value">{score_pct}%</span>
  </div>
  <div class="citation-content">{highlighted_content}</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)


# ── User input handler ────────────────────────────────────────────────────────────

def handle_user_input(user_input: str):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
 
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        with st.spinner("Đang phân tích và suy nghĩ..."):
 
            # ── Q10: Self-RAG Pipeline ────────────────────────────────
            if st.session_state.self_rag_enabled and st.session_state.vector_store is not None:
                llm = get_llm()
                result = self_rag_pipeline(
                    question=user_input,
                    vector_store=st.session_state.vector_store,
                    llm=llm,
                    enable_query_rewrite=st.session_state.self_rag_query_rewrite,
                    enable_relevance_filter=st.session_state.self_rag_relevance_filter,
                    enable_answer_grading=st.session_state.self_rag_answer_grading,
                )
                result["search_mode"] = "self_rag"
                result["active_filter"] = st.session_state.active_file_filter
                self_rag_meta = result  # lưu lại để render
                co_rag_meta = None

            # ── Co-RAG: Multi-Agent Pipeline ──────────────────────────
            elif st.session_state.co_rag_enabled and st.session_state.vector_store is not None:
                llm = get_llm()
                result = co_rag_pipeline(
                    question=user_input,
                    vector_store=st.session_state.vector_store,
                    raw_documents=st.session_state.raw_documents,
                    llm=llm,
                    min_votes=config.CO_RAG_MIN_VOTES,
                    merge_strategy=st.session_state.co_rag_merge_strategy,
                    enable_agent_semantic=st.session_state.co_rag_agent_semantic,
                    enable_agent_keyword=st.session_state.co_rag_agent_keyword,
                    enable_agent_conceptual=st.session_state.co_rag_agent_conceptual,
                )
                result["search_mode"] = "co_rag"
                result["active_filter"] = st.session_state.active_file_filter
                co_rag_meta = result  # lưu lại để render
                self_rag_meta = None

            else:
                self_rag_meta = None
                co_rag_meta = None
 
                # ── Q7: Hybrid Search ─────────────────────────────────
                retriever = None
                if st.session_state.hybrid_enabled and st.session_state.vector_store is not None:
                    bm25 = get_cached_bm25_retriever()
                    if bm25 is None and st.session_state.raw_documents:
                        bm25 = create_bm25_retriever(st.session_state.raw_documents)
                    if bm25 is not None:
                        retriever = create_ensemble_retriever(st.session_state.vector_store, bm25)
 
                result = ask_question(
                    question=user_input,
                    vector_store=st.session_state.vector_store,
                    chat_history=st.session_state.chat_history,
                    retriever=retriever,
                    file_filter=st.session_state.active_file_filter,
                )
 
                # ── Q9: Cross-Encoder Reranking ───────────────────────
                if (
                    st.session_state.reranker_enabled
                    and st.session_state.vector_store is not None
                ):
                    doc_score_pairs = st.session_state.vector_store.similarity_search_with_score(
                        user_input
                    )
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
                                "score": ce_score,          # cross-encoder score
                                "bi_encoder_score": bi_score,  # bi-encoder score để so sánh
                            })
                        result["sources"] = reranked_sources
                        result["search_mode"] = result.get("search_mode", "vector") + "+reranked"
 
        # ── Hiển thị câu trả lời ─────────────────────────────────────
        st.markdown(result["answer"])
 
        # ── Q10: Self-RAG metadata panel ─────────────────────────────
        if self_rag_meta:
            render_self_rag_metadata(self_rag_meta)

        # ── Co-RAG metadata panel ────────────────────────────────────
        if co_rag_meta:
            render_co_rag_metadata(co_rag_meta)

        # ── Badge chế độ tìm kiếm ────────────────────────────────────
        mode = result.get("search_mode", "vector")
        active_filter = result.get("active_filter", [])
        badge_parts = []
        if "self_rag" in mode:
            badge_parts.append("Self-RAG")
        elif "co_rag" in mode:
            badge_parts.append("Co-RAG (Multi-Agent)")
        elif "hybrid" in mode:
            badge_parts.append("Hybrid Search")
        else:
            badge_parts.append("Vector Search")
        if "reranked" in mode:
            badge_parts.append("Cross-Encoder Reranked")
        if active_filter:
            badge_parts.append(f"Lọc: {', '.join(active_filter)}")
        st.caption(" · ".join(badge_parts))
 
        # ── Sources ───────────────────────────────────────────────────
        if result.get("sources"):
            render_sources(
                result["sources"],
                question=user_input,
                answer=result.get("answer", ""),
            )
 
        # ── Error / Fallback ──────────────────────────────────────────
        if result.get("error") and not result.get("used_fallback", False):
            st.error(f"{result['error']}")
        elif result.get("used_fallback", False):
            st.info("Đang dùng chế độ dự phòng do Ollama chưa phản hồi ổn định.")
 
    # Lưu vào history (kèm meta để render lại khi scroll)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "question_ctx": user_input,
        "answer_ctx": result.get("answer", ""),   # dùng cho highlight replay
        "self_rag_meta": self_rag_meta,
        "co_rag_meta": co_rag_meta,
    })

    # Persist lịch sử ra disk để khôi phục khi restart
    save_chat_history(st.session_state.chat_history)

    st.rerun()


# ============================================================
# Main Entry Point
# ============================================================
def main():
    """Entry point chính của ứng dụng."""
    # Khôi phục toàn bộ state từ disk — chỉ chạy một lần mỗi session
    if not st.session_state.get("_state_restored", False):
        st.session_state._state_restored = True

        # 1. Tải FAISS vector store
        saved_store = load_vector_store()
        if saved_store is not None:
            st.session_state.vector_store = saved_store
            logger.info("Đã tải vector store từ disk.")

            # 2. Tải danh sách file đã xử lý (chỉ khi có vector store)
            saved_files = load_processed_files()
            if saved_files:
                st.session_state.processed_files = saved_files
                st.session_state.total_chunks = sum(
                    f.get("chunks", 0) for f in saved_files
                )
                logger.info(
                    f"Đã khôi phục {len(saved_files)} file, "
                    f"{st.session_state.total_chunks} chunks từ disk."
                )

        # 3. Tải lịch sử chat (độc lập với vector store)
        if not st.session_state.chat_history:
            saved_history = load_chat_history()
            if saved_history:
                st.session_state.chat_history = saved_history
                logger.info(f"Đã khôi phục {len(saved_history)} tin nhắn từ disk.")

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
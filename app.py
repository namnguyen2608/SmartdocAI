"""
SmartDocAI - Streamlit Application
Giao diện chatbot hỏi đáp tài liệu PDF với RAG
Phiên bản 2.0 — Thiết kế lại hoàn toàn
"""

import os
import sys
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
)
from modules.rag_chain import ask_question, check_ollama_connection

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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Thiết kế hiện đại
# ============================================================
st.markdown(
    """
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── CSS Variables ── */
    :root {
        --bg-primary:    #0b0e14;
        --bg-secondary:  #111621;
        --bg-surface:    #161b27;
        --bg-elevated:   #1c2333;
        --bg-hover:      #222b3d;
        --border-subtle: #252e41;
        --border-default:#2d3751;
        --text-primary:  #e6eaf3;
        --text-secondary:#9aa5bc;
        --text-muted:    #6b7894;
        --accent:        #6c8cff;
        --accent-hover:  #8ba2ff;
        --accent-soft:   rgba(108, 140, 255, 0.12);
        --accent-border: rgba(108, 140, 255, 0.30);
        --success:       #4ecb8d;
        --success-soft:  rgba(78, 203, 141, 0.12);
        --success-border:rgba(78, 203, 141, 0.30);
        --error:         #f06e6e;
        --error-soft:    rgba(240, 110, 110, 0.12);
        --error-border:  rgba(240, 110, 110, 0.30);
        --warning:       #f0b44e;
        --warning-soft:  rgba(240, 180, 78, 0.12);
        --radius-sm:     8px;
        --radius-md:     12px;
        --radius-lg:     16px;
        --radius-xl:     20px;
        --shadow-sm:     0 2px 8px rgba(0,0,0,0.18);
        --shadow-md:     0 4px 20px rgba(0,0,0,0.25);
        --shadow-lg:     0 8px 40px rgba(0,0,0,0.35);
        --transition:    all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    .main .block-container {
        padding: 1.5rem 2rem 3rem 2rem;
        max-width: 100%;
    }
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111621 0%, #0d1018 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.2rem 1rem;
    }

    /* ── Sidebar Brand ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(108,140,255,0.10) 0%, rgba(108,140,255,0.03) 100%);
        border: 1px solid var(--accent-border);
        border-radius: var(--radius-lg);
        margin-bottom: 20px;
    }
    .sidebar-brand .brand-icon {
        width: 42px; height: 42px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, #6c8cff, #a78bfa);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 4px 16px rgba(108, 140, 255, 0.30);
    }
    .sidebar-brand .brand-text h3 {
        margin: 0; font-size: 1.05rem; font-weight: 700;
        color: var(--text-primary); line-height: 1.2;
    }
    .sidebar-brand .brand-text p {
        margin: 2px 0 0; font-size: 0.73rem;
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
        box-shadow: 0 0 8px var(--success);
        animation: pulse-dot 2s infinite;
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

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
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
        background: linear-gradient(135deg, rgba(108,140,255,0.15), rgba(167,139,250,0.15));
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
        background: linear-gradient(90deg, #6c8cff, #a78bfa) !important;
        border-radius: 999px !important;
    }

    /* ── Welcome Hero ── */
    .welcome-hero {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        text-align: center;
        padding: 4rem 2rem;
        min-height: 55vh;
    }
    .welcome-icon {
        width: 80px; height: 80px;
        border-radius: 24px;
        background: linear-gradient(135deg, #6c8cff, #a78bfa);
        display: flex; align-items: center; justify-content: center;
        font-size: 2.2rem;
        box-shadow: 0 8px 32px rgba(108, 140, 255, 0.35);
        margin-bottom: 24px;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    .welcome-hero h1 {
        font-size: 1.8rem; font-weight: 800;
        margin: 0 0 8px;
        background: linear-gradient(135deg, #e6eaf3, #9aa5bc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .welcome-hero p {
        font-size: 0.95rem; color: var(--text-secondary);
        max-width: 480px; line-height: 1.6;
        margin: 0 0 32px;
    }
    .welcome-steps {
        display: flex; gap: 12px; flex-wrap: wrap;
        justify-content: center; margin-bottom: 32px;
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

    /* ── Chat Input ── */
    [data-testid="stChatInput"] {
        border-radius: var(--radius-lg) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-soft) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
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
        background: linear-gradient(135deg, #6c8cff, #8b6cff) !important;
        border: none !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #8ba2ff, #a78bfa) !important;
        box-shadow: 0 4px 16px rgba(108, 140, 255, 0.35) !important;
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
        margin: 16px 0;
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

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--border-default);
        border-radius: 999px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
</style>
""",
    unsafe_allow_html=True,
)


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
                <div class="brand-icon">🧠</div>
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

        # ── Upload ──
        st.markdown('<div class="section-header">📤 Tải tài liệu lên</div>', unsafe_allow_html=True)
        render_upload_section()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Danh sách file ──
        st.markdown('<div class="section-header">📂 Tài liệu đã xử lý</div>', unsafe_allow_html=True)
        render_file_list()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Lịch sử hội thoại ──
        st.markdown('<div class="section-header">💬 Lịch sử hội thoại</div>', unsafe_allow_html=True)
        render_chat_history_sidebar()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Actions ──
        st.markdown('<div class="section-header">⚙️ Thao tác</div>', unsafe_allow_html=True)
        render_action_buttons()


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
            f"⚠️ Không thể kết nối tới Ollama!\n\n"
            f"1. Cài đặt Ollama: https://ollama.com\n"
            f"2. Chạy: `ollama pull {config.OLLAMA_MODEL}`\n"
            f"3. Đảm bảo Ollama đang chạy trên cổng 11434"
        )

    if st.button("🔄 Kiểm tra kết nối", key="recheck_ollama", use_container_width=True):
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
    st.caption("📎 Hỗ trợ kéo-thả nhiều file PDF & DOCX cùng lúc")

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
            file_icon = "📝" if uf.name.lower().endswith(".docx") else "📄"

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
        btn_label = "⏳ Đang xử lý..." if st.session_state.is_processing else "🚀 Xử lý tài liệu"
        if st.button(
            btn_label,
            use_container_width=True,
            type="primary",
            disabled=st.session_state.is_processing,
            key="process_btn",
        ):
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
                📭 Chưa có tài liệu nào được xử lý.<br>
                Hãy tải file PDF hoặc DOCX lên ở phía trên.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for f in st.session_state.processed_files[::-1]:
        pages = f.get("pages", "?")
        chunks = f.get("chunks", "?")
        file_icon = "📝" if f["name"].lower().endswith(".docx") else "📄"
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
    """Render lịch sử hội thoại trong sidebar."""
    history = st.session_state.chat_history

    # Lọc ra các cặp câu hỏi - câu trả lời
    qa_pairs = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            question = history[i]["content"]
            answer = ""
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                answer = history[i + 1]["content"]
                i += 2
            else:
                i += 1
            qa_pairs.append({"question": question, "answer": answer})
        else:
            i += 1

    if not qa_pairs:
        st.markdown(
            """
            <div class="empty-history">
                💭 Chưa có cuộc hội thoại nào.<br>
                Hãy đặt câu hỏi để bắt đầu.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Hiển thị danh sách câu hỏi (mới nhất lên đầu)
    st.caption(f"📋 {len(qa_pairs)} câu hỏi đã được hỏi")

    for idx, pair in enumerate(reversed(qa_pairs)):
        q_display = pair["question"]
        a_preview = pair["answer"][:120] + "..." if len(pair["answer"]) > 120 else pair["answer"]
        # Sanitize: loại bỏ newlines, escape HTML entities và quotes
        q_display = q_display.replace("\n", " ").replace("\r", " ")
        a_preview = a_preview.replace("\n", " ").replace("\r", " ")
        q_display = q_display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        a_preview = a_preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        st.markdown(
            f"""<div class="history-item"><div class="history-question">❓ {q_display}</div><div class="history-answer-preview">💡 {a_preview}</div></div>""",
            unsafe_allow_html=True,
        )

    # Expander để xem chi tiết từng câu hỏi
    with st.expander("🔎 Xem chi tiết câu hỏi đã hỏi", expanded=False):
        for idx, pair in enumerate(reversed(qa_pairs)):
            st.markdown(f"**❓ Câu hỏi {len(qa_pairs) - idx}:**")
            st.markdown(f"> {pair['question']}")
            st.markdown(f"**💡 Câu trả lời:**")
            st.markdown(pair["answer"][:500] + ("..." if len(pair["answer"]) > 500 else ""))
            if idx < len(qa_pairs) - 1:
                st.markdown("---")


@st.dialog("⚠️ Xác nhận xóa lịch sử")
def confirm_clear_history_dialog():
    """Dialog xác nhận xóa lịch sử chat — hiển thị ở giữa màn hình."""
    st.markdown(
        """
        <div class="confirm-dialog">
            <div class="confirm-title">⚠️ Xác nhận xóa lịch sử</div>
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
        if st.button("✅ Xác nhận xóa", use_container_width=True, key="confirm_clear_history_yes", type="primary"):
            st.session_state.chat_history = []
            st.rerun()
    with confirm_col2:
        if st.button("❌ Hủy bỏ", use_container_width=True, key="confirm_clear_history_no"):
            st.rerun()


@st.dialog("⚠️ Xác nhận xóa tài liệu")
def confirm_clear_vectorstore_dialog():
    """Dialog xác nhận xóa vector store — hiển thị ở giữa màn hình."""
    st.markdown(
        """
        <div class="confirm-dialog">
            <div class="confirm-title">⚠️ Xác nhận xóa tài liệu</div>
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
        if st.button("✅ Xác nhận xóa", use_container_width=True, key="confirm_clear_vs_yes", type="primary"):
            clear_vector_store()
            st.session_state.vector_store = None
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.session_state.last_processed_upload_signature = ""
            st.rerun()
    with confirm_col2:
        if st.button("❌ Hủy bỏ", use_container_width=True, key="confirm_clear_vs_no"):
            st.rerun()


def render_action_buttons():
    """Render các nút thao tác — dialog xác nhận hiển thị ở giữa màn hình."""
    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("🗑️ Xóa lịch sử", use_container_width=True, key="clear_chat_btn"):
            confirm_clear_history_dialog()

    with action_col2:
        if st.button("📦 Xóa tài liệu", use_container_width=True, key="clear_vs_btn"):
            confirm_clear_vectorstore_dialog()


# ============================================================
# Document Processing
# ============================================================
def process_documents(uploaded_files, upload_signature: str = ""):
    """Xử lý các file PDF / DOCX đã upload với giao diện mượt mà."""
    st.session_state.is_processing = True

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
                📖 Đang đọc <strong>{file_name}</strong> ({idx + 1}/{total})...
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
                    ✂️ Đang chia nhỏ <strong>{file_name}</strong> ({num_pages} trang)...
                </div>
                """,
                unsafe_allow_html=True,
            )
            progress_bar.progress(base_progress + (0.6 / total), text=f"Chia nhỏ {file_name}...")

            chunks = split_documents(raw_docs)

            if chunks:
                all_chunks.extend(chunks)
                new_files_info.append(
                    {"name": file_name, "chunks": len(chunks), "pages": num_pages}
                )
                logger.info(
                    f"Đã xử lý '{file_name}': {num_pages} trang → {len(chunks)} chunks"
                )
            else:
                st.warning(f"⚠️ '{file_name}' không có nội dung văn bản.")

        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý '{file_name}': {str(e)}")
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
                🔧 Đang tạo vector index cho <strong>{len(all_chunks)}</strong> chunks...
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

            # Lưu vector store
            save_vector_store(st.session_state.vector_store)

            # Cập nhật thông tin
            st.session_state.processed_files.extend(new_files_info)
            st.session_state.total_chunks += len(all_chunks)

            progress_bar.progress(1.0, text="Hoàn tất!")

            total_new_chunks = sum(f["chunks"] for f in new_files_info)
            total_new_pages = sum(f["pages"] for f in new_files_info)
            status_container.markdown(
                f"""
                <div class="success-toast">
                    🎉 Hoàn tất! {len(new_files_info)} tài liệu · {total_new_pages} trang · {total_new_chunks} chunks
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(2)

        except Exception as e:
            st.error(f"❌ Lỗi khi tạo vector store: {str(e)}")
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
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            else:
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(content)
                    if sources:
                        render_sources(sources)

    # Chat input
    if prompt := st.chat_input(
        "Nhập câu hỏi của bạn... (ví dụ: Tóm tắt tài liệu, Trích xuất thông tin quan trọng)",
        key="chat_input",
    ):
        handle_user_input(prompt)


def render_welcome():
    """Render welcome hero khi chưa có cuộc hội thoại."""
    has_docs = len(st.session_state.processed_files) > 0

    if has_docs:
        st.markdown(
            """
            <div class="welcome-hero">
                <div class="welcome-icon">💬</div>
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
                <div class="welcome-icon">🧠</div>
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

    # Suggested questions
    st.markdown("")  # spacer
    example_cols = st.columns(2, gap="small")
    examples = [
        ("📝", "Tóm tắt nội dung chính của tài liệu"),
        ("🔍", "What are the key findings?"),
        ("📊", "Giải thích phần kết luận chi tiết"),
        ("📋", "Liệt kê các khuyến nghị quan trọng"),
    ]

    for i, (icon, example) in enumerate(examples):
        with example_cols[i % 2]:
            if st.button(
                f"{icon}  {example}",
                key=f"example_{i}",
                use_container_width=True,
            ):
                handle_user_input(example)


def render_sources(sources):
    """Hiển thị nguồn tham khảo dạng tags."""
    source_html = " ".join(
        f'<span class="source-tag">📄 {s["file"]} — Trang {s["page"]}</span>'
        for s in sources
    )
    st.markdown(
        f'<div style="margin-top: 10px;">{source_html}</div>',
        unsafe_allow_html=True,
    )


def handle_user_input(user_input: str):
    """Xử lý input từ người dùng."""
    # Thêm câu hỏi vào history
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    # Hiển thị câu hỏi
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Tạo câu trả lời
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🤔 Đang phân tích và suy nghĩ..."):
            result = ask_question(
                question=user_input,
                vector_store=st.session_state.vector_store,
                chat_history=st.session_state.chat_history,
            )

        # Hiển thị câu trả lời
        st.markdown(result["answer"])

        # Hiển thị nguồn tham khảo
        if result["sources"]:
            render_sources(result["sources"])

        # Hiển thị lỗi nếu có
        if result["error"] and not result.get("used_fallback", False):
            st.error(f"⚠️ {result['error']}")
        elif result.get("used_fallback", False):
            st.info(
                "💡 Đang dùng chế độ dự phòng do Ollama/LLM chưa phản hồi ổn định. "
                "Bạn vẫn có thể xem các đoạn liên quan ở trên."
            )

    # Lưu vào history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
        }
    )

    # Rerun để cập nhật lịch sử hội thoại trong sidebar ngay lập tức
    st.rerun()


# ============================================================
# Main Entry Point
# ============================================================
def main():
    """Entry point chính của ứng dụng."""
    # Tải vector store từ disk nếu có
    if st.session_state.vector_store is None:
        saved_store = load_vector_store()
        if saved_store is not None:
            st.session_state.vector_store = saved_store
            logger.info("Đã tải vector store từ disk.")

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()

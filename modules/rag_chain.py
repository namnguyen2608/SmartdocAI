"""
SmartDocAI - RAG Chain
Xây dựng luồng RAG: Retrieval → Augmentation → Generation
"""

import logging
import os
import re
from typing import Optional, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from modules.vector_store import similarity_search
from modules.language_detector import detect_language, get_language_instruction
import config

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Templates
# ============================================================

RAG_PROMPT_TEMPLATE = """Bạn là SmartDocAI, một trợ lý AI thông minh chuyên phân tích và trả lời câu hỏi dựa trên nội dung tài liệu.

{language_instruction}

### QUY TẮC:
1. CHỈ trả lời dựa trên thông tin trong phần CONTEXT bên dưới.
2. Nếu CONTEXT không chứa đủ thông tin để trả lời, hãy nói rõ rằng thông tin không có trong tài liệu.
3. Trích dẫn nguồn (tên file, số trang) khi có thể.
4. Trả lời có cấu trúc, rõ ràng, dễ đọc.
5. Không bịa đặt thông tin ngoài CONTEXT.

### CONTEXT:
{context}

### CÂU HỎI:
{question}

### TRẢ LỜI:"""


NO_CONTEXT_PROMPT_TEMPLATE = """Bạn là SmartDocAI, một trợ lý AI thông minh.

{language_instruction}

Người dùng chưa tải tài liệu lên hệ thống. Hãy thông báo lịch sự rằng:
- Họ cần tải file PDF lên trước khi đặt câu hỏi.
- Hướng dẫn họ bấm nút "Tải tài liệu" ở phía trên để tải file lên.

### CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

### TRẢ LỜI:"""


def _clean_source_name(source: str) -> str:
    """Chuẩn hóa tên file nguồn để hiển thị thân thiện."""
    name = os.path.basename(str(source or "N/A"))
    if re.match(r"^tmp[a-zA-Z0-9_\\-]+\\.pdf$", name):
        return "Tai lieu da tai len (du lieu cu)"
    return name


def _build_fallback_answer(relevant_docs: list[Document], language: str) -> str:
    """Tạo câu trả lời dự phòng từ context khi LLM không khả dụng."""
    if not relevant_docs:
        if language == "vi":
            return (
                "Mình chưa thể truy cập mô hình AI để sinh câu trả lời chi tiết. "
                "Hiện cũng không tìm thấy đoạn nội dung phù hợp trong tài liệu."
            )
        return (
            "I cannot reach the AI model right now, and no relevant document "
            "segments were found for this question."
        )

    snippets = []
    for idx, doc in enumerate(relevant_docs[:3], 1):
        source = _clean_source_name(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        short_text = " ".join(doc.page_content.split())
        if len(short_text) > 260:
            short_text = f"{short_text[:260].rstrip()}..."
        snippets.append(f"{idx}. ({source} - Trang {page}) {short_text}")

    if language == "vi":
        intro = (
            "Ollama/LLM hiện chưa sẵn sàng, nên mình tạm gửi các đoạn liên quan nhất "
            "để bạn tham khảo nhanh:"
        )
    else:
        intro = (
            "The Ollama/LLM service is currently unavailable, so here are the most "
            "relevant document snippets for quick reference:"
        )

    return f"{intro}\n\n" + "\n\n".join(snippets)


def get_llm() -> ChatOllama:
    """
    Khởi tạo kết nối tới Ollama LLM.

    Returns:
        ChatOllama instance

    Raises:
        ConnectionError: Khi không thể kết nối tới Ollama
    """
    try:
        llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.OLLAMA_TEMPERATURE,
        )
        return llm
    except Exception as e:
        logger.error(f"Lỗi kết nối Ollama: {str(e)}")
        raise ConnectionError(
            f"Không thể kết nối tới Ollama tại {config.OLLAMA_BASE_URL}. "
            f"Vui lòng kiểm tra:\n"
            f"1. Ollama đã được cài đặt và đang chạy\n"
            f"2. Model '{config.OLLAMA_MODEL}' đã được pull\n"
            f"   (chạy: ollama pull {config.OLLAMA_MODEL})\n"
            f"Chi tiết lỗi: {str(e)}"
        )


def check_ollama_connection() -> bool:
    """
    Kiểm tra kết nối tới Ollama server.

    Returns:
        True nếu kết nối thành công, False nếu không
    """
    try:
        import urllib.request
        url = f"{config.OLLAMA_BASE_URL}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def format_context(documents: list[Document]) -> str:
    """
    Định dạng danh sách Document thành chuỗi context cho prompt.

    Args:
        documents: Danh sách Document từ similarity search

    Returns:
        Chuỗi context đã format
    """
    if not documents:
        return "Không tìm thấy thông tin liên quan trong tài liệu."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = _clean_source_name(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Nguồn {i}: {source} - Trang {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def ask_question(
    question: str,
    vector_store=None,
    chat_history: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Xử lý câu hỏi của người dùng qua pipeline RAG.
    """
    # QUAN TRỌNG: Phải khởi tạo biến result ngay từ đầu
    result = {
        "answer": "",
        "sources": [],
        "language": "en",
        "error": None,
        "used_fallback": False,
    }

    try:
        # Bước 1: Phát hiện ngôn ngữ
        language = detect_language(question)
        result["language"] = language

        # Bước 2: Khởi tạo LLM
        llm = get_llm()

        # Bước 3: Xử lý dựa trên việc có/không có tài liệu
        if vector_store is None:
            # Dùng prompt thông báo chưa có tài liệu
            prompt = ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE)
            # Lưu ý: Cần truyền language_instruction nếu NO_CONTEXT_PROMPT_TEMPLATE dùng nó
            language_instruction = get_language_instruction(language) 
            chain = prompt | llm
            response = chain.invoke({
                "question": question,
                "language_instruction": language_instruction,
            })
            result["answer"] = response.content
        else:
            # Pipeline RAG đầy đủ (Cập nhật theo Listing 6 và Mục 7.2)
            relevant_docs = similarity_search(vector_store, question)
            context = format_context(relevant_docs)

            language_instruction = get_language_instruction(language)
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            chain = prompt | llm
            response = chain.invoke({
                "question": question,
                "context": context,
                "language_instruction": language_instruction,
            })

            result["answer"] = response.content

            # Trích xuất nguồn tham khảo
            sources = []
            for doc in relevant_docs:
                source_info = {
                    "file": _clean_source_name(doc.metadata.get("source", "N/A")),
                    "page": doc.metadata.get("page", "N/A"),
                }
                if source_info not in sources:
                    sources.append(source_info)
            result["sources"] = sources

    except Exception as e:
        error_msg = f"Lỗi xử lý: {str(e)}"
        result["error"] = error_msg
        logger.error(error_msg)
        
        # Nếu có vector_store, dùng fallback để trả về các đoạn text thô
        if vector_store is not None:
            relevant_docs = similarity_search(vector_store, question)
            result["answer"] = _build_fallback_answer(relevant_docs, result["language"])
            result["used_fallback"] = True

    return result

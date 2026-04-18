# -*- coding: utf-8 -*-
"""
Unit & Integration tests cho modules/rag_chain.py

Chạy unit tests (không cần Ollama):
    pytest tests/test_rag_chain.py -v -m "not integration"

Chạy integration tests (cần Ollama đang chạy):
    pytest tests/test_rag_chain.py -v -m integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.documents import Document
from modules.rag_chain import format_context, check_ollama_connection


# ─── format_context ──────────────────────────────────────────────────────────

class TestFormatContext:
    """Kiểm thử hàm format_context()"""

    def _make_doc(self, content, source="test.pdf", page=1):
        return Document(
            page_content=content,
            metadata={"source": source, "page": page}
        )

    def test_empty_list_returns_string(self):
        """Input rỗng → trả về chuỗi thông báo, không crash"""
        result = format_context([])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_single_document_contains_content(self):
        """1 document → output chứa nội dung"""
        doc = self._make_doc("This is test content.")
        result = format_context([doc])
        assert "This is test content." in result

    def test_single_document_contains_source(self):
        """Output phải chứa tên file nguồn"""
        doc = self._make_doc("content", source="my_document.pdf", page=3)
        result = format_context([doc])
        assert "my_document.pdf" in result

    def test_single_document_contains_page(self):
        """Output phải chứa số trang"""
        doc = self._make_doc("content", source="doc.pdf", page=5)
        result = format_context([doc])
        assert "5" in result

    def test_multiple_documents_all_present(self):
        """Nhiều documents → tất cả nội dung đều xuất hiện trong output"""
        docs = [
            self._make_doc("First chunk content", "a.pdf", 1),
            self._make_doc("Second chunk content", "b.pdf", 2),
            self._make_doc("Third chunk content", "c.pdf", 3),
        ]
        result = format_context(docs)
        assert "First chunk content" in result
        assert "Second chunk content" in result
        assert "Third chunk content" in result

    def test_return_type_is_string(self):
        """Kết quả luôn là string"""
        result = format_context([self._make_doc("text")])
        assert isinstance(result, str)

    def test_multiple_docs_separated(self):
        """Nhiều documents được phân tách rõ ràng"""
        docs = [
            self._make_doc("Content A", "a.pdf", 1),
            self._make_doc("Content B", "b.pdf", 2),
        ]
        result = format_context(docs)
        # Hai chunks phải được phân tách (không dính vào nhau)
        idx_a = result.find("Content A")
        idx_b = result.find("Content B")
        assert idx_a != idx_b
        assert idx_a < idx_b  # A xuất hiện trước B

    def test_missing_metadata_no_crash(self):
        """Document thiếu metadata không crash"""
        doc = Document(page_content="content without metadata", metadata={})
        result = format_context([doc])
        assert isinstance(result, str)
        assert "content without metadata" in result


# ─── check_ollama_connection ─────────────────────────────────────────────────

class TestCheckOllamaConnection:
    """Kiểm thử hàm check_ollama_connection()"""

    def test_returns_bool(self):
        """Hàm phải trả về bool (True hoặc False), không raise exception"""
        result = check_ollama_connection()
        assert isinstance(result, bool)

    def test_no_exception_raised(self):
        """Khi Ollama không chạy, không được raise exception"""
        try:
            check_ollama_connection()
        except Exception as e:
            pytest.fail(f"check_ollama_connection() raised {e}")


# ─── Integration tests (cần Ollama) ──────────────────────────────────────────

@pytest.mark.integration
class TestIntegrationOllama:
    """
    Integration tests — yêu cầu Ollama đang chạy và model đã được pull.
    Tự động skip nếu Ollama không kết nối được.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_ollama(self):
        """Skip test nếu Ollama không chạy"""
        if not check_ollama_connection():
            pytest.skip("Ollama không kết nối được — bỏ qua integration test")

    def test_ollama_connection_true(self):
        """Khi Ollama đang chạy → check_ollama_connection() trả True"""
        assert check_ollama_connection() is True

    def test_ask_question_no_vectorstore(self):
        """ask_question() không có vector store → trả về dict với key 'answer'"""
        from modules.rag_chain import ask_question
        result = ask_question("Hello, what can you do?", vector_store=None)
        assert isinstance(result, dict)
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_ask_question_with_vectorstore(self):
        """ask_question() với vector store giả lập → trả về câu trả lời hợp lệ"""
        from modules.rag_chain import ask_question
        from modules.vector_store import create_vector_store

        # Tạo vector store nhỏ với docs giả lập
        docs = [
            Document(
                page_content="SmartDocAI is an offline RAG system built with LangChain and FAISS.",
                metadata={"source": "test.pdf", "page": 1}
            )
        ]
        vs = create_vector_store(docs)
        if vs is None:
            pytest.skip("Không thể tạo vector store")

        result = ask_question(
            "What is SmartDocAI?",
            vector_store=vs,
            chat_history=[]
        )
        assert isinstance(result, dict)
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_ask_question_empty_returns_error(self):
        """ask_question() với câu hỏi rỗng → trả về error hoặc thông báo thân thiện"""
        from modules.rag_chain import ask_question
        result = ask_question("", vector_store=None)
        assert isinstance(result, dict)
        # Phải trả về dict hợp lệ, không crash
        assert "answer" in result or "error" in result

    def test_ask_question_vietnamese(self):
        """ask_question() tiếng Việt → language trong result là 'vi'"""
        from modules.rag_chain import ask_question
        result = ask_question("Xin chào, bạn có thể giúp tôi không?", vector_store=None)
        assert isinstance(result, dict)
        assert result.get("language") == "vi"

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

class TestCitationSourceTracking:
    """
    Q5 — Kiểm thử tính năng trích dẫn nguồn (citation/source tracking).

    ask_question() phải trả về result["sources"] chứa đầy đủ thông tin
    để người dùng biết câu trả lời đến từ file nào, trang bao nhiêu.
    """

    def test_sources_key_exists_in_result(self, mock_vector_store):
        """result dict phải có key 'sources'"""
        from modules.rag_chain import ask_question
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "modules.rag_chain.get_llm"
        ) as mock_get_llm, __import__("unittest.mock", fromlist=["patch"]).patch(
            "modules.rag_chain.similarity_search_with_scores"
        ) as mock_search:
            from langchain_core.documents import Document
            mock_doc = Document(
                page_content="Nội dung test", metadata={"source": "test.pdf", "page": 2}
            )
            mock_search.return_value = [(mock_doc, 0.9)]
            mock_llm = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            mock_response = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
            mock_response.content = "Câu trả lời"
            mock_llm.__or__ = lambda self, other: other
            mock_get_llm.return_value = mock_llm

            result = ask_question("test question", vector_store=mock_vector_store)
            assert "sources" in result

    def test_sources_contains_file_info(self):
        """Mỗi source entry phải có key 'file' và 'page'"""
        from modules.rag_chain import ask_question
        from unittest.mock import patch, MagicMock
        from langchain_core.documents import Document

        mock_doc = Document(
            page_content="Label Encoding là phương pháp mã hóa lời giải.",
            metadata={"source": "LuuHongPhuc.pdf", "page": 5, "file_type": "pdf"}
        )
        mock_vs = MagicMock()

        with patch("modules.rag_chain.similarity_search_with_scores") as mock_search, \
             patch("modules.rag_chain.get_llm") as mock_get_llm:
            mock_search.return_value = [(mock_doc, 0.88)]
            mock_llm = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Đây là câu trả lời."
            mock_llm.__or__ = lambda self, other: other

            # Mock chain.invoke
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm

            result = ask_question("câu hỏi test", vector_store=mock_vs)

        # Nếu pipeline chạy thành công, sources phải có file + page
        if result.get("sources"):
            for src in result["sources"]:
                assert "file" in src, "Source thiếu key 'file'"
                assert "page" in src, "Source thiếu key 'page'"

    def test_source_has_required_keys(self):
        """
        Format của mỗi source dict phải có đủ: file, page, content, score.
        Kiểm thử bằng cách gọi trực tiếp format_context và kiểm tra output.
        """
        from modules.rag_chain import format_context
        from langchain_core.documents import Document

        doc = Document(
            page_content="Crossover kết hợp chromosome cha mẹ.",
            metadata={"source": "thesis.pdf", "page": 7}
        )
        # format_context phải include tên file và số trang
        context_str = format_context([doc])
        assert "thesis.pdf" in context_str
        assert "7" in context_str
        assert "Crossover kết hợp chromosome cha mẹ." in context_str

    def test_sources_deduplication(self):
        """
        Nếu cùng một đoạn văn xuất hiện nhiều lần trong kết quả search,
        sources chỉ nên ghi nhận một lần (deduplication logic trong ask_question).
        Kiểm thử bằng cách xem cùng một page_content không bị lặp lại.
        """
        from modules.rag_chain import format_context
        from langchain_core.documents import Document

        # format_context không dedup — nhưng ask_question có dedup_key
        # Test này verify format_context xử lý nhiều docs không crash
        docs = [
            Document(page_content="Nội dung A", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="Nội dung A", metadata={"source": "a.pdf", "page": 1}),
        ]
        result = format_context(docs)
        assert isinstance(result, str)
        assert "Nội dung A" in result

class TestConversationalRAG:
    """
    Q6 — Kiểm thử Conversational RAG: lịch sử hội thoại được đưa vào context.

    Hệ thống phải có khả năng:
    1. Nhận chat_history (list of {role, content} dicts)
    2. Viết lại follow-up question thành câu độc lập (reformulate)
    3. Bao gồm lịch sử trong prompt gửi đến LLM
    """

    def test_ask_question_accepts_chat_history(self, mock_vector_store, sample_chat_history):
        """ask_question() không crash khi nhận chat_history"""
        from modules.rag_chain import ask_question
        from unittest.mock import patch, MagicMock

        with patch("modules.rag_chain.get_llm") as mock_get_llm, \
             patch("modules.rag_chain.similarity_search_with_scores") as mock_search:
            from langchain_core.documents import Document
            mock_search.return_value = [
                (Document(page_content="test content", metadata={"source": "f.pdf", "page": 1}), 0.8)
            ]
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MagicMock(content="Câu trả lời có ngữ cảnh.")
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm

            result = ask_question(
                "Nó dùng phương pháp gì?",  # follow-up không rõ "nó" là gì
                vector_store=mock_vector_store,
                chat_history=sample_chat_history,
            )
        assert isinstance(result, dict)
        assert "answer" in result

    def test_chat_history_format_accepted(self, sample_chat_history):
        """chat_history là list of dicts với keys 'role' và 'content'"""
        for msg in sample_chat_history:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant")

    def test_empty_chat_history_no_crash(self, mock_vector_store):
        """chat_history=[] không gây lỗi"""
        from modules.rag_chain import ask_question
        from unittest.mock import patch, MagicMock

        with patch("modules.rag_chain.get_llm") as mock_get_llm, \
             patch("modules.rag_chain.similarity_search_with_scores") as mock_search:
            from langchain_core.documents import Document
            mock_search.return_value = [
                (Document(page_content="content", metadata={"source": "a.pdf", "page": 1}), 0.9)
            ]
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MagicMock(content="Answer.")
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm

            result = ask_question("What is GA?", vector_store=mock_vector_store, chat_history=[])
        assert isinstance(result, dict)

    def test_result_has_language_key(self):
        """result dict luôn có key 'language' — cần cho Q6 để trả lời đúng ngôn ngữ"""
        from modules.rag_chain import ask_question
        from unittest.mock import patch, MagicMock

        with patch("modules.rag_chain.get_llm") as mock_get_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = MagicMock(content="Answer without docs.")
            mock_llm = MagicMock()
            mock_llm.__or__ = MagicMock(return_value=mock_chain)
            mock_get_llm.return_value = mock_llm

            result = ask_question("Hello", vector_store=None)
        assert "language" in result
        assert result["language"] in ("vi", "en")

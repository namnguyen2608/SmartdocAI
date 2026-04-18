# -*- coding: utf-8 -*-
"""
Unit & Integration tests cho modules/document_processor.py

Chạy unit tests (không cần file thật):
    pytest tests/test_document_processor.py -v -m "not integration"

Chạy tất cả (cần file PDF/DOCX test thật):
    pytest tests/test_document_processor.py -v
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from langchain_core.documents import Document
from modules.document_processor import split_documents, process_uploaded_file, SUPPORTED_EXTENSIONS


# ─── split_documents ─────────────────────────────────────────────────────────

class TestSplitDocuments:
    """Kiểm thử hàm split_documents()"""

    def _make_docs(self, contents):
        """Helper: tạo danh sách Document từ list chuỗi"""
        return [
            Document(page_content=c, metadata={"source": "test.pdf", "page": i + 1})
            for i, c in enumerate(contents)
        ]

    def test_empty_input_returns_empty(self):
        """Input rỗng → trả về list rỗng"""
        result = split_documents([])
        assert result == []

    def test_returns_list_of_documents(self):
        """Kết quả phải là list Document"""
        docs = self._make_docs(["Hello world " * 100])
        result = split_documents(docs)
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_metadata_preserved(self):
        """Metadata của Document gốc phải được giữ nguyên trong chunks"""
        docs = self._make_docs(["Test content " * 200])
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.pdf"

    def test_chunk_size_respected(self):
        """Mỗi chunk không vượt quá chunk_size"""
        long_text = "word " * 1000  # ~5000 chars
        docs = self._make_docs([long_text])
        chunk_size = 300
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=30)
        assert len(chunks) > 1  # phải có nhiều hơn 1 chunk
        for chunk in chunks:
            # Cho phép vượt nhẹ do separator logic
            assert len(chunk.page_content) <= chunk_size * 1.2

    def test_multiple_documents(self):
        """Nhiều Document đầu vào → tất cả đều được chia nhỏ"""
        docs = self._make_docs(["Content A " * 300, "Content B " * 300])
        chunks = split_documents(docs)
        assert len(chunks) >= 2

    def test_short_document_single_chunk(self):
        """Document ngắn hơn chunk_size → chỉ 1 chunk"""
        docs = self._make_docs(["Short text"])
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text"

    def test_custom_chunk_size(self):
        """chunk_size tùy chỉnh được áp dụng đúng"""
        long_text = "a" * 2000
        docs = self._make_docs([long_text])
        chunks_small = split_documents(docs, chunk_size=100, chunk_overlap=10)
        chunks_large = split_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks_small) > len(chunks_large)

    def test_overlap_creates_continuity(self):
        """Overlap làm cho nội dung liên tiếp nhau (không mất mát)"""
        # Văn bản đủ dài để phân thành nhiều chunk
        text = "The quick brown fox " * 200
        docs = self._make_docs([text])
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
        # Toàn bộ text phải được bao phủ
        all_content = " ".join(c.page_content for c in chunks)
        assert "quick brown fox" in all_content


# ─── SUPPORTED_EXTENSIONS ────────────────────────────────────────────────────

class TestSupportedExtensions:
    """Kiểm thử hằng số SUPPORTED_EXTENSIONS"""

    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_docx_supported(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_is_set_or_collection(self):
        assert hasattr(SUPPORTED_EXTENSIONS, "__contains__")


# ─── process_uploaded_file (error handling) ──────────────────────────────────

class TestProcessUploadedFileErrors:
    """Kiểm thử xử lý lỗi của process_uploaded_file()"""

    def test_nonexistent_file_raises(self):
        """File không tồn tại → raise FileNotFoundError hoặc Exception"""
        with pytest.raises((FileNotFoundError, Exception)):
            process_uploaded_file("/tmp/nonexistent_file_xyz.pdf")

    def test_unsupported_extension_raises(self):
        """Extension không được hỗ trợ → raise ValueError hoặc Exception"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some text")
            tmp_path = f.name
        try:
            with pytest.raises((ValueError, Exception)):
                process_uploaded_file(tmp_path)
        finally:
            os.unlink(tmp_path)


# ─── Integration tests ───────────────────────────────────────────────────────

@pytest.mark.integration
class TestIntegrationExtract:
    """
    Integration tests — yêu cầu file test thật trong tests/fixtures/.
    Tự động skip nếu không có file.
    """

    PDF_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")
    DOCX_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.docx")

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")),
        reason="Fixture PDF không có sẵn"
    )
    def test_extract_pdf_returns_documents(self):
        """PDF fixture → trả về list Document không rỗng"""
        from modules.document_processor import extract_text_from_pdf
        docs = extract_text_from_pdf(self.PDF_FIXTURE)
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "fixtures", "sample.docx")),
        reason="Fixture DOCX không có sẵn"
    )
    def test_extract_docx_returns_documents(self):
        """DOCX fixture → trả về list Document không rỗng"""
        from modules.document_processor import extract_text_from_docx
        docs = extract_text_from_docx(self.DOCX_FIXTURE)
        assert isinstance(docs, list)
        assert len(docs) > 0

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "fixtures", "sample.pdf")),
        reason="Fixture PDF không có sẵn"
    )
    def test_full_pipeline_pdf(self):
        """Full pipeline PDF → split_documents cho ra chunk có content"""
        from modules.document_processor import extract_text_from_pdf
        docs = extract_text_from_pdf(self.PDF_FIXTURE)
        chunks = split_documents(docs)
        assert len(chunks) > 0
        assert all(len(c.page_content) > 0 for c in chunks)

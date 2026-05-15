"""
Unit & Integration tests cho modules/vector_store.py

Chạy unit tests (không load embedding model thật):
    pytest tests/test_vector_store.py -v -m "not integration"

Chạy integration tests (sẽ download/load embedding model ~500MB):
    pytest tests/test_vector_store.py -v -m integration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

class TestGetEmbeddingModelSingleton:
    """Kiểm thử singleton pattern của get_embedding_model()"""

    def test_same_object_returned_twice(self):
        """Gọi 2 lần phải trả về cùng một object (singleton)"""
        # Mock để không load model thật
        mock_model = MagicMock()
        with patch("modules.vector_store._embedding_model", mock_model):
            from modules.vector_store import get_embedding_model
            result1 = get_embedding_model()
            result2 = get_embedding_model()
            assert result1 is result2

    def test_returns_non_none(self):
        """Kết quả không phải None khi model đã được cache"""
        mock_model = MagicMock()
        with patch("modules.vector_store._embedding_model", mock_model):
            from modules.vector_store import get_embedding_model
            result = get_embedding_model()
            assert result is not None

class TestCreateVectorStore:
    """Kiểm thử hàm create_vector_store()"""

    def test_empty_documents_returns_none(self):
        """Input rỗng → trả về None"""
        from modules.vector_store import create_vector_store
        result = create_vector_store([])
        assert result is None

    def test_requires_documents_list(self):
        """Không crash với list hợp lệ rỗng"""
        from modules.vector_store import create_vector_store
        # Không raise exception
        result = create_vector_store([])
        assert result is None

class TestSaveLoadVectorStore:
    """Kiểm thử save_vector_store và load_vector_store"""

    def test_load_returns_none_when_no_index(self, tmp_path):
        """Chưa có index → load_vector_store() trả None"""
        import config
        original_dir = config.VECTORSTORE_DIR
        config.VECTORSTORE_DIR = str(tmp_path)
        try:
            from modules.vector_store import load_vector_store
            result = load_vector_store("nonexistent_index")
            assert result is None
        finally:
            config.VECTORSTORE_DIR = original_dir

class TestSimilaritySearch:
    """Kiểm thử hàm similarity_search()"""

    def test_empty_vector_store_returns_empty(self):
        """Khi vector_store không có kết quả, trả về list rỗng (không crash)"""
        from modules.vector_store import similarity_search
        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = []
        result = similarity_search(mock_vs, "test query", top_k=5)
        assert isinstance(result, list)
        assert result == []

    def test_returns_documents(self):
        """Kết quả phải là list Document"""
        from modules.vector_store import similarity_search
        mock_doc = Document(page_content="test", metadata={"source": "a.pdf"})
        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = [mock_doc]
        result = similarity_search(mock_vs, "query", top_k=1)
        assert len(result) == 1
        assert isinstance(result[0], Document)

    def test_exception_returns_empty_list(self):
        """Khi vector store raise exception → trả về list rỗng"""
        from modules.vector_store import similarity_search
        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.side_effect = Exception("Test error")
        result = similarity_search(mock_vs, "query")
        assert result == []

class TestSimilaritySearchWithScores:
    """Kiểm thử hàm similarity_search_with_scores()"""

    def test_returns_tuples(self):
        """Kết quả là list of (Document, float)"""
        from modules.vector_store import similarity_search_with_scores
        mock_doc = Document(page_content="content", metadata={})
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.return_value = [(mock_doc, 0.85)]
        result = similarity_search_with_scores(mock_vs, "query")
        assert len(result) == 1
        doc, score = result[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_exception_returns_empty(self):
        """Exception → trả về list rỗng"""
        from modules.vector_store import similarity_search_with_scores
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_relevance_scores.side_effect = Exception("err")
        result = similarity_search_with_scores(mock_vs, "query")
        assert result == []

@pytest.mark.integration
class TestIntegrationVectorStore:
    """
    Integration tests — tải embedding model thật (~500MB lần đầu).
    Dùng cache: lần đầu chậm, các lần sau nhanh.
    """

    @pytest.fixture(scope="class")
    def sample_docs(self):
        return [
            Document(
                page_content="FAISS is a library for efficient similarity search.",
                metadata={"source": "test.pdf", "page": 1}
            ),
            Document(
                page_content="LangChain provides tools for building RAG pipelines.",
                metadata={"source": "test.pdf", "page": 2}
            ),
            Document(
                page_content="Ollama allows running large language models locally.",
                metadata={"source": "test.pdf", "page": 3}
            ),
        ]

    def test_create_vector_store_success(self, sample_docs):
        """create_vector_store() với docs thật → không trả None"""
        from modules.vector_store import create_vector_store
        vs = create_vector_store(sample_docs)
        assert vs is not None

    def test_similarity_search_returns_results(self, sample_docs):
        """Sau khi tạo vector store, search trả về kết quả liên quan"""
        from modules.vector_store import create_vector_store, similarity_search
        vs = create_vector_store(sample_docs)
        results = similarity_search(vs, "What is FAISS?", top_k=2)
        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)

    def test_search_with_scores_in_range(self, sample_docs):
        """Score từ similarity_search_with_scores() nằm trong [0, 1]"""
        from modules.vector_store import create_vector_store, similarity_search_with_scores
        vs = create_vector_store(sample_docs)
        results = similarity_search_with_scores(vs, "LangChain RAG", top_k=2)
        for doc, score in results:
            assert 0.0 <= score <= 1.0, f"Score {score} ngoài phạm vi [0,1]"

    def test_save_and_load_vector_store(self, sample_docs, tmp_path):
        """Lưu vector store và tải lại → vẫn tìm kiếm được"""
        import config
        from modules.vector_store import create_vector_store, save_vector_store, load_vector_store, similarity_search

        original_dir = config.VECTORSTORE_DIR
        config.VECTORSTORE_DIR = str(tmp_path)
        try:
            vs = create_vector_store(sample_docs)
            save_vector_store(vs, "test_index")
            loaded_vs = load_vector_store("test_index")
            assert loaded_vs is not None
            results = similarity_search(loaded_vs, "FAISS", top_k=1)
            assert len(results) > 0
        finally:
            config.VECTORSTORE_DIR = original_dir

    def test_singleton_embedding_model(self):
        """get_embedding_model() gọi 2 lần với model thật → cùng object"""
        # Reset cache trước
        import modules.vector_store as vs_module
        original = vs_module._embedding_model
        vs_module._embedding_model = None
        try:
            from modules.vector_store import get_embedding_model
            m1 = get_embedding_model()
            m2 = get_embedding_model()
            assert m1 is m2
        finally:
            vs_module._embedding_model = original

class TestMetadataFiltering:
    """
    Q8 — Kiểm thử tính năng lọc tài liệu theo metadata (Multi-document RAG).

    Khi người dùng chọn chỉ tìm trong file A, hệ thống phải:
    1. Loại bỏ các chunks đến từ file B, C, ...
    2. Chỉ trả về chunks có source khớp với filter
    3. Xử lý đúng khi filter là list rỗng (tìm tất cả)
    """

    def test_filter_keeps_matching_source(self, sample_docs):
        """
        Sau khi apply file_filter, chỉ giữ docs có source chứa tên file được chọn.
        Logic này nằm trong ask_question() — test trực tiếp filter expression.
        """
        file_filter = ["fileA.pdf"]
        filtered = [
            doc for doc in sample_docs
            if any(f in doc.metadata.get("source", "") for f in file_filter)
        ]
        assert len(filtered) == 2  # 2 docs từ fileA, 1 từ fileB
        assert all("fileA.pdf" in d.metadata["source"] for d in filtered)

    def test_filter_excludes_other_sources(self, sample_docs):
        """Docs từ file không được chọn bị loại khỏi kết quả"""
        file_filter = ["fileB.pdf"]
        filtered = [
            doc for doc in sample_docs
            if any(f in doc.metadata.get("source", "") for f in file_filter)
        ]
        assert all("fileB.pdf" in d.metadata["source"] for d in filtered)
        assert not any("fileA.pdf" in d.metadata["source"] for d in filtered)

    def test_empty_filter_returns_all(self, sample_docs):
        """file_filter=[] → không lọc gì, trả về tất cả docs"""
        file_filter = []
        if file_filter:
            filtered = [
                doc for doc in sample_docs
                if any(f in doc.metadata.get("source", "") for f in file_filter)
            ]
        else:
            filtered = sample_docs  # No filter → all docs
        assert len(filtered) == len(sample_docs)

    def test_filter_nonexistent_file_returns_empty(self, sample_docs):
        """Lọc theo file không tồn tại → kết quả rỗng"""
        file_filter = ["nonexistent_file.pdf"]
        filtered = [
            doc for doc in sample_docs
            if any(f in doc.metadata.get("source", "") for f in file_filter)
        ]
        assert filtered == []

    def test_multiple_files_in_filter(self, sample_docs):
        """Lọc đồng thời nhiều file → trả về docs từ tất cả file đã chọn"""
        file_filter = ["fileA.pdf", "fileB.pdf"]
        filtered = [
            doc for doc in sample_docs
            if any(f in doc.metadata.get("source", "") for f in file_filter)
        ]
        assert len(filtered) == len(sample_docs)  # tất cả 3 docs

    def test_doc_metadata_has_source_key(self, sample_docs):
        """Mỗi document phải có metadata['source'] để filter hoạt động"""
        for doc in sample_docs:
            assert "source" in doc.metadata, \
                f"Doc thiếu metadata['source']: {doc.page_content[:30]}"

    def test_similarity_search_mock_with_filter(self):
        """
        Mô phỏng toàn bộ flow Q8 trong ask_question:
        search → trả về mixed sources → apply filter → chỉ còn fileA.
        """
        from langchain_core.documents import Document

        # Giả lập kết quả search trả về docs từ nhiều file
        raw_results = [
            (Document(page_content="GA info", metadata={"source": "fileA.pdf", "page": 1}), 0.90),
            (Document(page_content="FAISS info", metadata={"source": "fileB.pdf", "page": 3}), 0.85),
            (Document(page_content="More GA", metadata={"source": "fileA.pdf", "page": 5}), 0.80),
        ]

        # Apply filter như trong ask_question()
        file_filter = ["fileA.pdf"]
        filtered = [
            (doc, score) for doc, score in raw_results
            if any(f in doc.metadata.get("source", "") for f in file_filter)
        ]

        assert len(filtered) == 2
        assert all("fileA.pdf" in doc.metadata["source"] for doc, _ in filtered)

@pytest.mark.integration
class TestIntegrationMetadataFilter:
    """
    Integration test Q8: Tạo vector store thật với nhiều file,
    kiểm tra filter hoạt động end-to-end.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_embedding(self):
        """Skip nếu không load được embedding model"""
        try:
            from modules.vector_store import get_embedding_model
            model = get_embedding_model()
            if model is None:
                pytest.skip("Embedding model không khả dụng")
        except Exception:
            pytest.skip("Không thể load embedding model")

    def test_multi_file_vector_store_created(self):
        """Tạo vector store từ docs của 2 file khác nhau → không None"""
        from modules.vector_store import create_vector_store
        docs = [
            Document(
                page_content="Content from file A about genetic algorithms.",
                metadata={"source": "fileA.pdf", "page": 1}
            ),
            Document(
                page_content="Content from file B about neural networks.",
                metadata={"source": "fileB.pdf", "page": 1}
            ),
        ]
        vs = create_vector_store(docs)
        assert vs is not None

    def test_filter_reduces_results(self):
        """Filter theo 1 file → ít kết quả hơn so với không filter"""
        from modules.vector_store import create_vector_store, similarity_search_with_scores
        docs = [
            Document(page_content="Genetic algorithm crossover mutation.", metadata={"source": "fileA.pdf", "page": 1}),
            Document(page_content="Genetic algorithm selection pressure.", metadata={"source": "fileA.pdf", "page": 2}),
            Document(page_content="Neural network backpropagation gradient.", metadata={"source": "fileB.pdf", "page": 1}),
        ]
        vs = create_vector_store(docs)
        all_results = similarity_search_with_scores(vs, "genetic algorithm", top_k=5)
        file_filter = ["fileA.pdf"]
        filtered_results = [
            (doc, score) for doc, score in all_results
            if any(f in doc.metadata.get("source", "") for f in file_filter)
        ]
        assert len(filtered_results) <= len(all_results)

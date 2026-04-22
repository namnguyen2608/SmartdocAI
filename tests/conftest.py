# -*- coding: utf-8 -*-
"""
Shared pytest fixtures dùng chung cho toàn bộ test suite SmartdocAI.

Tại sao cần conftest.py?
    Tránh duplicate code: mỗi test file không cần tự tạo lại Document mẫu,
    mock vector store, hay sample chunks. Pytest tự động load file này trước
    khi chạy bất kỳ test nào trong cùng thư mục.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


# ─── Document fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_doc():
    """Một Document đơn giản với metadata đầy đủ."""
    return Document(
        page_content="Thuật toán di truyền (Genetic Algorithm) sử dụng mã hóa Label Encoding.",
        metadata={"source": "sample.pdf", "page": 1, "total_pages": 10, "file_type": "pdf"},
    )


@pytest.fixture
def sample_docs():
    """Danh sách 3 Documents từ 2 file khác nhau — dùng cho metadata filter test."""
    return [
        Document(
            page_content="Chương 1: Giới thiệu về mạng xã hội và bài toán phát hiện cộng đồng.",
            metadata={"source": "fileA.pdf", "page": 1, "total_pages": 50, "file_type": "pdf"},
        ),
        Document(
            page_content="Thuật toán GA áp dụng crossover và mutation trực tiếp trên mảng nhãn.",
            metadata={"source": "fileA.pdf", "page": 12, "total_pages": 50, "file_type": "pdf"},
        ),
        Document(
            page_content="FAISS (Facebook AI Similarity Search) cho phép tìm kiếm vector hiệu quả.",
            metadata={"source": "fileB.pdf", "page": 3, "total_pages": 20, "file_type": "pdf"},
        ),
    ]


@pytest.fixture
def sample_chunks():
    """5 chunks dùng cho hybrid search và reranker test."""
    texts = [
        "Label Encoding là phương pháp mã hóa lời giải trong thuật toán di truyền.",
        "Modularity (Q) là chỉ số đánh giá chất lượng phát hiện cộng đồng.",
        "Crossover kết hợp hai chromosome cha mẹ để tạo ra offspring mới.",
        "FAISS hỗ trợ tìm kiếm vector với hàng triệu chiều một cách hiệu quả.",
        "Self-RAG cho phép LLM tự đánh giá độ tin cậy của câu trả lời.",
    ]
    return [
        Document(
            page_content=text,
            metadata={"source": f"doc{i}.pdf", "page": i + 1},
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def doc_score_pairs(sample_chunks):
    """Danh sách (Document, float) giả lập kết quả từ FAISS similarity search."""
    scores = [0.92, 0.85, 0.78, 0.71, 0.63]
    return list(zip(sample_chunks, scores))


# ─── Mock fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store(sample_docs):
    """
    Mock FAISS vector store — không cần load embedding model thật.
    Trả về sample_docs khi gọi similarity_search hoặc similarity_search_with_score.
    """
    vs = MagicMock()
    vs.similarity_search.return_value = sample_docs
    vs.similarity_search_with_score.return_value = [
        (doc, 0.85 - i * 0.05) for i, doc in enumerate(sample_docs)
    ]
    return vs


@pytest.fixture
def mock_llm():
    """
    Mock ChatOllama — không cần Ollama đang chạy.
    Trả về response giả lập cho bất kỳ chain.invoke() nào.
    """
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Đây là câu trả lời giả lập từ mock LLM."
    llm.invoke.return_value = mock_response
    return llm


@pytest.fixture
def sample_chat_history():
    """Lịch sử hội thoại mẫu với 2 turns — dùng cho Conversational RAG test."""
    return [
        {"role": "user", "content": "Thuật toán GA là gì?"},
        {"role": "assistant", "content": "GA là thuật toán tiến hóa lấy cảm hứng từ chọn lọc tự nhiên."},
        {"role": "user", "content": "Nó được áp dụng vào đâu?"},
        {"role": "assistant", "content": "GA được áp dụng vào tối ưu hóa tổ hợp, lập lịch, và phát hiện cộng đồng."},
    ]

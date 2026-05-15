"""
SmartDocAI - Vector Store
Quản lý FAISS vector store: tạo, lưu, tải, tìm kiếm
"""

import os
import logging
import shutil
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import config

logger = logging.getLogger(__name__)

# Cache embedding model để tránh tải lại nhiều lần
_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Lấy hoặc khởi tạo embedding model (singleton pattern).

    Returns:
        HuggingFaceEmbeddings instance
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Đang tải embedding model: {config.EMBEDDING_MODEL}...")
        logger.info("Nếu đây là lần đầu, quá trình tải có thể mất vài phút (~470 MB).")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Đã tải embedding model thành công!")

    return _embedding_model


def create_vector_store(documents: List[Document]) -> Optional[FAISS]:
    """
    Tạo FAISS vector store từ danh sách documents.

    Args:
        documents: Danh sách Document đã được chunking

    Returns:
        FAISS vector store instance, hoặc None nếu documents rỗng
    """
    if not documents:
        logger.warning("Không có documents để tạo vector store.")
        return None

    try:
        embeddings = get_embedding_model()
        logger.info(f"Đang tạo vector store từ {len(documents)} chunks...")
        vector_store = FAISS.from_documents(documents, embeddings)
        logger.info("Đã tạo vector store thành công!")
        return vector_store

    except Exception as e:
        logger.error(f"Lỗi khi tạo vector store: {str(e)}")
        raise Exception(
            f"Không thể tạo vector store. Chi tiết: {str(e)}"
        )


def save_vector_store(vector_store: FAISS, index_name: Optional[str] = None):
    """
    Lưu FAISS vector store xuống ổ đĩa.

    Args:
        vector_store: FAISS vector store instance
        index_name: Tên index (mặc định lấy từ config)
    """
    _index_name = index_name or config.FAISS_INDEX_NAME
    save_path = os.path.join(config.VECTORSTORE_DIR, _index_name)

    try:
        vector_store.save_local(save_path)
        logger.info(f"Đã lưu vector store tại: {save_path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu vector store: {str(e)}")
        raise


def load_vector_store(index_name: Optional[str] = None) -> Optional[FAISS]:
    """
    Tải FAISS vector store từ ổ đĩa.

    Args:
        index_name: Tên index (mặc định lấy từ config)

    Returns:
        FAISS vector store instance, hoặc None nếu chưa có index
    """
    _index_name = index_name or config.FAISS_INDEX_NAME
    load_path = os.path.join(config.VECTORSTORE_DIR, _index_name)

    if not os.path.exists(load_path):
        logger.info("Chưa có vector store được lưu trước đó.")
        return None

    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.load_local(
            load_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Đã tải vector store từ: {load_path}")
        return vector_store

    except Exception as e:
        logger.error(f"Lỗi khi tải vector store: {str(e)}")
        return None


def add_documents_to_store(
    vector_store: FAISS, documents: List[Document]
) -> FAISS:
    """
    Thêm documents mới vào vector store đã có.

    Args:
        vector_store: FAISS vector store hiện tại
        documents: Danh sách Document mới cần thêm

    Returns:
        FAISS vector store đã cập nhật
    """
    if not documents:
        return vector_store

    try:
        embeddings = get_embedding_model()
        new_store = FAISS.from_documents(documents, embeddings)
        vector_store.merge_from(new_store)
        logger.info(f"Đã thêm {len(documents)} chunks vào vector store.")
        return vector_store

    except Exception as e:
        logger.error(f"Lỗi khi thêm documents: {str(e)}")
        raise


def similarity_search(
    vector_store: FAISS, query: str, top_k: Optional[int] = None
) -> List[Document]:
    """Tìm kiếm similarity và trả về danh sách Document."""
    _top_k = top_k or config.RETRIEVAL_TOP_K
    _fetch_k = config.RETRIEVAL_FETCH_K
    _lambda_mult = config.RETRIEVAL_LAMBDA_MULT

    try:
        results = vector_store.max_marginal_relevance_search(
            query,
            k=_top_k,
            fetch_k=_fetch_k,
            lambda_mult=_lambda_mult,
        )

        logger.info(f"Tìm thấy {len(results)} kết quả (fetch_k={_fetch_k}) cho query: '{query[:50]}...'")
        return results

    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm: {str(e)}")
        return []


def similarity_search_with_scores(
    vector_store: FAISS, query: str, top_k: Optional[int] = None
) -> List[tuple]:
    """
    Tìm kiếm similarity và trả về danh sách (Document, relevance_score).

    Score được chuẩn hoá về [0, 1] — số càng cao càng liên quan.
    Sử dụng similarity_search_with_relevance_scores của FAISS.

    Args:
        vector_store: FAISS vector store instance
        query: Câu truy vấn
        top_k: Số kết quả tối đa

    Returns:
        Danh sách tuple (Document, float) sắp xếp theo score giảm dần
    """
    _top_k = top_k or config.RETRIEVAL_TOP_K

    try:
        # similarity_search_with_relevance_scores trả về score đã chuẩn hoá [0,1]
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=_top_k,
        )
        logger.info(
            f"[with_scores] Tìm thấy {len(results)} kết quả cho query: '{query[:50]}...'"
        )
        return results  # list of (Document, float)

    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm với score: {str(e)}")
        # Fallback: dùng MMR không có score, gán score = 0.0
        docs = similarity_search(vector_store, query, top_k)
        return [(doc, 0.0) for doc in docs]


def clear_vector_store(index_name: Optional[str] = None) -> bool:
    """
    Xóa FAISS index trên ổ đĩa.

    Returns:
        True nếu đã xóa hoặc không tồn tại, False nếu lỗi.
    """
    _index_name = index_name or config.FAISS_INDEX_NAME
    load_path = os.path.join(config.VECTORSTORE_DIR, _index_name)
    try:
        if os.path.exists(load_path):
            shutil.rmtree(load_path)
            logger.info(f"Đã xóa vector store tại: {load_path}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi xóa vector store: {str(e)}")
        return False

# ============================================================
# Q7 — Hybrid Search (BM25 + Ensemble)
# ============================================================

# Cache BM25 retriever
_bm25_retriever_cache = None

def create_bm25_retriever(documents: List[Document], top_k: Optional[int] = None):
    """Tạo BM25Retriever từ danh sách documents."""
    global _bm25_retriever_cache
    try:
        from langchain_community.retrievers import BM25Retriever
        _k = top_k or config.HYBRID_TOP_K
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = _k
        _bm25_retriever_cache = retriever
        logger.info(f"Đã tạo BM25 retriever với {len(documents)} documents, k={_k}")
        return retriever
    except Exception as e:
        logger.error(f"Lỗi khi tạo BM25 retriever: {str(e)}")
        return None


def get_cached_bm25_retriever():
    """Lấy BM25 retriever đã cache."""
    return _bm25_retriever_cache


def create_ensemble_retriever(vector_store: FAISS, bm25_retriever):
    """Tạo EnsembleRetriever kết hợp FAISS và BM25."""
    try:
        from langchain_classic.retrievers import EnsembleRetriever
        faiss_retriever = vector_store.as_retriever(
            search_type=config.RETRIEVAL_SEARCH_TYPE,
            search_kwargs={
                "k": config.HYBRID_TOP_K,
                "fetch_k": config.RETRIEVAL_FETCH_K,
                "lambda_mult": config.RETRIEVAL_LAMBDA_MULT,
            },
        )
        ensemble = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[config.HYBRID_VECTOR_WEIGHT, config.HYBRID_BM25_WEIGHT],
        )
        logger.info("Đã tạo EnsembleRetriever (FAISS + BM25)")
        return ensemble
    except Exception as e:
        logger.error(f"Lỗi khi tạo EnsembleRetriever: {str(e)}")
        return None
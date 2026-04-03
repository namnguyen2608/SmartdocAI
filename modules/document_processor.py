"""
SmartDocAI - Document Processor
Xử lý tài liệu PDF: đọc, trích xuất văn bản, chia nhỏ (chunking)
"""

import os
import logging
from typing import List, Optional

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str, source_name: Optional[str] = None) -> List[Document]:
    """
    Đọc file PDF và trích xuất văn bản từ từng trang.

    Args:
        file_path: Đường dẫn tới file PDF

    Returns:
        Danh sách Document objects, mỗi object chứa nội dung 1 trang
        kèm metadata (source, page)

    Raises:
        FileNotFoundError: Khi file không tồn tại
        Exception: Khi file PDF bị hỏng hoặc không thể đọc
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    documents = []
    file_name = source_name or os.path.basename(file_path)

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"Đang đọc file '{file_name}' - {total_pages} trang")

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                doc = Document(
                    page_content=text.strip(),
                    metadata={
                        "source": file_name,
                        "page": page_num + 1,
                        "total_pages": total_pages,
                    },
                )
                documents.append(doc)

        if not documents:
            logger.warning(
                f"File '{file_name}' không chứa văn bản có thể trích xuất."
            )

        logger.info(
            f"Đã trích xuất {len(documents)} trang văn bản từ '{file_name}'"
        )

    except Exception as e:
        logger.error(f"Lỗi khi đọc file PDF '{file_name}': {str(e)}")
        raise Exception(
            f"Không thể đọc file PDF '{file_name}'. "
            f"File có thể bị hỏng hoặc được bảo vệ bằng mật khẩu. "
            f"Chi tiết: {str(e)}"
        )

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Chia nhỏ danh sách Document thành các chunks.

    Args:
        documents: Danh sách Document cần chia nhỏ
        chunk_size: Kích thước mỗi chunk (mặc định lấy từ config)
        chunk_overlap: Số ký tự overlap giữa các chunk (mặc định lấy từ config)

    Returns:
        Danh sách Document đã được chia nhỏ
    """
    if not documents:
        return []

    _chunk_size = chunk_size or config.CHUNK_SIZE
    _chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(
        f"Đã chia {len(documents)} trang thành {len(chunks)} chunks "
        f"(chunk_size={_chunk_size}, overlap={_chunk_overlap})"
    )

    return chunks


def process_uploaded_file(file_path: str) -> List[Document]:
    """
    Pipeline hoàn chỉnh: Đọc PDF → Trích xuất → Chunking.

    Args:
        file_path: Đường dẫn tới file PDF

    Returns:
        Danh sách Document chunks đã xử lý
    """
    # Bước 1: Trích xuất văn bản
    raw_documents = extract_text_from_pdf(file_path)

    if not raw_documents:
        return []

    # Bước 2: Chia nhỏ văn bản
    chunks = split_documents(raw_documents)

    return chunks

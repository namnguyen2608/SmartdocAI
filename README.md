# 🧠 SmartDocAI — Trợ lý Tài liệu Thông minh

Hệ thống chatbot RAG (Retrieval-Augmented Generation) chạy **hoàn toàn offline**, hỗ trợ hỏi đáp thông minh dựa trên nội dung tài liệu PDF.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5:7b-green)

---

## 🚀 Tính năng chính

| Tính năng | Mô tả |
|---|---|
| 📄 **Upload PDF** | Tải nhiều file PDF cùng lúc |
| 🔍 **Tìm kiếm ngữ nghĩa** | Tự động chunking, indexing và truy xuất thông tin liên quan |
| 🤖 **Hỏi đáp AI** | Trả lời câu hỏi dựa trên nội dung tài liệu bằng Qwen2.5:7b |
| 🌐 **Đa ngôn ngữ** | Tự động nhận diện và trả lời bằng Tiếng Việt hoặc Tiếng Anh |
| 💬 **Lịch sử chat** | Hiển thị lịch sử trò chuyện trực quan |
| ⚠️ **Xử lý lỗi** | Thông báo rõ ràng khi gặp sự cố |

---

## 🛠️ Công nghệ sử dụng

- **LLM**: Qwen2.5:7b qua [Ollama](https://ollama.com)
- **Framework AI**: LangChain
- **Embedding**: MPNet (`all-mpnet-base-v2`)
- **Vector DB**: FAISS
- **Xử lý PDF**: PyPDF2
- **Giao diện**: Streamlit

---

## 📦 Cài đặt

### 1. Cài đặt Ollama

Tải và cài đặt Ollama từ: https://ollama.com

Pull model Qwen2.5:7b:
```bash
ollama pull qwen2.5:7b
```

### 2. Cài đặt dependencies Python

```bash
cd SmartDocAI
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Truy cập: http://localhost:8501

---

## 📖 Hướng dẫn sử dụng

1. **Đảm bảo Ollama đang chạy** — Kiểm tra trạng thái ở sidebar
2. **Tải file PDF** — Click "Browse files" ở sidebar, chọn 1 hoặc nhiều file PDF
3. **Xử lý tài liệu** — Click "🚀 Xử lý tài liệu" và chờ hoàn tất
4. **Đặt câu hỏi** — Gõ câu hỏi ở ô chat bên dưới
5. **Xem câu trả lời** — AI sẽ trả lời kèm nguồn tham khảo (file, trang)

---

## 📁 Cấu trúc dự án

```
SmartDocAI/
├── app.py                      # Giao diện Streamlit
├── config.py                   # Cấu hình hệ thống
├── requirements.txt            # Dependencies
├── README.md
├── modules/
│   ├── document_processor.py   # Xử lý PDF, chunking
│   ├── vector_store.py         # FAISS vector store
│   ├── rag_chain.py            # LangChain RAG pipeline
│   └── language_detector.py    # Nhận diện ngôn ngữ
├── data/uploads/               # File PDF đã tải lên
└── vectorstore/                # FAISS index
```

---

## ⚙️ Cấu hình

Chỉnh sửa `config.py` để thay đổi:

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Model LLM |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Model embedding |
| `CHUNK_SIZE` | `1000` | Kích thước chunk |
| `CHUNK_OVERLAP` | `200` | Overlap giữa chunks |
| `RETRIEVAL_TOP_K` | `4` | Số chunks trả về khi tìm kiếm |

---

## 🧪 Kiểm thử (Testing)

### Cài đặt pytest

```bash
pip install pytest
```

### Chạy unit tests (không cần Ollama)

```bash
pytest tests/ -v -m "not integration"
```

### Chạy tất cả tests (cần Ollama + embedding model)

```bash
pytest tests/ -v
```

### Chạy integration tests riêng lẻ

```bash
pytest tests/ -v -m integration
```

### Chạy theo module cụ thể

```bash
pytest tests/test_language_detector.py -v
pytest tests/test_document_processor.py -v
pytest tests/test_rag_chain.py -v
pytest tests/test_vector_store.py -v
```

---

## 🏗️ Kiến trúc hệ thống

```
User (Streamlit UI - app.py)
       │
       ▼
  language_detector.py
  → Phát hiện ngôn ngữ câu hỏi (vi / en)
       │
       ▼
  rag_chain.py (ask_question)
  → Gọi LLM qua Ollama
  → Hybrid retrieval (BM25 + FAISS)
       │
       ├──▶ vector_store.py
       │    → FAISS index (similarity search)
       │    → Embedding: MPNet multilingual
       │
       ├──▶ BM25Retriever
       │    → Sparse keyword matching
       │
       └──▶ EnsembleRetriever
            → FAISS 60% + BM25 40%
            → MMR reranking
       │
       ▼
  document_processor.py
  → Extract text từ PDF/DOCX
  → Chunking (size=1000, overlap=200)
```

---

## ⚙️ Cấu hình chi tiết (Configuration Reference)

| Nhóm | Tham số | Mặc định | Mô tả |
|---|---|---|---|
| **LLM** | `OLLAMA_MODEL` | `qwen2.5:7b` | Model LangChain |
| **LLM** | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| **LLM** | `LLM_TEMPERATURE` | `0.1` | Độ sáng tạo (0=xác định, 1=ngẫu nhiên) |
| **Embedding** | `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Model embedding 768-dim |
| **Chunking** | `CHUNK_SIZE` | `1000` | Số ký tự mỗi chunk |
| **Chunking** | `CHUNK_OVERLAP` | `200` | Ký tự chồng lặp giữa chunks |
| **Retrieval** | `TOP_K` | `3` | Số chunk FAISS trả về |
| **Retrieval** | `FETCH_K` | `30` | Số chunk FAISS lấy trước khi MMR rerank |
| **Hybrid** | `VECTOR_WEIGHT` | `0.6` | Trọng số FAISS trong hybrid search |
| **Hybrid** | `BM25_WEIGHT` | `0.4` | Trọng số BM25 trong hybrid search |

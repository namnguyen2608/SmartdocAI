"""
SmartDocAI — Benchmark: So sánh latency FAISS / BM25 / Hybrid Search
Chạy: python benchmark_retrieval.py
"""

import time
import random
import string
import sys
import os
import pickle

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    print(f"[ERROR] Thiếu thư viện: {e}")
    print("Hãy kích hoạt virtualenv và chạy lại.")
    sys.exit(1)

CHUNK_SIZES   = [1000, 3000, 5000, 10000]   # số lượng chunk cần test
NUM_RUNS      = 5                        # số lần đo mỗi query để lấy trung bình
HYBRID_TOP_K  = 5                        # khớp config.HYBRID_TOP_K
VECTOR_WEIGHT = 0.6                      # khớp config.HYBRID_VECTOR_WEIGHT
BM25_WEIGHT   = 0.4                      # khớp config.HYBRID_BM25_WEIGHT

CACHE_DIR     = "benchmark_cache"        # thư mục lưu FAISS index & docs

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# 5 câu truy vấn mẫu về AI và RAG (giống mô tả trong báo cáo)
QUERIES = [
    "Giải thích kiến trúc Retrieval-Augmented Generation",
    "Mô hình embedding đa ngôn ngữ hoạt động như thế nào?",
    "So sánh FAISS và BM25 trong tìm kiếm tài liệu",
    "Cách triển khai LLM cục bộ với Ollama",
    "Phương pháp chunking văn bản tối ưu cho tiếng Việt",
]

AI_TERMS = [
    "retrieval augmented generation", "vector embedding", "semantic search",
    "large language model", "transformer architecture", "attention mechanism",
    "FAISS index", "BM25 keyword", "chunk document", "cosine similarity",
    "dense retrieval", "sparse retrieval", "hybrid search", "ensemble method",
    "natural language processing", "text chunking", "information retrieval",
    "Ollama runtime", "LangChain framework", "sentence transformers",
    "tiếng Việt", "đa ngôn ngữ", "mô hình ngôn ngữ", "truy xuất thông tin",
    "phân đoạn văn bản", "tìm kiếm ngữ nghĩa", "vector store", "chỉ mục",
]

def generate_synthetic_chunk(idx: int) -> Document:
    """Tạo một chunk giả lập với nội dung liên quan đến AI/RAG."""
    num_terms = random.randint(10, 20)
    terms = random.choices(AI_TERMS, k=num_terms)
    filler = " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(random.randint(30, 60))
    )
    content = f"Đoạn {idx}: {' '.join(terms)}. {filler}"
    return Document(
        page_content=content,
        metadata={"source": f"doc_{idx % 5}.pdf", "chunk_id": idx}
    )

def generate_documents(n: int):
    random.seed(42)
    return [generate_synthetic_chunk(i) for i in range(n)]

def get_or_build_faiss(n_chunks: int, embeddings):
    """Load FAISS index + docs từ cache; nếu chưa có thì build và lưu."""
    faiss_path = os.path.join(CACHE_DIR, f"faiss_{n_chunks}")
    docs_path  = os.path.join(CACHE_DIR, f"docs_{n_chunks}.pkl")

    if os.path.exists(faiss_path) and os.path.exists(docs_path):
        print(f"  Đang tải từ cache: {n_chunks} chunks...")
        with open(docs_path, "rb") as f:
            docs = pickle.load(f)
        vs = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        print(f"  ✓ Tải cache xong.")
    else:
        docs = generate_documents(n_chunks)
        print(f"  Đã tạo {n_chunks} synthetic chunks.")
        print(f"  Đang build FAISS index ({n_chunks} vectors)...")
        vs = FAISS.from_documents(docs, embeddings)
        os.makedirs(CACHE_DIR, exist_ok=True)
        vs.save_local(faiss_path)
        with open(docs_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"  ✓ Đã lưu cache → {faiss_path}")

    return vs, docs

def measure_latency_ms(retriever, query: str, runs: int) -> float:
    """Đo thời gian trung bình (ms) của retriever trên một query, chạy `runs` lần."""
    # Warmup: 1 lần chạy không tính để loại bỏ cold-start artifact
    retriever.invoke(query)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        retriever.invoke(query)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return round(sum(times) / len(times), 2)

def avg_over_queries(retriever, queries: list, runs: int) -> float:
    """Trung bình latency (ms) qua tất cả các queries."""
    all_ms = [measure_latency_ms(retriever, q, runs) for q in queries]
    return round(sum(all_ms) / len(all_ms), 2)

def main():
    print("=" * 60)
    print("  SmartDocAI — Benchmark Retrieval Latency")
    print("=" * 60)
    print(f"\nĐang tải embedding model: {EMBEDDING_MODEL}")
    print("(Lần đầu sẽ mất ~30-60 giây để download/load model...)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✓ Đã tải embedding model.\n")

    # Warmup toàn bộ pipeline: chạy 1 lần dummy để loại cold-start
    print("Đang warmup pipeline...")
    _dummy_docs = [Document(page_content="warmup", metadata={})]
    _dummy_vs = FAISS.from_documents(_dummy_docs, embeddings)
    _dummy_ret = _dummy_vs.as_retriever(search_kwargs={"k": 1})
    _dummy_ret.invoke("warmup query")
    print("✓ Warmup xong.\n")

    NUM_PASSES = 10  # chạy 10 lần lấy average để ổn định số liệu
    all_passes = []

    for pass_num in range(1, NUM_PASSES + 1):
        print(f"════ PASS {pass_num}/{NUM_PASSES} ════")
        pass_results = []

        for n_chunks in CHUNK_SIZES:
            print(f"  ─── {n_chunks} chunks ───")

            vs, docs = get_or_build_faiss(n_chunks, embeddings)

            faiss_retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": HYBRID_TOP_K, "fetch_k": 30, "lambda_mult": 0.7},
            )
            faiss_ms = avg_over_queries(faiss_retriever, QUERIES, NUM_RUNS)
            print(f"    FAISS  : {faiss_ms:7.2f} ms")

            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = HYBRID_TOP_K
            bm25_ms = avg_over_queries(bm25_retriever, QUERIES, NUM_RUNS)
            print(f"    BM25   : {bm25_ms:7.2f} ms")

            ensemble = EnsembleRetriever(
                retrievers=[faiss_retriever, bm25_retriever],
                weights=[VECTOR_WEIGHT, BM25_WEIGHT],
            )
            hybrid_ms = avg_over_queries(ensemble, QUERIES, NUM_RUNS)
            print(f"    Hybrid : {hybrid_ms:7.2f} ms")

            pass_results.append((n_chunks, faiss_ms, bm25_ms, hybrid_ms))

        all_passes.append(pass_results)
        print()

    # Average qua các passes
    results = []
    for i, n_chunks in enumerate(CHUNK_SIZES):
        f_avg = round(sum(p[i][1] for p in all_passes) / NUM_PASSES, 2)
        b_avg = round(sum(p[i][2] for p in all_passes) / NUM_PASSES, 2)
        h_avg = round(sum(p[i][3] for p in all_passes) / NUM_PASSES, 2)
        results.append((n_chunks, f_avg, b_avg, h_avg))

    print(f"\n(Kết quả sau khi average qua {NUM_PASSES} passes)")

    print("\n" + "=" * 60)
    print("  KẾT QUẢ BENCHMARK (đơn vị: ms)")
    print("=" * 60)
    print(f"{'Chunks':>8} | {'FAISS (Semantic)':>18} | {'BM25 (Keyword)':>16} | {'Hybrid (Ensemble)':>18}")
    print("-" * 70)
    for n, f, b, h in results:
        print(f"{n:>8} | {f:>18.2f} | {b:>16.2f} | {h:>18.2f}")

    print("\n  (Trung bình qua 5 queries x 5 lần đo mỗi query)")
    print("  Cấu hình: MMR k=5, fetch_k=30; BM25 k=5; Hybrid weights=[0.6, 0.4]")
    print("\n  Sao chép bảng markdown cho báo cáo:")
    print()
    print("| Số lượng Chunk | FAISS (Semantic) | BM25 (Keyword) | Hybrid (Ensemble) |")
    print("| :---: | :---: | :---: | :---: |")
    for n, f, b, h in results:
        print(f"| **{n}** | {f} | {b} | {h} |")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        chunks_list = [r[0] for r in results]
        faiss_list  = [r[1] for r in results]
        bm25_list   = [r[2] for r in results]
        hybrid_list = [r[3] for r in results]

        y_max = max(max(faiss_list), max(bm25_list), max(hybrid_list))
        y_top = max(y_max * 1.25, 10)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(chunks_list, faiss_list,  marker="o", color="#1f77b4", linewidth=2,
                label="FAISS (Semantic)")
        ax.plot(chunks_list, bm25_list,   marker="s", color="#ff7f0e", linewidth=2,
                label="BM25 (Keyword)")
        ax.plot(chunks_list, hybrid_list, marker="^", color="#2ca02c", linewidth=2,
                label="Hybrid (Ensemble)")

        for x, y in zip(chunks_list, faiss_list):
            ax.annotate(f"{y}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, color="#1f77b4")
        for x, y in zip(chunks_list, bm25_list):
            ax.annotate(f"{y}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, color="#ff7f0e")
        for x, y in zip(chunks_list, hybrid_list):
            ax.annotate(f"{y}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5, color="#2ca02c")

        ax.set_title("Thời gian truy xuất theo kích thước dữ liệu (Latency vs Data Size)",
                     fontsize=13, pad=14)
        ax.set_xlabel("Số lượng Chunk (Data Size)", fontsize=11)
        ax.set_ylabel("Thời gian phản hồi trung bình (ms)", fontsize=11)
        ax.set_xticks(chunks_list)
        ax.set_xlim(chunks_list[0] - 50, chunks_list[-1] + 100)
        ax.set_ylim(-2, y_top)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.grid(True, which="minor", linestyle=":",  linewidth=0.4, alpha=0.4)
        ax.legend(loc="upper left", fontsize=10)
        fig.tight_layout()

        out_png = "benchmark_chart.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"\n✓ Đã lưu biểu đồ: {out_png}")
    except ImportError:
        print("\n[INFO] matplotlib chưa được cài — bỏ qua vẽ biểu đồ.")
        print("       Cài bằng: pip install matplotlib")

if __name__ == "__main__":
    main()

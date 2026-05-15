"""
benchmark_self_rag.py — Đo thời gian end-to-end RAG chuẩn vs RAG + Self-RAG
Chạy: python benchmark_self_rag.py
Kết quả in ra terminal và ghi vào benchmark_results.txt
"""

import time
import sys
import os
import statistics
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, không cần GUI
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, không cần GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document
from modules.rag_chain import get_llm, check_ollama_connection, format_context
from modules.self_rag import rewrite_query, grade_document_relevance, grade_answer
import config

# ============================================================
# Tài liệu giả lập — cố định để đảm bảo kết quả tái lặp được
# ============================================================

FAKE_DOCS = [
    Document(
        page_content=(
            "RAG (Retrieval-Augmented Generation) là kiến trúc kết hợp tìm kiếm tài liệu "
            "với mô hình ngôn ngữ lớn. Hệ thống truy xuất các đoạn liên quan từ kho tài liệu "
            "rồi đưa vào context cho LLM sinh câu trả lời dựa trên dữ liệu thực tế."
        ),
        metadata={"source": "rag_overview.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "FAISS (Facebook AI Similarity Search) là thư viện tìm kiếm vector hiệu năng cao. "
            "Với 10.000 chunks và embedding 768 chiều, FAISS có thể trả kết quả trong vòng 1–2ms "
            "sau khi đã encode câu hỏi. Bước encode chiếm ~35ms, là bottleneck chính."
        ),
        metadata={"source": "faiss_docs.pdf", "page": 3},
    ),
    Document(
        page_content=(
            "BM25 là thuật toán tìm kiếm từ khóa dựa trên TF-IDF. Điểm mạnh là khớp "
            "chính xác thuật ngữ chuyên ngành. Điểm yếu là không hiểu paraphrase hay "
            "đồng nghĩa. Thời gian tìm kiếm tăng tuyến tính theo số lượng tài liệu."
        ),
        metadata={"source": "bm25_theory.pdf", "page": 2},
    ),
    Document(
        page_content=(
            "Hybrid Search kết hợp FAISS và BM25 qua Reciprocal Rank Fusion (RRF). "
            "Trọng số mặc định FAISS=0.6, BM25=0.4. Phương pháp này cho kết quả toàn diện "
            "hơn vì bù đắp điểm yếu của từng phương pháp đơn lẻ."
        ),
        metadata={"source": "hybrid_search.pdf", "page": 5},
    ),
    Document(
        page_content=(
            "Self-RAG triển khai ba tầng kiểm soát: Query Expansion sinh 3 phiên bản câu hỏi, "
            "Relevance Grading lọc document không liên quan, Answer Grading kiểm tra "
            "hallucination và trả về confidence score trong khoảng [0.0, 1.0]."
        ),
        metadata={"source": "self_rag_paper.pdf", "page": 7},
    ),
]

# ============================================================
# Kịch bản câu hỏi
# ============================================================

SCENARIOS = [
    {
        "label": "Câu hỏi đơn giản",
        "desc": "1 sự kiện, trả lời từ 1 đoạn",
        "question": "FAISS là gì?",
        "docs": FAKE_DOCS[:1],
    },
    {
        "label": "Câu hỏi trung bình",
        "desc": "Tổng hợp 2–3 đoạn tài liệu",
        "question": "So sánh FAISS và BM25 về ưu nhược điểm.",
        "docs": FAKE_DOCS[:3],
    },
    {
        "label": "Câu hỏi phức tạp",
        "desc": "Đa bước, yêu cầu lý luận nhiều đoạn",
        "question": (
            "Tại sao Hybrid Search lại cho kết quả tốt hơn FAISS hoặc BM25 đơn lẻ, "
            "và Self-RAG giúp cải thiện chất lượng như thế nào?"
        ),
        "docs": FAKE_DOCS,
    },
]

N_RUNS = 3  # số lần đo mỗi kịch bản (tăng lên 5 nếu muốn chính xác hơn)

# ============================================================
# Hàm đo RAG chuẩn
# ============================================================

def measure_standard_rag(question: str, docs: list, llm) -> float:
    """Đo thời gian RAG chuẩn: format context → gọi LLM → nhận kết quả."""
    from langchain_core.prompts import ChatPromptTemplate

    RAG_TEMPLATE = (
        "Bạn là SmartDocAI. Trả lời câu hỏi dựa trên CONTEXT bên dưới.\n\n"
        "CONTEXT:\n{context}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
    )
    context = format_context(docs)
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = prompt | llm

    t0 = time.perf_counter()
    response = chain.invoke({"context": context, "question": question})
    elapsed = time.perf_counter() - t0
    return elapsed, response.content

# ============================================================
# Hàm đo RAG + Self-RAG
# ============================================================

def measure_self_rag(question: str, docs: list, llm) -> float:
    """Đo thời gian RAG + Self-RAG đầy đủ 3 tầng."""
    from langchain_core.prompts import ChatPromptTemplate

    RAG_TEMPLATE = (
        "Bạn là SmartDocAI. Trả lời câu hỏi dựa trên CONTEXT bên dưới.\n\n"
        "CONTEXT:\n{context}\n\nCÂU HỎI: {question}\n\nTRẢ LỜI:"
    )

    t0 = time.perf_counter()

    # Tầng 1: Query Expansion
    t1_start = time.perf_counter()
    _ = rewrite_query(question, llm)
    t1 = time.perf_counter() - t1_start

    # Tầng 2: Relevance Grading (grade từng doc)
    t2_start = time.perf_counter()
    relevant_docs = [d for d in docs if grade_document_relevance(question, d, llm)]
    if not relevant_docs:
        relevant_docs = docs  # fallback: giữ tất cả nếu grade hết KHÔNG
    t2 = time.perf_counter() - t2_start

    # Generation (RAG chuẩn trên docs đã lọc)
    tgen_start = time.perf_counter()
    context = format_context(relevant_docs)
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    answer = response.content
    tgen = time.perf_counter() - tgen_start

    # Tầng 3: Answer Grading
    t3_start = time.perf_counter()
    _ = grade_answer(question, context, answer, llm)
    t3 = time.perf_counter() - t3_start

    total = time.perf_counter() - t0

    breakdown = {
        "query_expansion": round(t1, 2),
        "relevance_grading": round(t2, 2),
        "generation": round(tgen, 2),
        "answer_grading": round(t3, 2),
        "docs_graded": len(docs),
        "docs_kept": len(relevant_docs),
    }
    return total, breakdown, answer

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  SmartDocAI — Self-RAG Benchmark")
    print("=" * 60)

    # Kiểm tra Ollama
    print("\n[1/2] Kiểm tra kết nối Ollama...", end=" ", flush=True)
    if not check_ollama_connection():
        print("THẤT BẠI")
        print("  → Khởi động Ollama: ollama serve")
        print(f"  → Pull model:       ollama pull {config.OLLAMA_MODEL}")
        sys.exit(1)
    print("OK")

    print("[2/2] Khởi tạo LLM...", end=" ", flush=True)
    llm = get_llm()
    print("OK\n")

    results = []

    for scenario in SCENARIOS:
        label = scenario["label"]
        question = scenario["question"]
        docs = scenario["docs"]

        print(f"{'─'*60}")
        print(f"Kịch bản : {label}")
        print(f"Mô tả    : {scenario['desc']}")
        print(f"Câu hỏi  : {question[:70]}...")
        print(f"Docs     : {len(docs)} đoạn")
        print(f"Số lần đo: {N_RUNS} lần\n")

        rag_times = []
        self_rag_times = []
        breakdowns = []

        for run in range(1, N_RUNS + 1):
            print(f"  Lần {run}/{N_RUNS}:", end=" ", flush=True)

            # RAG chuẩn
            t_rag, _ = measure_standard_rag(question, docs, llm)
            print(f"RAG={t_rag:.1f}s", end=" ", flush=True)
            rag_times.append(t_rag)

            # RAG + Self-RAG
            t_self, breakdown, _ = measure_self_rag(question, docs, llm)
            print(f"Self-RAG={t_self:.1f}s  [exp={breakdown['query_expansion']}s grade={breakdown['relevance_grading']}s gen={breakdown['generation']}s ans={breakdown['answer_grading']}s]")
            self_rag_times.append(t_self)
            breakdowns.append(breakdown)

        avg_rag = statistics.mean(rag_times)
        avg_self = statistics.mean(self_rag_times)
        avg_overhead = avg_self - avg_rag

        avg_expansion = statistics.mean(b["query_expansion"] for b in breakdowns)
        avg_grading = statistics.mean(b["relevance_grading"] for b in breakdowns)
        avg_gen = statistics.mean(b["generation"] for b in breakdowns)
        avg_ans = statistics.mean(b["answer_grading"] for b in breakdowns)

        print(f"\n  KẾT QUẢ TRUNG BÌNH ({N_RUNS} lần):")
        print(f"    RAG chuẩn      : {avg_rag:.1f} s")
        print(f"    RAG + Self-RAG : {avg_self:.1f} s")
        print(f"    Overhead        : +{avg_overhead:.1f} s")
        print(f"    Breakdown overhead: expansion={avg_expansion:.1f}s | relevance_grade={avg_grading:.1f}s | answer_grade={avg_ans:.1f}s")

        results.append({
            "label": label,
            "desc": scenario["desc"],
            "avg_rag": round(avg_rag, 1),
            "avg_self_rag": round(avg_self, 1),
            "overhead": round(avg_overhead, 1),
            "breakdown": {
                "query_expansion": round(avg_expansion, 1),
                "relevance_grading": round(avg_grading, 1),
                "generation": round(avg_gen, 1),
                "answer_grading": round(avg_ans, 1),
            },
        })

    # In bảng tổng kết
    print(f"\n{'='*60}")
    print("  BẢNG TỔNG KẾT (số liệu thực đo)")
    print(f"{'='*60}")
    print(f"{'Kịch bản':<25} {'RAG':>8} {'Self-RAG':>10} {'Overhead':>10}")
    print(f"{'─'*25} {'─'*8} {'─'*10} {'─'*10}")
    for r in results:
        print(f"{r['label']:<25} {r['avg_rag']:>7.1f}s {r['avg_self_rag']:>9.1f}s {'+'+str(r['overhead']):>9}s")

    print(f"\n  Hardware: {config.OLLAMA_MODEL} trên CPU (không GPU)")
    print(f"  Phương pháp: time.perf_counter() end-to-end, trung bình {N_RUNS} lần/kịch bản")

    # Ghi file kết quả
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SmartDocAI — Self-RAG Benchmark Results\n")
        f.write(f"Model: {config.OLLAMA_MODEL} | N_RUNS={N_RUNS}\n")
        f.write(f"Phương pháp đo: time.perf_counter() từ đầu vào đến kết thúc grade_answer()\n\n")
        f.write(f"{'Kịch bản':<25} {'RAG(s)':>8} {'Self-RAG(s)':>12} {'Overhead(s)':>12}\n")
        f.write(f"{'─'*25} {'─'*8} {'─'*12} {'─'*12}\n")
        for r in results:
            f.write(
                f"{r['label']:<25} {r['avg_rag']:>8.1f} {r['avg_self_rag']:>12.1f} {'+'+str(r['overhead']):>12}\n"
            )
        f.write("\nBreakdown overhead trung bình:\n")
        for r in results:
            b = r["breakdown"]
            f.write(
                f"  {r['label']}: expansion={b['query_expansion']}s | "
                f"relevance_grade={b['relevance_grading']}s | "
                f"answer_grade={b['answer_grading']}s\n"
            )
    print(f"\n  Kết quả ghi vào: {out_path}")
    print("  Dùng số liệu này để cập nhật Bảng IV trong báo cáo.")

    # Vẽ chart
    chart_path = _plot_benchmark_chart(results, os.path.dirname(os.path.abspath(__file__)))
    print(f"  Chart đã lưu  : {chart_path}")

def _plot_benchmark_chart(results: list, out_dir: str) -> str:
    """Vẽ grouped bar chart (RAG vs Self-RAG) + stacked overhead breakdown."""
    # Đảm bảo font hỗ trợ Unicode/tiếng Việt
    plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
    labels = [r["label"] for r in results]
    rag_vals = [r["avg_rag"] for r in results]
    self_vals = [r["avg_self_rag"] for r in results]
    overheads = [r["overhead"] for r in results]
    exp_vals  = [r["breakdown"]["query_expansion"] for r in results]
    grade_vals = [r["breakdown"]["relevance_grading"] for r in results]
    ans_vals  = [r["breakdown"]["answer_grading"] for r in results]

    x = np.arange(len(labels))
    bar_w = 0.32

    fig, axes = plt.subplots(2, 1, figsize=(8, 11))
    fig.suptitle(
        "SmartDocAI — Self-RAG Benchmark\n"
        "qwen2.5:7b · i5-12400F · 24 GB RAM · CPU only",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0]
    bars_rag  = ax1.bar(x - bar_w / 2, rag_vals, bar_w,
                        label="RAG chuẩn", color="#4C8BF5", zorder=3)
    bars_self = ax1.bar(x + bar_w / 2, self_vals, bar_w,
                        label="RAG + Self-RAG", color="#F4A300", zorder=3)

    for bar in bars_rag:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                 f"{h:.1f}s", ha="center", va="bottom", fontsize=9)
    for bar in bars_self:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                 f"{h:.1f}s", ha="center", va="bottom", fontsize=9)

    for i, (rv, sv, ov) in enumerate(zip(rag_vals, self_vals, overheads)):
        color = "#28a745" if ov <= 0 else "#dc3545"
        sign  = "+" if ov >= 0 else ""
        ax1.annotate(
            f"{sign}{ov:.1f}s",
            xy=(x[i], max(rv, sv) + 0.8),
            ha="center", fontsize=9, color=color, fontweight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("Thời gian phản hồi (giây)", fontsize=10)
    ax1.set_title("Thời gian RAG chuẩn vs RAG + Self-RAG", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, max(self_vals) * 1.35)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    ax2 = axes[1]
    b1 = ax2.bar(x, exp_vals, bar_w * 2, label="Query Expansion",
                 color="#A8D5BA", zorder=3)
    b2 = ax2.bar(x, grade_vals, bar_w * 2, bottom=exp_vals,
                 label="Relevance Grading", color="#FFD580", zorder=3)
    bottom2 = [e + g for e, g in zip(exp_vals, grade_vals)]
    b3 = ax2.bar(x, ans_vals, bar_w * 2, bottom=bottom2,
                 label="Answer Grading", color="#F4A3A3", zorder=3)

    total_oh = [e + g + a for e, g, a in zip(exp_vals, grade_vals, ans_vals)]
    for i, tot in enumerate(total_oh):
        ax2.text(x[i], tot + 0.1, f"{tot:.1f}s",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Overhead (giây)", fontsize=10)
    ax2.set_title("Phân tích Overhead Self-RAG theo tầng", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, max(total_oh) * 1.4)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    chart_path = os.path.join(out_dir, "benchmark_self_rag_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    # Lưu thêm hinh2.png để dùng trực tiếp trong báo cáo LaTeX
    hinh2_path = os.path.join(out_dir, "hinh2.png")
    plt.savefig(hinh2_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path

if __name__ == "__main__":
    main()

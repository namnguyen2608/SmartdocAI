# -*- coding: utf-8 -*-
"""
SmartDocAI - Co-RAG (Collaborative RAG)
Kiến trúc RAG hợp tác: nhiều agent song song truy xuất từ góc độ khác nhau,
sau đó Consensus Merger bầu chọn ngữ cảnh tốt nhất.

Kiến trúc:
    Câu hỏi
       ↓
    ┌─────────────────────────────────────────────────────────┐
    │                 Co-RAG Orchestrator                     │
    ├──────────────┬──────────────┬─────────────────────────┤
    │  Agent 1:    │  Agent 2:    │  Agent 3:               │
    │  Semantic    │  Keyword     │  Conceptual             │
    │  Retriever   │  Retriever   │  Decomposition          │
    │  (FAISS MMR) │  (BM25)      │  (LLM sub-questions)    │
    └──────────────┴──────────────┴─────────────────────────┘
           ↓               ↓               ↓
    ┌─────────────────────────────────────────────────────────┐
    │                Consensus Merger                         │
    │  - Dedup + score voting                                 │
    │  - Docs được nhiều agent agree → score cao hơn          │
    │  - Lọc theo CO_RAG_MIN_VOTES nếu dùng "voting"         │
    └─────────────────────────────────────────────────────────┘
           ↓
    Final Context → LLM → Answer
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import config

logger = logging.getLogger(__name__)


# ============================================================
# Prompt cho Conceptual Decomposition Agent
# ============================================================

CO_RAG_DECOMPOSE_TEMPLATE = """Bạn là chuyên gia phân tích câu hỏi phức tạp.

Hãy phân rã câu hỏi sau thành TỐI ĐA 3 câu hỏi con đơn giản hơn để tìm kiếm thông tin.
Mỗi câu hỏi con nên hỏi về một khía cạnh cụ thể của câu hỏi gốc.

Chỉ trả về các câu hỏi con, mỗi câu một dòng, bắt đầu bằng số thứ tự.
Không giải thích thêm. Nếu câu hỏi đã đủ đơn giản, chỉ trả về chính câu hỏi đó.

Câu hỏi gốc: {question}

Các câu hỏi con:"""


# ============================================================
# Agent 1: Semantic Retriever (FAISS MMR)
# ============================================================

def semantic_retriever_agent(
    query: str,
    vector_store,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Agent 1 — Semantic Retriever dùng FAISS similarity search với relevance scores.

    Args:
        query: Câu truy vấn
        vector_store: FAISS vector store instance
        top_k: Số docs cần lấy

    Returns:
        List of (Document, score) tuples
    """
    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=top_k,
        )
        logger.info(f"[Co-RAG Agent1/Semantic] Truy xuất {len(results)} docs cho: '{query[:50]}'")
        return results
    except Exception as e:
        logger.warning(f"[Co-RAG Agent1/Semantic] Lỗi: {e}")
        return []


# ============================================================
# Agent 2: Keyword Retriever (BM25)
# ============================================================

def keyword_retriever_agent(
    query: str,
    raw_documents: List[Document],
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Agent 2 — Keyword Retriever dùng BM25 (exact/partial keyword matching).

    Args:
        query: Câu truy vấn
        raw_documents: Toàn bộ documents gốc để xây BM25 index
        top_k: Số docs cần lấy

    Returns:
        List of (Document, score) tuples — score BM25 được chuẩn hoá về [0, 1]
    """
    if not raw_documents:
        logger.warning("[Co-RAG Agent2/Keyword] Không có documents để xây BM25 index.")
        return []

    try:
        from langchain_community.retrievers import BM25Retriever

        retriever = BM25Retriever.from_documents(raw_documents)
        retriever.k = top_k
        docs = retriever.invoke(query)

        # BM25 không có score chuẩn hoá sẵn → dùng RRF formula (k=60)
        # normalized về [0,1]: score(rank) = (1/(k+1)) / (1/(k+rank)) = (k+rank)/(k+1)^-1
        # → score(rank) = (k+1) / (k+rank), rank 1 = 1.0, rank 5 ≈ 0.938
        _rrf_k = 60
        _rrf_max = 1.0 / (_rrf_k + 1)  # score lý thuyết tại rank 1
        results = []
        for i, doc in enumerate(docs):
            rrf_score = 1.0 / (_rrf_k + (i + 1))
            approx_score = round(rrf_score / _rrf_max, 4)  # normalize về [0,1]
            results.append((doc, approx_score))

        logger.info(f"[Co-RAG Agent2/Keyword] Truy xuất {len(results)} docs cho: '{query[:50]}'")
        return results

    except Exception as e:
        logger.warning(f"[Co-RAG Agent2/Keyword] Lỗi: {e}")
        return []


# ============================================================
# Agent 3: Conceptual Decomposition Agent
# ============================================================

def conceptual_decomposer_agent(
    question: str,
    vector_store,
    llm: ChatOllama,
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Agent 3 — Conceptual Decomposition: LLM phân rã câu hỏi phức tạp thành
    các câu hỏi con, truy xuất riêng cho từng câu, rồi hợp nhất kết quả.

    Args:
        question: Câu hỏi gốc
        vector_store: FAISS vector store instance
        llm: ChatOllama instance
        top_k: Số docs mỗi sub-question

    Returns:
        List of (Document, score) tuples — đã dedup
    """
    # Bước 1: LLM phân rã câu hỏi
    sub_questions = [question]  # fallback: dùng câu gốc
    try:
        prompt = ChatPromptTemplate.from_template(CO_RAG_DECOMPOSE_TEMPLATE)
        chain = prompt | llm
        response = chain.invoke({"question": question})
        raw = response.content.strip()

        parsed = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 5:
                parsed.append(cleaned)

        if parsed:
            sub_questions = parsed[:3]

        logger.info(f"[Co-RAG Agent3/Conceptual] Phân rã thành {len(sub_questions)} sub-questions")

    except Exception as e:
        logger.warning(f"[Co-RAG Agent3/Conceptual] Phân rã câu hỏi thất bại: {e}. Dùng câu gốc.")

    # Bước 2: Retrieve với mỗi sub-question, dedup
    seen_content_keys = set()
    all_results: List[Tuple[Document, float]] = []

    for sub_q in sub_questions:
        try:
            results = vector_store.similarity_search_with_relevance_scores(
                sub_q,
                k=top_k,
            )
            for doc, score in results:
                key = doc.page_content[:120]
                if key not in seen_content_keys:
                    seen_content_keys.add(key)
                    all_results.append((doc, score))
        except Exception as e:
            logger.warning(f"[Co-RAG Agent3/Conceptual] Lỗi retrieve sub-question '{sub_q[:40]}': {e}")

    logger.info(f"[Co-RAG Agent3/Conceptual] Tổng {len(all_results)} docs (sau dedup)")
    return all_results


# ============================================================
# Consensus Merger
# ============================================================

def consensus_merger(
    agent_results: Dict[str, List[Tuple[Document, float]]],
    strategy: str = "voting",
    min_votes: int = 2,
) -> List[Tuple[Document, float, int]]:
    """
    Hợp nhất kết quả từ nhiều agents thông qua voting/scoring.

    Mỗi document được nhận diện qua 120 ký tự đầu của content (fingerprint).
    Score cuối = trung bình score các agents có chứa doc đó, nhân với hệ số vote.

    Args:
        agent_results: Dict {agent_name: [(Document, score), ...]}
        strategy: "voting" — chỉ giữ docs có ≥ min_votes agents agree
                  "union"  — giữ tất cả docs từ mọi agents
                  "intersection" — chỉ giữ docs có trong MỌI agents
        min_votes: Ngưỡng tối thiểu khi strategy="voting"

    Returns:
        List of (Document, merged_score, vote_count) sắp xếp theo merged_score giảm dần
    """
    # Xây dựng bảng: fingerprint → [(agent_name, doc, score)]
    doc_registry: Dict[str, List[Tuple[str, Document, float]]] = {}

    for agent_name, results in agent_results.items():
        for doc, score in results:
            fp = doc.page_content[:120]
            if fp not in doc_registry:
                doc_registry[fp] = []
            doc_registry[fp].append((agent_name, doc, score))

    num_agents = len(agent_results)
    merged: List[Tuple[Document, float, int]] = []

    for fp, entries in doc_registry.items():
        vote_count = len(entries)
        avg_score = sum(s for _, _, s in entries) / vote_count
        # Boost score theo số lượt vote (normalized)
        vote_boost = 1.0 + (vote_count - 1) * 0.15
        merged_score = round(min(1.0, avg_score * vote_boost), 3)

        # Lấy doc từ entry đầu tiên
        representative_doc = entries[0][1]

        # Áp dụng strategy
        if strategy == "voting" and vote_count < min_votes:
            continue
        elif strategy == "intersection" and vote_count < num_agents:
            continue
        # strategy == "union": giữ tất cả

        merged.append((representative_doc, merged_score, vote_count))

    # Sắp xếp theo score giảm dần
    merged.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        f"[Co-RAG Merger] Strategy='{strategy}', Total docs: "
        f"{len(doc_registry)} → Sau filter: {len(merged)}"
    )
    return merged


# ============================================================
# Co-RAG Pipeline (Orchestrator)
# ============================================================

def co_rag_pipeline(
    question: str,
    vector_store,
    raw_documents: List[Document],
    llm: ChatOllama,
    top_k_per_agent: Optional[int] = None,
    min_votes: Optional[int] = None,
    merge_strategy: Optional[str] = None,
    enable_agent_semantic: bool = True,
    enable_agent_keyword: bool = True,
    enable_agent_conceptual: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline Co-RAG đầy đủ:
    1. Chạy song song 3 agents (Semantic, Keyword, Conceptual)
    2. Consensus Merger bầu chọn docs tốt nhất
    3. Generate answer với context đã merge

    Args:
        question: Câu hỏi của người dùng
        vector_store: FAISS vector store
        raw_documents: Toàn bộ chunks (cần cho BM25)
        llm: ChatOllama instance
        top_k_per_agent: Số docs mỗi agent lấy (mặc định từ config)
        min_votes: Ngưỡng vote tối thiểu (mặc định từ config)
        merge_strategy: Chiến lược merge (mặc định từ config)
        enable_agent_*: Bật/tắt từng agent

    Returns:
        dict với answer, sources, co_rag metadata
    """
    from modules.rag_chain import format_context, RAG_PROMPT_TEMPLATE
    from modules.language_detector import detect_language, get_language_instruction

    _top_k = top_k_per_agent or config.CO_RAG_TOP_K_PER_AGENT
    _min_votes = min_votes or config.CO_RAG_MIN_VOTES
    _strategy = merge_strategy or config.CO_RAG_MERGE_STRATEGY

    result: Dict[str, Any] = {
        "answer": "",
        "sources": [],
        "language": "vi",
        "error": None,
        # Co-RAG specific metadata
        "co_rag_agent_counts": {},   # {agent_name: num_docs_retrieved}
        "co_rag_total_before_merge": 0,
        "co_rag_total_after_merge": 0,
        "co_rag_merge_strategy": _strategy,
        "co_rag_sub_questions": [],  # sub-questions từ Conceptual Agent
    }

    try:
        language = detect_language(question)
        result["language"] = language

        # ── Bước 1: Chạy các agents ──────────────────────────────────────
        agent_results: Dict[str, List[Tuple[Document, float]]] = {}

        if enable_agent_semantic and vector_store is not None:
            agent1 = semantic_retriever_agent(question, vector_store, top_k=_top_k)
            agent_results["Semantic (FAISS)"] = agent1
            result["co_rag_agent_counts"]["Semantic (FAISS)"] = len(agent1)

        if enable_agent_keyword and raw_documents:
            agent2 = keyword_retriever_agent(question, raw_documents, top_k=_top_k)
            agent_results["Keyword (BM25)"] = agent2
            result["co_rag_agent_counts"]["Keyword (BM25)"] = len(agent2)

        if enable_agent_conceptual and vector_store is not None:
            agent3 = conceptual_decomposer_agent(question, vector_store, llm, top_k=_top_k)
            agent_results["Conceptual (LLM)"] = agent3
            result["co_rag_agent_counts"]["Conceptual (LLM)"] = len(agent3)

        if not agent_results:
            result["error"] = "Không có agent nào được kích hoạt hoặc không có dữ liệu."
            result["answer"] = "Không thể xử lý câu hỏi do chưa có tài liệu hoặc cấu hình sai."
            return result

        # ── Bước 2: Consensus Merger ─────────────────────────────────────
        total_before = sum(len(v) for v in agent_results.values())
        result["co_rag_total_before_merge"] = total_before

        merged_docs_with_votes = consensus_merger(
            agent_results,
            strategy=_strategy,
            min_votes=_min_votes,
        )

        result["co_rag_total_after_merge"] = len(merged_docs_with_votes)

        # Nếu chiến lược voting lọc sạch, fallback sang union top docs
        if not merged_docs_with_votes:
            logger.warning("[Co-RAG] Voting cho 0 docs, fallback sang union top docs.")
            merged_docs_with_votes = consensus_merger(
                agent_results,
                strategy="union",
                min_votes=1,
            )

        # Lấy top docs
        top_merged = merged_docs_with_votes[:config.RETRIEVAL_TOP_K]

        # ── Bước 3: Format context & generate ────────────────────────────
        docs_for_context = [doc for doc, _, _ in top_merged]
        context_text = format_context(docs_for_context)

        language_instruction = get_language_instruction(language)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        chain = prompt | llm
        response = chain.invoke({
            "question": question,
            "context": context_text,
            "language_instruction": language_instruction,
            "chat_history_section": "",
        })
        result["answer"] = response.content

        # ── Bước 4: Build sources metadata ───────────────────────────────
        import os as _os

        def _clean(s: str) -> str:
            return _os.path.basename(str(s or "N/A"))

        seen_keys = set()
        sources = []
        for idx, (doc, merged_score, vote_count) in enumerate(top_merged):
            fname = _clean(doc.metadata.get("source", "N/A"))
            page = doc.metadata.get("page", "N/A")
            dedup_key = (fname, page, doc.page_content[:80])
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            sources.append({
                "file": fname,
                "page": page,
                "total_pages": doc.metadata.get("total_pages"),
                "file_type": doc.metadata.get("file_type", "pdf"),
                "content": doc.page_content,
                "chunk_index": idx + 1,
                "score": merged_score,
                "vote_count": vote_count,          # Co-RAG specific: số agents đồng ý
            })
        result["sources"] = sources

        logger.info(
            f"[Co-RAG] Hoàn thành: {len(agent_results)} agents, "
            f"{total_before} docs → {len(merged_docs_with_votes)} sau merge → "
            f"{len(sources)} nguồn cuối"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Co-RAG] Pipeline lỗi: {e}")

    return result

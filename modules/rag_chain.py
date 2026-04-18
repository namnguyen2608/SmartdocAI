"""
SmartDocAI - RAG Chain
X├óy dß╗▒ng luß╗ông RAG: Retrieval ΓåÆ Augmentation ΓåÆ Generation
"""

import logging
import os
import re
from typing import Optional, Dict, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from modules.vector_store import similarity_search, similarity_search_with_scores
from modules.language_detector import detect_language, get_language_instruction
import config

logger = logging.getLogger(__name__)

# ============================================================
# Prompt Templates
# ============================================================

RAG_PROMPT_TEMPLATE = """Bß║ín l├á SmartDocAI, mß╗Öt trß╗ú l├╜ AI th├┤ng minh chuy├¬n ph├ón t├¡ch v├á trß║ú lß╗¥i c├óu hß╗Åi dß╗▒a tr├¬n nß╗Öi dung t├ái liß╗çu.

{language_instruction}

### QUY Tß║«C:
1. CHß╗ê trß║ú lß╗¥i dß╗▒a tr├¬n th├┤ng tin trong phß║ºn CONTEXT b├¬n d╞░ß╗¢i.
2. Nß║┐u CONTEXT kh├┤ng chß╗⌐a ─æß╗º th├┤ng tin ─æß╗â trß║ú lß╗¥i, h├úy n├│i r├╡ rß║▒ng th├┤ng tin kh├┤ng c├│ trong t├ái liß╗çu.
3. Tr├¡ch dß║½n nguß╗ôn (t├¬n file, sß╗æ trang) khi c├│ thß╗â.
4. Trß║ú lß╗¥i c├│ cß║Ñu tr├║c, r├╡ r├áng, dß╗à ─æß╗ìc.
5. Kh├┤ng bß╗ïa ─æß║╖t th├┤ng tin ngo├ái CONTEXT.
6. Nß║┐u c├óu hß╗Åi li├¬n quan ─æß║┐n lß╗ïch sß╗¡ hß╗Öi thoß║íi, h├úy tham chiß║┐u c├íc c├óu trß║ú lß╗¥i tr╞░ß╗¢c ─æ├│.

{chat_history_section}### CONTEXT:
{context}

### C├éU Hß╗ÄI:
{question}

### TRß║ó Lß╗£I:"""


REFORMULATE_QUESTION_TEMPLATE = """Dß╗▒a v├áo lß╗ïch sß╗¡ hß╗Öi thoß║íi b├¬n d╞░ß╗¢i v├á c├óu hß╗Åi tiß║┐p theo cß╗ºa ng╞░ß╗¥i d├╣ng, h├úy viß║┐t lß║íi c├óu hß╗Åi th├ánh mß╗Öt c├óu ho├án chß╗ënh, ─æß╗Öc lß║¡p (standalone question) ─æß╗â c├│ thß╗â t├¼m kiß║┐m trong t├ái liß╗çu m├á kh├┤ng cß║ºn ngß╗» cß║únh hß╗Öi thoß║íi.

Chß╗ë trß║ú vß╗ü c├óu hß╗Åi ─æ├ú viß║┐t lß║íi, kh├┤ng giß║úi th├¡ch th├¬m.

### Lß╗èCH Sß╗¼ Hß╗ÿI THOß║áI:
{chat_history}

### C├éU Hß╗ÄI TIß║╛P THEO:
{question}

### C├éU Hß╗ÄI ─É├â VIß║╛T Lß║áI:"""

NO_CONTEXT_PROMPT_TEMPLATE = """Bß║ín l├á SmartDocAI, mß╗Öt trß╗ú l├╜ AI th├┤ng minh.

{language_instruction}

Ng╞░ß╗¥i d├╣ng ch╞░a tß║úi t├ái liß╗çu l├¬n hß╗ç thß╗æng. H├úy th├┤ng b├ío lß╗ïch sß╗▒ rß║▒ng:
- Hß╗ì cß║ºn tß║úi file PDF l├¬n tr╞░ß╗¢c khi ─æß║╖t c├óu hß╗Åi.
- H╞░ß╗¢ng dß║½n hß╗ì bß║Ñm n├║t "Tß║úi t├ái liß╗çu" ß╗ƒ ph├¡a tr├¬n ─æß╗â tß║úi file l├¬n.

### C├éU Hß╗ÄI Cß╗ªA NG╞»ß╗£I D├ÖNG:
{question}

### TRß║ó Lß╗£I:"""


def _clean_source_name(source: str) -> str:
    """Chuß║⌐n h├│a t├¬n file nguß╗ôn ─æß╗â hiß╗ân thß╗ï th├ón thiß╗çn."""
    name = os.path.basename(str(source or "N/A"))
    if re.match(r"^tmp[a-zA-Z0-9_\\-]+\\.pdf$", name):
        return "Tai lieu da tai len (du lieu cu)"
    return name


def _build_fallback_answer(relevant_docs: list[Document], language: str) -> str:
    """Tß║ío c├óu trß║ú lß╗¥i dß╗▒ ph├▓ng tß╗½ context khi LLM kh├┤ng khß║ú dß╗Ñng."""
    if not relevant_docs:
        if language == "vi":
            return (
                "M├¼nh ch╞░a thß╗â truy cß║¡p m├┤ h├¼nh AI ─æß╗â sinh c├óu trß║ú lß╗¥i chi tiß║┐t. "
                "Hiß╗çn c┼⌐ng kh├┤ng t├¼m thß║Ñy ─æoß║ín nß╗Öi dung ph├╣ hß╗úp trong t├ái liß╗çu."
            )
        return (
            "I cannot reach the AI model right now, and no relevant document "
            "segments were found for this question."
        )

    snippets = []
    for idx, doc in enumerate(relevant_docs[:3], 1):
        source = _clean_source_name(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        short_text = " ".join(doc.page_content.split())
        if len(short_text) > 260:
            short_text = f"{short_text[:260].rstrip()}..."
        snippets.append(f"{idx}. ({source} - Trang {page}) {short_text}")

    if language == "vi":
        intro = (
            "Ollama/LLM hiß╗çn ch╞░a sß║╡n s├áng, n├¬n m├¼nh tß║ím gß╗¡i c├íc ─æoß║ín li├¬n quan nhß║Ñt "
            "─æß╗â bß║ín tham khß║úo nhanh:"
        )
    else:
        intro = (
            "The Ollama/LLM service is currently unavailable, so here are the most "
            "relevant document snippets for quick reference:"
        )

    return f"{intro}\n\n" + "\n\n".join(snippets)


def get_llm() -> ChatOllama:
    """
    Khß╗ƒi tß║ío kß║┐t nß╗æi tß╗¢i Ollama LLM.

    Returns:
        ChatOllama instance

    Raises:
        ConnectionError: Khi kh├┤ng thß╗â kß║┐t nß╗æi tß╗¢i Ollama
    """
    try:
        llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.OLLAMA_TEMPERATURE,
        )
        return llm
    except Exception as e:
        logger.error(f"Lß╗ùi kß║┐t nß╗æi Ollama: {str(e)}")
        raise ConnectionError(
            f"Kh├┤ng thß╗â kß║┐t nß╗æi tß╗¢i Ollama tß║íi {config.OLLAMA_BASE_URL}. "
            f"Vui l├▓ng kiß╗âm tra:\n"
            f"1. Ollama ─æ├ú ─æ╞░ß╗úc c├ái ─æß║╖t v├á ─æang chß║íy\n"
            f"2. Model '{config.OLLAMA_MODEL}' ─æ├ú ─æ╞░ß╗úc pull\n"
            f"   (chß║íy: ollama pull {config.OLLAMA_MODEL})\n"
            f"Chi tiß║┐t lß╗ùi: {str(e)}"
        )


def check_ollama_connection() -> bool:
    """
    Kiß╗âm tra kß║┐t nß╗æi tß╗¢i Ollama server.

    Returns:
        True nß║┐u kß║┐t nß╗æi th├ánh c├┤ng, False nß║┐u kh├┤ng
    """
    try:
        import urllib.request
        url = f"{config.OLLAMA_BASE_URL}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def _reformulate_question(question: str, chat_history: list, llm: ChatOllama) -> str:
    """
    Viß║┐t lß║íi follow-up question th├ánh c├óu hß╗Åi ─æß╗Öc lß║¡p ─æß╗â search vector store ch├¡nh x├íc h╞ín.
    Nß║┐u kh├┤ng c├│ lß╗ïch sß╗¡ hoß║╖c LLM lß╗ùi, trß║ú vß╗ü c├óu hß╗Åi gß╗æc.
    """
    if not chat_history:
        return question

    recent = [m for m in chat_history if m.get("role") in ("user", "assistant")][-6:]
    if not recent:
        return question

    history_text = ""
    for msg in recent:
        role_label = "Ng╞░ß╗¥i d├╣ng" if msg["role"] == "user" else "Trß╗ú l├╜"
        history_text += f"{role_label}: {msg['content']}\n"

    try:
        prompt = ChatPromptTemplate.from_template(REFORMULATE_QUESTION_TEMPLATE)
        chain = prompt | llm
        response = chain.invoke({"chat_history": history_text, "question": question})
        reformulated = response.content.strip()
        if reformulated:
            logger.info(f"Reformulated: '{question}' -> '{reformulated}'")
            return reformulated
    except Exception as e:
        logger.warning(f"Kh├┤ng thß╗â reformulate question: {e}")

    return question


def format_context(documents: list[Document]) -> str:
    """
    ─Éß╗ïnh dß║íng danh s├ích Document th├ánh chuß╗ùi context cho prompt.

    Args:
        documents: Danh s├ích Document tß╗½ similarity search

    Returns:
        Chuß╗ùi context ─æ├ú format
    """
    if not documents:
        return "Kh├┤ng t├¼m thß║Ñy th├┤ng tin li├¬n quan trong t├ái liß╗çu."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = _clean_source_name(doc.metadata.get("source", "N/A"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[Nguß╗ôn {i}: {source} - Trang {page}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def ask_question(
    question: str,
    vector_store=None,
    chat_history: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Xß╗¡ l├╜ c├óu hß╗Åi cß╗ºa ng╞░ß╗¥i d├╣ng qua pipeline RAG.
    """
    # QUAN TRß╗îNG: Phß║úi khß╗ƒi tß║ío biß║┐n result ngay tß╗½ ─æß║ºu
    result = {
        "answer": "",
        "sources": [],
        "language": "en",
        "error": None,
        "used_fallback": False,
    }

    # Validate đầu vào: câu hỏi không được rỗng hoặc None
    if not question or not str(question).strip():
        result["answer"] = (
            "Vui lòng nhập câu hỏi của bạn trước khi gửi."
            if detect_language(question or "") == "vi"
            else "Please enter your question before submitting."
        )
        result["error"] = "empty_question"
        return result

    question = str(question).strip()

    try:
        # B╞░ß╗¢c 1: Ph├ít hiß╗çn ng├┤n ngß╗»
        language = detect_language(question)
        result["language"] = language

        # B╞░ß╗¢c 2: Khß╗ƒi tß║ío LLM
        llm = get_llm()

        # B╞░ß╗¢c 2.5: Format lß╗ïch sß╗¡ hß╗Öi thoß║íi (Q6 - Conversational RAG)
        chat_history_section = ""
        if chat_history:
            # Lß║Ñy tß╗æi ─æa 6 turns gß║ºn nhß║Ñt (3 cß║╖p hß╗Åi-─æ├íp) ─æß╗â tr├ính qu├í d├ái
            recent_history = [m for m in chat_history if m.get("role") in ("user", "assistant")][-6:]
            if recent_history:
                lines = []
                for msg in recent_history:
                    role_label = "Ng╞░ß╗¥i d├╣ng" if msg["role"] == "user" else "Trß╗ú l├╜"
                    lines.append(f"{role_label}: {msg['content']}")
                chat_history_section = "### Lß╗èCH Sß╗¼ Hß╗ÿI THOß║áI:\n" + "\n".join(lines) + "\n\n"

        # B╞░ß╗¢c 3: Xß╗¡ l├╜ dß╗▒a tr├¬n viß╗çc c├│/kh├┤ng c├│ t├ái liß╗çu
        if vector_store is None:
            # D├╣ng prompt th├┤ng b├ío ch╞░a c├│ t├ái liß╗çu
            prompt = ChatPromptTemplate.from_template(NO_CONTEXT_PROMPT_TEMPLATE)
            language_instruction = get_language_instruction(language) 
            chain = prompt | llm
            response = chain.invoke({
                "question": question,
                "language_instruction": language_instruction,
            })
            result["answer"] = response.content
        else:
            # Pipeline RAG ─æß║ºy ─æß╗º
            # B╞░ß╗¢c 3.1: Viß║┐t lß║íi c├óu hß╗Åi nß║┐u l├á follow-up (Conversational RAG)
            search_question = _reformulate_question(question, chat_history, llm)
            doc_score_pairs = similarity_search_with_scores(vector_store, search_question)
            relevant_docs = [doc for doc, _ in doc_score_pairs]
            context = format_context(relevant_docs)

            language_instruction = get_language_instruction(language)
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

            chain = prompt | llm
            response = chain.invoke({
                "question": question,
                "context": context,
                "language_instruction": language_instruction,
                "chat_history_section": chat_history_section,
            })

            result["answer"] = response.content

            # Tr├¡ch xuß║Ñt nguß╗ôn tham khß║úo ─æß║ºy ─æß╗º (d├╣ng cho citation UI)
            seen_keys = set()
            sources = []
            for chunk_idx, (doc, score) in enumerate(doc_score_pairs):
                file_name = _clean_source_name(doc.metadata.get("source", "N/A"))
                page = doc.metadata.get("page", "N/A")
                dedup_key = (file_name, page, doc.page_content[:80])
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                sources.append({
                    "file": file_name,
                    "page": page,
                    "total_pages": doc.metadata.get("total_pages"),
                    "file_type": doc.metadata.get("file_type", "pdf"),
                    "content": doc.page_content,
                    "chunk_index": chunk_idx + 1,
                    "score": round(float(score), 3),
                })
            result["sources"] = sources

    except Exception as e:
        error_msg = f"Lß╗ùi xß╗¡ l├╜: {str(e)}"
        result["error"] = error_msg
        logger.error(error_msg)
        
        # Nß║┐u c├│ vector_store, d├╣ng fallback ─æß╗â trß║ú vß╗ü c├íc ─æoß║ín text th├┤
        if vector_store is not None:
            doc_score_pairs = similarity_search_with_scores(vector_store, question)
            relevant_docs = [doc for doc, _ in doc_score_pairs]
            result["answer"] = _build_fallback_answer(relevant_docs, result["language"])
            result["used_fallback"] = True
            # Gß║»n sources c╞í bß║ún cho fallback
            seen_keys = set()
            sources = []
            for chunk_idx, (doc, score) in enumerate(doc_score_pairs):
                file_name = _clean_source_name(doc.metadata.get("source", "N/A"))
                page = doc.metadata.get("page", "N/A")
                dedup_key = (file_name, page, doc.page_content[:80])
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                sources.append({
                    "file": file_name,
                    "page": page,
                    "total_pages": doc.metadata.get("total_pages"),
                    "file_type": doc.metadata.get("file_type", "pdf"),
                    "content": doc.page_content,
                    "chunk_index": chunk_idx + 1,
                    "score": round(float(score), 3),
                })
            result["sources"] = sources

    return result

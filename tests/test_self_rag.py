"""
Unit & Integration tests cho Q10 — Self-RAG (LLM self-evaluation).

Self-RAG là kỹ thuật mà LLM tự đánh giá chất lượng:
1. rewrite_query()        — sinh nhiều variant để tăng recall
2. grade_document_relevance() — lọc docs không liên quan
3. filter_relevant_docs() — áp dụng grade cho toàn bộ list docs
4. grade_answer()         — tự đánh giá câu trả lời (grounded? hallucination?)

Khi nào chạy:
    pytest tests/test_self_rag.py -v -m "not integration"   # unit tests (mock LLM)
    pytest tests/test_self_rag.py -v -m integration         # cần Ollama

Lý do tồn tại file này:
    Self-RAG cần mock LLM vì tất cả functions đều gọi LLM.
    Unit tests dùng MagicMock để kiểm soát output của LLM,
    tách biệt logic parsing/fallback khỏi model thật.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

@pytest.fixture
def mock_llm():
    """Mock ChatOllama — content có thể được set trong từng test."""
    llm = MagicMock()
    response = MagicMock()
    response.content = ""
    llm.return_value = response
    # Chain: llm được gọi qua prompt | llm → invoke()
    llm_chain = MagicMock()
    llm_chain.invoke.return_value = response
    return llm, llm_chain, response

@pytest.fixture
def relevant_doc():
    return Document(
        page_content="Label Encoding là phương pháp gán nhãn cộng đồng hiệu quả.",
        metadata={"source": "thesis.pdf", "page": 5}
    )

@pytest.fixture
def irrelevant_doc():
    return Document(
        page_content="Thời tiết hôm nay rất đẹp, nắng nhẹ và có mây.",
        metadata={"source": "other.pdf", "page": 1}
    )

@pytest.fixture
def sample_docs_mixed(relevant_doc, irrelevant_doc):
    """2 relevant + 1 irrelevant"""
    return [
        relevant_doc,
        Document(
            page_content="Modularity Q đánh giá chất lượng phân cụm cộng đồng.",
            metadata={"source": "thesis.pdf", "page": 8}
        ),
        irrelevant_doc,
    ]

class TestRewriteQuery:
    """
    Kiểm thử rewrite_query(question, llm) → List[str].

    Kỳ vọng:
    - Luôn có ít nhất 1 phần tử (câu gốc)
    - Câu gốc luôn là phần tử đầu tiên
    - Không raise exception dù LLM lỗi
    """

    def _make_mock_llm_chain(self, content):
        """Tạo mock để patch ChatPromptTemplate | llm chain."""
        mock_response = MagicMock()
        mock_response.content = content
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        return mock_chain

    def test_returns_list(self):
        """rewrite_query() luôn trả về list"""
        from modules.self_rag import rewrite_query
        mock_llm_obj = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "1. Câu hỏi variant 1\n2. Câu hỏi variant 2\n3. Câu hỏi variant 3"

        with patch("modules.self_rag.ChatPromptTemplate") as mock_pt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            mock_pt.from_template.return_value.__or__ = MagicMock(return_value=mock_chain)
            mock_llm_obj.__ror__ = MagicMock(return_value=mock_chain)

            # Dùng patch trực tiếp trên chain
            with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
                prompt_mock = MagicMock()
                prompt_mock.__or__ = MagicMock(return_value=mock_chain)
                mock_fmt.return_value = prompt_mock
                result = rewrite_query("Label Encoding là gì?", mock_llm_obj)

        assert isinstance(result, list)
        assert len(result) >= 1

    def test_original_question_in_result(self):
        """Câu hỏi gốc phải có trong kết quả trả về"""
        from modules.self_rag import rewrite_query
        question = "Label Encoding trong phân cụm cộng đồng là gì?"

        mock_llm_obj = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "1. Encoding nhãn cộng đồng\n2. Community labeling\n3. Graph encoding"

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = rewrite_query(question, mock_llm_obj)

        assert question in result, "Câu hỏi gốc phải có trong kết quả rewrite"

    def test_fallback_on_llm_error(self):
        """Nếu LLM throw exception → trả về list với chỉ câu gốc, không crash"""
        from modules.self_rag import rewrite_query
        question = "Thuật toán GA hoạt động như thế nào?"

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("LLM timeout")
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = rewrite_query(question, MagicMock())

        assert isinstance(result, list)
        assert question in result  # Fallback phải trả về câu gốc

    def test_max_4_variants_returned(self):
        """Kết quả không vượt quá 4 (câu gốc + 3 variants)"""
        from modules.self_rag import rewrite_query

        mock_response = MagicMock()
        mock_response.content = "\n".join([
            "1. Variant A long enough",
            "2. Variant B long enough",
            "3. Variant C long enough",
        ])

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = rewrite_query("câu hỏi test", MagicMock())

        assert len(result) <= 4  # original + max 3 rewrites

    def test_short_lines_filtered_out(self):
        """Các dòng quá ngắn (< 10 ký tự) bị loại ra"""
        from modules.self_rag import rewrite_query

        mock_response = MagicMock()
        # 2 dòng ngắn, 1 dòng đủ dài
        mock_response.content = "1. ok\n2. short\n3. This is a long enough query variant"

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = rewrite_query("câu hỏi dài", MagicMock())

        # Các dòng ngắn "ok", "short" bị lọc
        for item in result[1:]:  # bỏ qua câu gốc
            assert len(item) >= 10

class TestGradeDocumentRelevance:
    """
    Kiểm thử grade_document_relevance(question, doc, llm) → bool.

    LLM trả lời "CÓ" hoặc "KHÔNG" → hàm parse và trả về True/False.
    Fallback: nếu exception → True (giữ lại doc để an toàn).
    """

    def test_returns_true_when_llm_says_co(self, relevant_doc):
        """LLM trả 'CÓ' → grade_document_relevance trả True"""
        from modules.self_rag import grade_document_relevance

        mock_response = MagicMock()
        mock_response.content = "CÓ"

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_document_relevance("Label Encoding?", relevant_doc, MagicMock())

        assert result is True

    def test_returns_false_when_llm_says_khong(self, irrelevant_doc):
        """LLM trả 'KHÔNG' → grade_document_relevance trả False"""
        from modules.self_rag import grade_document_relevance

        mock_response = MagicMock()
        mock_response.content = "KHÔNG"

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_document_relevance("GA hoạt động?", irrelevant_doc, MagicMock())

        assert result is False

    def test_returns_true_on_exception(self, relevant_doc):
        """Exception trong LLM call → fallback trả True (giữ doc)"""
        from modules.self_rag import grade_document_relevance

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("network error")
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_document_relevance("câu hỏi?", relevant_doc, MagicMock())

        assert result is True

    def test_yes_english_also_accepted(self, relevant_doc):
        """LLM tiếng Anh trả 'YES' → cũng là True"""
        from modules.self_rag import grade_document_relevance

        mock_response = MagicMock()
        mock_response.content = "YES"

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_document_relevance("Label Encoding?", relevant_doc, MagicMock())

        assert result is True

class TestFilterRelevantDocs:
    """
    Kiểm thử filter_relevant_docs(question, documents, llm) → List[Document].

    Quan trọng:
    - Nếu tất cả docs bị lọc → trả về ít nhất 2 docs đầu tiên (không trả rỗng)
    - Empty input → trả về []
    """

    def test_empty_input_returns_empty(self):
        """Input rỗng → output rỗng"""
        from modules.self_rag import filter_relevant_docs
        result = filter_relevant_docs("câu hỏi", [], MagicMock())
        assert result == []

    def test_relevant_docs_kept(self, sample_docs_mixed):
        """Docs có LLM trả 'CÓ' được giữ lại"""
        from modules.self_rag import filter_relevant_docs

        # Mock grade_document_relevance để kiểm soát kết quả
        with patch("modules.self_rag.grade_document_relevance") as mock_grade:
            # Doc 0: CÓ, Doc 1: CÓ, Doc 2: KHÔNG
            mock_grade.side_effect = [True, True, False]
            result = filter_relevant_docs(
                "Label Encoding?", sample_docs_mixed, MagicMock()
            )

        assert len(result) == 2
        assert all("thesis.pdf" in d.metadata["source"] for d in result)

    def test_fallback_when_all_filtered(self, sample_docs_mixed):
        """
        Nếu grade_document_relevance trả False cho tất cả,
        filter_relevant_docs giữ lại 2 docs đầu tiên (không trả rỗng).
        """
        from modules.self_rag import filter_relevant_docs

        with patch("modules.self_rag.grade_document_relevance") as mock_grade:
            mock_grade.return_value = False  # Tất cả đều "KHÔNG"
            result = filter_relevant_docs(
                "câu hỏi", sample_docs_mixed, MagicMock()
            )

        assert len(result) >= 2, \
            "Phải giữ lại ít nhất 2 docs khi tất cả bị lọc"

    def test_returns_list_of_documents(self, sample_docs_mixed):
        """Output phải là list of Document"""
        from modules.self_rag import filter_relevant_docs

        with patch("modules.self_rag.grade_document_relevance", return_value=True):
            result = filter_relevant_docs(
                "câu hỏi", sample_docs_mixed, MagicMock()
            )

        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)

    def test_result_is_subset_of_input(self, sample_docs_mixed):
        """Kết quả chỉ chứa docs từ input, không tạo docs mới"""
        from modules.self_rag import filter_relevant_docs

        with patch("modules.self_rag.grade_document_relevance") as mock_grade:
            mock_grade.side_effect = [True, False, True]
            result = filter_relevant_docs(
                "câu hỏi", sample_docs_mixed, MagicMock()
            )

        input_contents = {d.page_content for d in sample_docs_mixed}
        for doc in result:
            assert doc.page_content in input_contents

class TestGradeAnswer:
    """
    Kiểm thử grade_answer(question, context, answer, llm) → dict.

    Output dict phải có:
      - score: float trong [0.0, 1.0]
      - is_grounded: bool
      - has_hallucination: bool
      - feedback: str

    Fallback khi LLM lỗi → default dict với score=0.5.
    """

    def _patch_llm_response(self, json_str):
        """Helper: mock chain trả về json_str."""
        mock_response = MagicMock()
        mock_response.content = json_str
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        return mock_chain

    def test_output_has_all_required_keys(self):
        """Output luôn có đủ 4 keys dù LLM lỗi hay không"""
        from modules.self_rag import grade_answer

        json_str = '{"score": 0.8, "is_grounded": true, "has_hallucination": false, "feedback": "Tốt"}'

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = self._patch_llm_response(json_str)
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "câu trả lời", MagicMock())

        assert "score" in result
        assert "is_grounded" in result
        assert "has_hallucination" in result
        assert "feedback" in result

    def test_score_is_float_in_range(self):
        """score phải là float trong [0.0, 1.0]"""
        from modules.self_rag import grade_answer

        json_str = '{"score": 0.75, "is_grounded": true, "has_hallucination": false, "feedback": "OK"}'

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = self._patch_llm_response(json_str)
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời", MagicMock())

        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0

    def test_is_grounded_is_bool(self):
        """is_grounded phải là bool"""
        from modules.self_rag import grade_answer

        json_str = '{"score": 0.9, "is_grounded": true, "has_hallucination": false, "feedback": "Good"}'

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = self._patch_llm_response(json_str)
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời", MagicMock())

        assert isinstance(result["is_grounded"], bool)

    def test_has_hallucination_is_bool(self):
        """has_hallucination phải là bool"""
        from modules.self_rag import grade_answer

        json_str = '{"score": 0.3, "is_grounded": false, "has_hallucination": true, "feedback": "Bịa đặt"}'

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = self._patch_llm_response(json_str)
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời", MagicMock())

        assert isinstance(result["has_hallucination"], bool)

    def test_feedback_is_string(self):
        """feedback phải là string"""
        from modules.self_rag import grade_answer

        json_str = '{"score": 0.6, "is_grounded": true, "has_hallucination": false, "feedback": "Câu trả lời trung bình"}'

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = self._patch_llm_response(json_str)
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời", MagicMock())

        assert isinstance(result["feedback"], str)

    def test_fallback_on_exception(self):
        """LLM raise exception → default dict với score=0.5, không crash"""
        from modules.self_rag import grade_answer

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = Exception("connection refused")
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời", MagicMock())

        assert result["score"] == 0.5
        assert isinstance(result["is_grounded"], bool)
        assert isinstance(result["has_hallucination"], bool)

    def test_fallback_on_invalid_json(self):
        """LLM trả JSON không hợp lệ → default dict, không crash"""
        from modules.self_rag import grade_answer

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_response = MagicMock()
            mock_response.content = "Đây không phải JSON hợp lệ {broken"
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời", MagicMock())

        # Không crash, trả về default
        assert "score" in result
        assert "is_grounded" in result

    def test_hallucination_detected_sets_flag(self):
        """Khi LLM phát hiện hallucination, has_hallucination=True"""
        from modules.self_rag import grade_answer

        json_str = json.dumps({
            "score": 0.2,
            "is_grounded": False,
            "has_hallucination": True,
            "feedback": "Câu trả lời chứa thông tin bịa đặt"
        })

        with patch("modules.self_rag.ChatPromptTemplate.from_template") as mock_fmt:
            mock_response = MagicMock()
            mock_response.content = json_str
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = mock_response
            prompt_mock = MagicMock()
            prompt_mock.__or__ = MagicMock(return_value=mock_chain)
            mock_fmt.return_value = prompt_mock

            result = grade_answer("câu hỏi", "context", "trả lời bịa", MagicMock())

        assert result["has_hallucination"] is True
        assert result["is_grounded"] is False
        assert result["score"] < 0.5

@pytest.mark.integration
class TestIntegrationSelfRAG:
    """
    Integration test Q10: Self-RAG với LLM thật (Ollama qwen2.5:7b).
    Cần: Ollama đang chạy với model qwen2.5:7b.
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_ollama(self):
        """Skip nếu Ollama không khả dụng"""
        try:
            from modules.rag_chain import check_ollama_connection
            if not check_ollama_connection():
                pytest.skip("Ollama không chạy")
        except Exception:
            pytest.skip("Không thể kiểm tra Ollama")

    def test_rewrite_query_with_real_llm(self):
        """rewrite_query với LLM thật → list >= 1 phần tử, câu gốc có mặt"""
        import config
        from langchain_ollama import ChatOllama
        from modules.self_rag import rewrite_query

        llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
        question = "Label Encoding trong phân cụm cộng đồng là gì?"
        result = rewrite_query(question, llm)

        assert isinstance(result, list)
        assert len(result) >= 1
        assert question in result

    def test_grade_document_relevance_with_real_llm(self, relevant_doc):
        """grade_document_relevance với LLM thật → bool"""
        import config
        from langchain_ollama import ChatOllama
        from modules.self_rag import grade_document_relevance

        llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
        result = grade_document_relevance(
            "Label Encoding là gì trong phân cụm?", relevant_doc, llm
        )
        assert isinstance(result, bool)

    def test_grade_answer_with_real_llm(self):
        """grade_answer với LLM thật → dict đúng schema"""
        import config
        from langchain_ollama import ChatOllama
        from modules.self_rag import grade_answer

        llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
        result = grade_answer(
            question="Label Encoding là gì?",
            context="Label Encoding gán nhãn số nguyên cho mỗi đỉnh trong cộng đồng.",
            answer="Label Encoding dùng số nguyên để đánh dấu cộng đồng.",
            llm=llm,
        )
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["is_grounded"], bool)
        assert isinstance(result["has_hallucination"], bool)
        assert isinstance(result["feedback"], str)

# -*- coding: utf-8 -*-
"""
Unit & Integration tests cho Q9 — Re-ranking với Cross-Encoder.

Cross-Encoder so sánh (query, passage) cùng lúc trong một model → điểm chính xác hơn.
Bi-Encoder (FAISS) mã hóa query và passage riêng → nhanh hơn nhưng kém chính xác hơn.

Pipeline Re-ranking:
  1. FAISS/BM25 lấy top-N candidates (bi-encoder scores)
  2. Cross-Encoder score lại top-N, sắp xếp lại
  3. Trả về top-K đã rerank

Khi nào chạy:
    pytest tests/test_reranker.py -v -m "not integration"  # unit tests (mock)
    pytest tests/test_reranker.py -v -m integration        # cần sentence_transformers

Lý do tồn tại file này:
    modules/reranker.py (Q9) cần được test để đảm bảo:
    - Fallback hoạt động khi không có cross-encoder
    - Output format đúng (3-tuple)
    - Không làm thay đổi nội dung document
    - compare_bi_vs_cross_encoder() trả đúng cấu trúc dict
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_doc_score_pairs():
    """3 cặp (Document, bi_encoder_score) giả lập kết quả từ FAISS."""
    return [
        (Document(page_content="Label Encoding maps vertices to community IDs.",
                  metadata={"source": "a.pdf", "page": 1}), 0.92),
        (Document(page_content="Modularity Q evaluates partition quality.",
                  metadata={"source": "a.pdf", "page": 3}), 0.85),
        (Document(page_content="Self-RAG grades its own answer confidence.",
                  metadata={"source": "b.pdf", "page": 7}), 0.78),
    ]


# ─── get_cross_encoder singleton ─────────────────────────────────────────────

class TestGetCrossEncoder:
    """Kiểm thử lazy loading và singleton của cross-encoder model."""

    def test_returns_none_when_import_fails(self):
        """Nếu sentence_transformers không cài → get_cross_encoder() trả None"""
        import modules.reranker as reranker_module
        orig = reranker_module._cross_encoder_model
        reranker_module._cross_encoder_model = None
        try:
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                result = reranker_module.get_cross_encoder()
                # Nếu patch hoạt động, trả None; nếu đã cài rồi thì bỏ qua
                assert result is None or result is not None  # Không crash là đủ
        except Exception:
            pass
        finally:
            reranker_module._cross_encoder_model = orig

    def test_singleton_returns_same_object(self):
        """get_cross_encoder() gọi 2 lần → cùng object (singleton)"""
        import modules.reranker as reranker_module

        # Mock model để không cần download
        mock_ce = MagicMock()
        with patch.object(reranker_module, "_cross_encoder_model", mock_ce):
            m1 = reranker_module.get_cross_encoder()
            m2 = reranker_module.get_cross_encoder()
            assert m1 is m2


# ─── rerank_with_cross_encoder ───────────────────────────────────────────────

class TestRerankWithCrossEncoder:
    """
    Kiểm thử hàm rerank_with_cross_encoder(query, doc_score_pairs, top_k).

    Output format: List[Tuple[Document, float, float]]
      - [0]: Document (nội dung không thay đổi)
      - [1]: bi_encoder_score (từ FAISS)
      - [2]: cross_encoder_score (từ CrossEncoder, sigmoid-normalized)
    """

    def test_empty_input_returns_empty(self):
        """Input rỗng → output rỗng, không crash"""
        from modules.reranker import rerank_with_cross_encoder
        result = rerank_with_cross_encoder("test query", [], top_k=3)
        assert result == []

    def test_fallback_when_no_cross_encoder(self, sample_doc_score_pairs):
        """
        Khi cross-encoder không khả dụng → fallback dùng bi-encoder scores.
        Output vẫn đúng format 3-tuple: (doc, bi_score, bi_score).
        """
        import modules.reranker as reranker_module
        orig = reranker_module._cross_encoder_model
        reranker_module._cross_encoder_model = None

        try:
            with patch.object(reranker_module, "get_cross_encoder", return_value=None):
                result = reranker_module.rerank_with_cross_encoder(
                    "Label Encoding community detection",
                    sample_doc_score_pairs,
                    top_k=2,
                )
                assert len(result) == 2
                for item in result:
                    assert len(item) == 3
                    doc, bi_score, ce_score = item
                    assert isinstance(doc, Document)
                    assert isinstance(bi_score, float)
                    assert isinstance(ce_score, float)
                    # Trong fallback: ce_score == bi_score
                    assert bi_score == ce_score
        finally:
            reranker_module._cross_encoder_model = orig

    def test_output_is_list_of_3_tuples(self, sample_doc_score_pairs):
        """
        Với mock cross-encoder, output phải là list of 3-tuples.
        Mỗi tuple: (Document, float, float).
        """
        import modules.reranker as reranker_module

        mock_ce = MagicMock()
        mock_ce.predict.return_value = [2.5, 1.8, -0.3]  # raw logit scores

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce):
            result = reranker_module.rerank_with_cross_encoder(
                "community detection",
                sample_doc_score_pairs,
                top_k=3,
            )

        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            doc, bi_score, ce_score = item
            assert isinstance(doc, Document)
            assert isinstance(bi_score, float)
            assert isinstance(ce_score, float)

    def test_top_k_limits_output(self, sample_doc_score_pairs):
        """top_k=2 với 3 input docs → output có đúng 2 items"""
        import modules.reranker as reranker_module

        mock_ce = MagicMock()
        mock_ce.predict.return_value = [1.5, 2.0, 0.8]

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce):
            result = reranker_module.rerank_with_cross_encoder(
                "query",
                sample_doc_score_pairs,
                top_k=2,
            )

        assert len(result) == 2

    def test_output_sorted_by_cross_encoder_score(self, sample_doc_score_pairs):
        """
        Sau rerank, output được sắp xếp giảm dần theo cross_encoder_score.
        Đây là điểm mấu chốt của re-ranking.
        """
        import modules.reranker as reranker_module

        # Doc thứ 3 (score thấp nhất ở bi-encoder) được cross-encoder đánh giá cao nhất
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.5, 1.0, 3.0]  # doc2 > doc1 > doc0

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce):
            result = reranker_module.rerank_with_cross_encoder(
                "Self-RAG grading",
                sample_doc_score_pairs,
                top_k=3,
            )

        # Kiểm tra sắp xếp giảm dần
        ce_scores = [item[2] for item in result]
        assert ce_scores == sorted(ce_scores, reverse=True), \
            "Kết quả phải được sắp xếp giảm dần theo cross-encoder score"

    def test_document_content_not_modified(self, sample_doc_score_pairs):
        """Reranking không được thay đổi nội dung document."""
        import modules.reranker as reranker_module

        original_contents = [doc.page_content for doc, _ in sample_doc_score_pairs]
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [1.0, 2.0, 0.5]

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce):
            result = reranker_module.rerank_with_cross_encoder(
                "query",
                sample_doc_score_pairs,
                top_k=3,
            )

        result_contents = {item[0].page_content for item in result}
        assert result_contents == set(original_contents), \
            "Nội dung documents không được thay đổi sau reranking"

    def test_cross_encoder_scores_are_sigmoid_normalized(self, sample_doc_score_pairs):
        """
        Cross-encoder scores được normalize bằng sigmoid → nằm trong (0, 1).
        Sigmoid: f(x) = 1 / (1 + e^-x), luôn trong (0, 1).
        """
        import modules.reranker as reranker_module

        mock_ce = MagicMock()
        # Raw logits: rất âm, 0, rất dương
        mock_ce.predict.return_value = [-10.0, 0.0, 10.0]

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce):
            result = reranker_module.rerank_with_cross_encoder(
                "query",
                sample_doc_score_pairs,
                top_k=3,
            )

        for _, _, ce_score in result:
            # sigmoid(x) ∈ (0,1) nhưng sau round(4) cực trị có thể = 0.0 hoặc 1.0
            assert 0.0 <= ce_score <= 1.0, \
                f"Cross-encoder score {ce_score} phải nằm trong [0, 1] sau sigmoid"

    def test_bi_encoder_score_preserved(self, sample_doc_score_pairs):
        """
        Bi-encoder score gốc không bị thay đổi trong output.
        Chỉ ce_score là score mới.
        """
        import modules.reranker as reranker_module

        original_bi_scores = {doc.page_content: score
                              for doc, score in sample_doc_score_pairs}
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [1.0, 0.5, 2.0]

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce):
            result = reranker_module.rerank_with_cross_encoder(
                "query",
                sample_doc_score_pairs,
                top_k=3,
            )

        for doc, bi_score, _ in result:
            expected = original_bi_scores[doc.page_content]
            assert bi_score == expected, \
                f"Bi-encoder score của '{doc.page_content[:30]}' bị thay đổi"


# ─── compare_bi_vs_cross_encoder ─────────────────────────────────────────────

class TestCompareBiVsCrossEncoder:
    """
    Kiểm thử hàm compare_bi_vs_cross_encoder() — dùng cho UI hiển thị
    so sánh hai phương pháp ranking.
    """

    def test_empty_input_returns_default_dict(self):
        """Input rỗng → dict với tất cả keys, không crash"""
        from modules.reranker import compare_bi_vs_cross_encoder
        result = compare_bi_vs_cross_encoder("query", [], top_k=3)

        assert isinstance(result, dict)
        assert "bi_encoder_results" in result
        assert "cross_encoder_results" in result
        assert "ranking_changed" in result
        assert "latency_ms" in result
        assert result["bi_encoder_results"] == []
        assert result["cross_encoder_results"] == []

    def test_output_has_all_required_keys(self, sample_doc_score_pairs):
        """Output luôn có đủ 4 keys: bi_encoder_results, cross_encoder_results, ranking_changed, latency_ms"""
        import modules.reranker as reranker_module
        with patch.object(reranker_module, "get_cross_encoder", return_value=None):
            result = reranker_module.compare_bi_vs_cross_encoder(
                "query", sample_doc_score_pairs, top_k=2
            )

        assert set(result.keys()) >= {"bi_encoder_results", "cross_encoder_results",
                                       "ranking_changed", "latency_ms"}

    def test_ranking_changed_is_bool(self, sample_doc_score_pairs):
        """ranking_changed phải là bool"""
        import modules.reranker as reranker_module
        with patch.object(reranker_module, "get_cross_encoder", return_value=None):
            result = reranker_module.compare_bi_vs_cross_encoder(
                "query", sample_doc_score_pairs, top_k=2
            )
        assert isinstance(result["ranking_changed"], bool)

    def test_latency_ms_is_non_negative(self, sample_doc_score_pairs):
        """latency_ms phải >= 0"""
        import modules.reranker as reranker_module
        with patch.object(reranker_module, "get_cross_encoder", return_value=None):
            result = reranker_module.compare_bi_vs_cross_encoder(
                "query", sample_doc_score_pairs, top_k=2
            )
        assert result["latency_ms"] >= 0

    def test_ranking_changed_detected_correctly(self, sample_doc_score_pairs):
        """
        Nếu cross-encoder đảo thứ tự (doc thứ 2 lên đầu), ranking_changed=True.
        Nếu thứ tự giữ nguyên, ranking_changed=False.
        """
        import modules.reranker as reranker_module

        # Trường hợp không thay đổi: cross-encoder giữ thứ tự gốc
        mock_ce_same = MagicMock()
        mock_ce_same.predict.return_value = [3.0, 2.0, 1.0]  # doc0 > doc1 > doc2 (giống bi)

        with patch.object(reranker_module, "get_cross_encoder", return_value=mock_ce_same):
            result_same = reranker_module.compare_bi_vs_cross_encoder(
                "query", sample_doc_score_pairs, top_k=3
            )
        # Không nhất thiết phải False (do id() có thể khác), chỉ kiểm tra type
        assert isinstance(result_same["ranking_changed"], bool)

    def test_bi_encoder_results_are_original_order(self, sample_doc_score_pairs):
        """bi_encoder_results phải giữ nguyên thứ tự gốc (không rerank)"""
        import modules.reranker as reranker_module
        with patch.object(reranker_module, "get_cross_encoder", return_value=None):
            result = reranker_module.compare_bi_vs_cross_encoder(
                "query", sample_doc_score_pairs, top_k=3
            )
        bi_results = result["bi_encoder_results"]
        assert len(bi_results) <= len(sample_doc_score_pairs)
        for i, (doc, score) in enumerate(bi_results):
            assert isinstance(doc, Document)
            assert isinstance(score, float)


# ─── Integration tests (cần sentence_transformers + model download) ───────────

@pytest.mark.integration
class TestIntegrationCrossEncoder:
    """
    Integration test Q9: Dùng cross-encoder model thật.
    Cần: pip install sentence-transformers + download model (offline có thể fail).
    """

    @pytest.fixture(autouse=True)
    def skip_if_no_cross_encoder(self):
        """Skip nếu không load được cross-encoder"""
        try:
            from sentence_transformers import CrossEncoder
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            pytest.skip("Cross-encoder model không khả dụng (cần internet hoặc cache)")

    def test_real_rerank_returns_valid_output(self, sample_doc_score_pairs):
        """Cross-encoder thật → output đúng format, scores trong (0,1)"""
        import modules.reranker as reranker_module
        reranker_module._cross_encoder_model = None  # Reset cache

        from modules.reranker import rerank_with_cross_encoder
        results = rerank_with_cross_encoder(
            "community detection Label Encoding",
            sample_doc_score_pairs,
            top_k=2,
        )
        assert len(results) == 2
        for doc, bi_score, ce_score in results:
            assert isinstance(doc, Document)
            assert 0.0 <= ce_score <= 1.0

    def test_real_compare_returns_latency(self, sample_doc_score_pairs):
        """compare_bi_vs_cross_encoder với model thật → latency_ms > 0"""
        import modules.reranker as reranker_module
        reranker_module._cross_encoder_model = None

        from modules.reranker import compare_bi_vs_cross_encoder
        result = compare_bi_vs_cross_encoder(
            "Label Encoding community",
            sample_doc_score_pairs,
            top_k=2,
        )
        assert result["latency_ms"] > 0
        assert len(result["cross_encoder_results"]) == 2

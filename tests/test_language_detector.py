# -*- coding: utf-8 -*-
"""
Unit tests cho modules/language_detector.py

Chạy:
    pytest tests/test_language_detector.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.language_detector import detect_language, get_language_instruction, get_prompt_template


# ─── detect_language ─────────────────────────────────────────────────────────

class TestDetectLanguage:
    """Kiểm thử hàm detect_language()"""

    def test_vietnamese_with_accents(self):
        """Văn bản tiếng Việt có dấu → nhận diện là 'vi'"""
        text = "Xin chào, tôi muốn hỏi về tài liệu này."
        assert detect_language(text) == "vi"

    def test_vietnamese_common_words(self):
        """Văn bản tiếng Việt không dấu nhưng có từ phổ biến → nhận diện là 'vi'"""
        text = "cua va la co trong cho duoc"
        assert detect_language(text) == "vi"

    def test_english_text(self):
        """Văn bản tiếng Anh thuần → nhận diện là 'en'"""
        text = "What is the purpose of retrieval augmented generation?"
        assert detect_language(text) == "en"

    def test_empty_string(self):
        """Chuỗi rỗng → mặc định 'en'"""
        assert detect_language("") == "en"

    def test_whitespace_only(self):
        """Chuỗi chỉ có khoảng trắng → mặc định 'en'"""
        assert detect_language("   ") == "en"

    def test_none_input(self):
        """None input → mặc định 'en' (không crash)"""
        assert detect_language(None) == "en"

    def test_mixed_text_with_vietnamese(self):
        """Văn bản hỗn hợp có ký tự tiếng Việt → nhận diện là 'vi'"""
        text = "SmartDocAI là một hệ thống AI document Q&A"
        assert detect_language(text) == "vi"

    def test_single_vietnamese_char(self):
        """Chỉ 1 ký tự đặc trưng tiếng Việt → nhận diện là 'vi'"""
        assert detect_language("à") == "vi"

    def test_technical_english(self):
        """Văn bản kỹ thuật tiếng Anh → nhận diện là 'en'"""
        text = "FAISS uses IndexFlatL2 for brute-force similarity search."
        assert detect_language(text) == "en"

    def test_return_type(self):
        """Kết quả trả về phải là string"""
        result = detect_language("Hello world")
        assert isinstance(result, str)

    def test_return_values_valid(self):
        """Kết quả trả về chỉ có thể là 'vi' hoặc 'en'"""
        for text in ["Hello", "Xin chào", "", "123"]:
            result = detect_language(text)
            assert result in ("vi", "en"), f"Unexpected language '{result}' for '{text}'"


# ─── get_language_instruction ────────────────────────────────────────────────

class TestGetLanguageInstruction:
    """Kiểm thử hàm get_language_instruction()"""

    def test_vietnamese_instruction_returned(self):
        """Trả về hướng dẫn tiếng Việt khi language='vi'"""
        result = get_language_instruction("vi")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_english_instruction_returned(self):
        """Trả về hướng dẫn tiếng Anh khi language='en'"""
        result = get_language_instruction("en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_vi_and_en_differ(self):
        """Hướng dẫn VI và EN phải khác nhau"""
        vi = get_language_instruction("vi")
        en = get_language_instruction("en")
        assert vi != en

    def test_unknown_language_returns_string(self):
        """Language không rõ → vẫn trả về string, không crash"""
        result = get_language_instruction("fr")
        assert isinstance(result, str)


# ─── get_prompt_template ─────────────────────────────────────────────────────

class TestGetPromptTemplate:
    """Kiểm thử hàm get_prompt_template()"""

    def test_prompt_contains_context(self):
        """Prompt phải chứa context đã truyền vào"""
        prompt = get_prompt_template("en", "test context", "test question")
        assert "test context" in prompt

    def test_prompt_contains_question(self):
        """Prompt phải chứa câu hỏi đã truyền vào"""
        prompt = get_prompt_template("en", "ctx", "my question")
        assert "my question" in prompt

    def test_vietnamese_prompt(self):
        """Prompt tiếng Việt phải là string không rỗng"""
        prompt = get_prompt_template("vi", "ngữ cảnh", "câu hỏi")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_english_prompt(self):
        """Prompt tiếng Anh phải là string không rỗng"""
        prompt = get_prompt_template("en", "context", "question")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_vi_en_prompts_differ(self):
        """Prompt VI và EN phải khác nhau"""
        vi = get_prompt_template("vi", "ctx", "q")
        en = get_prompt_template("en", "ctx", "q")
        assert vi != en

"""ユニットテスト: src.core.extractor

  テスト対象:
    - FanCountExtractor
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.extractor import FanCountExtractor
from src.domain.logic import get_fan_count


class TestFanCountExtractor:
    """FanCountExtractor のテスト"""

    def test_extract_basic(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\nメンバーB\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"メンバーA": ["メンハーA"]}', encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        assert extractor.member_list == ["メンバーA", "メンバーB"]
        assert extractor.member_replace == {"メンバーA": ["メンハーA"]}

    def test_extract_with_texts(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"メンバーA": []}', encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = ["メンバーA 1,000,000 人"]
        fan_counts = extractor.extract(texts)
        assert "メンバーA" in fan_counts
        assert fan_counts["メンバーA"] == 1000000

    def test_extract_no_text_returns_zero(self, tmp_path):
        """テキストがない場合、ファン数は 0 になる"""
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        fan_counts = extractor.extract([])
        assert fan_counts == {"メンバーA": 0}


class TestMemberReplace:
    """FanCountExtractor 内部の置換ロジック"""

    def test_basic_replacement(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"メンバーA": ["メンハーA", "メンパーA"]}', encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = ["メンハーA 1,000,000 人"]
        fan_counts = extractor.extract(texts)
        assert "メンバーA" in fan_counts

    def test_unmatched_member_ignored(self, tmp_path):
        """replace に存在しない member はスキップされる"""
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"メンバーB": ["誤認識B"]}', encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = ["メンバーA 1,000,000 人"]
        fan_counts = extractor.extract(texts)
        assert fan_counts.get("メンバーA") == 1000000

    def test_no_false_replacement(self, tmp_path):
        """OCR誤認識パターンがテキストにない場合、何も変わらない"""
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"メンバーA": ["存在しないパターン"]}', encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = ["メンバーA 1,000,000 人"]
        fan_counts = extractor.extract(texts)
        assert fan_counts.get("メンバーA") == 1000000


class TestFanCountConsistency:
    """FanCountExtractor.extract() が get_fan_count と同等の結果を返す"""

    def test_consistency_with_existing_get_fan_count(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("万丈目準\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"万丈目準": []}', encoding="utf-8")

        texts = ["万丈目準 3,249,444,186 人"]

        # 直接呼び出し
        existing_result = get_fan_count(texts, "万丈目準", [])
        assert existing_result == 3249444186

        # FanCountExtractor 経由
        extractor = FanCountExtractor(list_file, replace_file)
        fan_counts = extractor.extract(texts)
        assert fan_counts.get("万丈目準") == 3249444186


class TestVLMJsonInput:
    """VLM JSON形式の入力からのファン数抽出"""

    def test_vlm_json_basic(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\nメンバーB\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [1000000], "メンバーB": [2000000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1000000
        assert fan_counts["メンバーB"] == 2000000

    def test_vlm_json_integer_value(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": 500000}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 500000

    def test_vlm_json_string_value(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": "1,500,000"}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1500000

    def test_vlm_json_multiple_frames_mode(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [1000000]}',
            '{"メンバーA": [1000000]}',
            '{"メンバーA": [2000000]}',
            '{"メンバーA": [1000000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1000000

    def test_vlm_json_member_replacement(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("モルガン\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text('{"モルガン": ["モルン"]}', encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"モルン": [300000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["モルガン"] == 300000

    def test_vlm_json_partial_members(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\nメンバーB\nメンバーC\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [1000000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1000000
        assert fan_counts["メンバーB"] == 0
        assert fan_counts["メンバーC"] == 0

    def test_vlm_json_unknown_member_ignored(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [1000000], "存在しない": [999999]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1000000

    def test_vlm_json_list_value(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [500000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 500000

    def test_vlm_json_list_multiple_values(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [100000, 200000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 300000

    def test_vlm_json_invalid_json_skipped(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            'not json at all',
            '{"メンバーA": [1000000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1000000

    def test_vlm_json_detection(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        assert extractor._is_vlm_json_input(['{"メンバーA": [1000]}']) is True
        assert extractor._is_vlm_json_input(['メンバーA 1,000']) is False
        assert extractor._is_vlm_json_input([]) is False
        assert extractor._is_vlm_json_input(['', '  ', '{"メンバーA": [1000]}']) is True

    def test_vlm_json_multiple_members_across_frames(self, tmp_path):
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\nメンバーB\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{}", encoding="utf-8")

        extractor = FanCountExtractor(list_file, replace_file)
        texts = [
            '{"メンバーA": [1000000]}',
            '{"メンバーB": [2000000]}',
            '{"メンバーA": [1000000], "メンバーB": [2000000]}',
        ]
        fan_counts = extractor.extract(texts)
        assert fan_counts["メンバーA"] == 1000000
        assert fan_counts["メンバーB"] == 2000000

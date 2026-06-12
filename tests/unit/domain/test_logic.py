"""ユニットテスト: src.domain.logic

  テスト対象:
    - get_fan_count()   — 正規表現マッチング・最頻値選択・重複排除
    - post_process_ocr_text() — OCR後処理（記号除去・カンマ正規化）
"""

from __future__ import annotations

from src.domain.logic import get_fan_count, post_process_ocr_text


# ===================================================================
# 1. get_fan_count()
# ===================================================================


class TestGetFanCount:
    """get_fan_count() の正規表現マッチング・最頻値選択・重複排除"""

    def test_basic_match(self):
        """カンマ区切り数値が1つだけ含まれるテキスト → その数値を返す"""
        texts = ["万丈目準 3,249,444,186 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result == 3249444186

    def test_no_match_returns_none(self):
        """カンマ区切り数値がない場合 → None"""
        texts = ["万丈目準 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_member_not_in_text(self):
        """メンバー名がテキストに含まれない場合 → None"""
        texts = ["別人A 1,234,567"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_prefers_most_frequent(self):
        """複数フレームで同じ値が最多出現 → その値が選ばれる"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 1,000,000 人",
            "万丈目準 2,000,000 人",
            "万丈目準 2,000,000 人",
            "万丈目準 2,000,000 人",
        ]
        result = get_fan_count(texts, "万丈目準", [])
        assert result == 2000000  # 3回出現 → 最頻

    def test_tie_returns_one_of(self):
        """同率の場合はソート順で先頭（降順ソートの最初）"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 9,999,999 人",
        ]
        result = get_fan_count(texts, "万丈目準", [])
        assert result in (1000000, 9999999)

    def test_skip_already_in_fans(self):
        """fans リストに含まれる値はスキップする"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 2,000,000 人",
        ]
        result = get_fan_count(texts, "万丈目準", [1000000])
        assert result == 2000000

    def test_all_values_skipped_returns_none(self):
        """fans リストにより全候補がスキップされた場合 → None"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 2,000,000 人",
        ]
        result = get_fan_count(texts, "万丈目準", [1000000, 2000000])
        assert result is None

    def test_rejects_non_comma_numbers(self):
        """カンマなしの数値（1234567）はマッチしない"""
        texts = ["万丈目準 1234567 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_rejects_fewer_than_4_digits(self):
        """4桁未満（1,234 など）も一応マッチする"""
        texts = ["万丈目準 1,234 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result == 1234

    def test_token_boundary_prevents_concatenation(self):
        """前後に英数字・日本語文字が連結している場合はマッチしない"""
        texts = ["メンバーA123,456"]
        result = get_fan_count(texts, "メンバーA", [])
        assert result is None

    def test_no_false_positive_on_partial_number(self):
        """途中にカンマがない数値（1234,567 のような不完全）はマッチしない"""
        texts = ["万丈目準 1234,567 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_multiple_members_in_one_text(self):
        """1テキストに複数メンバーが含まれている場合、pattern.search() は
        テキスト内の最初の数値（1,111,111）を返すのが現在の実装。"""
        texts = [
            "メンバーA 1,111,111 人 メンバーB 2,222,222 人",
        ]
        result_a = get_fan_count(texts, "メンバーA", [])
        result_b = get_fan_count(texts, "メンバーB", [])
        assert result_a == 1111111
        assert result_b == 1111111  # 最初の数値が返る（制約）

    def test_large_number(self):
        """大きな桁数のファン数（ウマ娘では数百億までありうる）"""
        texts = ["万丈目準 110,100,100,010,000 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result == 110100100010000

    def test_zero(self):
        """ファン数 0 はカンマがないのでマッチしない（0は非検出扱い）"""
        texts = ["万丈目準 0 人"]
        result = get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_empty_texts_list(self):
        """空の texts → None"""
        result = get_fan_count([], "万丈目準", [])
        assert result is None


# ===================================================================
# 2. post_process_ocr_text()
# ===================================================================


class TestPostProcessOcrText:
    """post_process_ocr_text() の記号除去・カンマ正規化"""

    def test_removes_spaces(self):
        assert post_process_ocr_text("a b c") == "abc"

    def test_removes_circled_one(self):
        assert post_process_ocr_text("名前①") == "名前"

    def test_removes_down_arrow(self):
        assert post_process_ocr_text("↓123") == "123"

    def test_removes_parentheses(self):
        assert post_process_ocr_text("(test)（テスト）") == "testテスト"

    def test_removes_at_sign(self):
        assert post_process_ocr_text("user@name") == "username"

    def test_normalizes_commas(self):
        assert post_process_ocr_text("a、b，c") == "a,b,c"

    def test_removes_30_30(self):
        assert post_process_ocr_text("30/30") == ""

    def test_inserts_space_around_人(self):
        assert post_process_ocr_text("1234人") == "1234 人"

    def test_inserts_space_after_ファン数(self):
        assert post_process_ocr_text("ファン数1234") == "ファン数 1234"

    def test_full_pipeline_realistic(self):
        raw = " メンバーA 3,249,444,186① 人 ↓ (30/30) @test "
        result = post_process_ocr_text(raw)
        assert "①" not in result
        assert "↓" not in result
        assert "(" not in result and ")" not in result
        assert "@" not in result
        assert "30/30" not in result
        assert " 人" in result
        assert "メンバーA" in result
        assert "3,249,444,186" in result

    def test_comma_normalization(self):
        raw = "a、b，c"
        result = post_process_ocr_text(raw)
        assert "," in result
        assert "、" not in result
        assert "，" not in result

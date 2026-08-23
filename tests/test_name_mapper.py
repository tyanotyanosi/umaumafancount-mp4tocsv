"""NameMapper / NameMapperLoader / MappedName のテスト。

設計書 ``docs/name_mapping_design.md`` の §10 新規テストに従う。
"""

import json
from pathlib import Path

import pytest

from src.parser.name_mapper import (
    NameMapper,
    NameMapperLoader,
    levenshtein_distance,
)
from src.parser.result_parser import ResultParser


@pytest.fixture
def sample_mapping():
    """設計書 §4.2 に準拠したサンプルマッピング定義。"""
    return {
        "user_names": {
            "とまる": {"aliases": ["とまる", "マル", "ٓとまる", "tōmaru", "TM"]},
            "たけし": {"aliases": ["たけし", "武", "タケ", "竹", "Takeshi"]},
        },
        "raw_to_user": {
            "固定の検知A": "とまる",
        },
    }


def _write_mapping(tmp_path, obj, name="name_mapping.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# levenshtein_distance
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert levenshtein_distance("とまる", "とまる") == 0

    def test_single_insertion(self):
        assert levenshtein_distance("とまる", "とまるる") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("マル", "ムル") == 1

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("あ", "") == 1
        assert levenshtein_distance("", "あ") == 1


# ---------------------------------------------------------------------------
# 完全一致
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_user_names(self, sample_mapping):
        mapper = NameMapper(sample_mapping)
        result = mapper.map("マル")
        assert result.matched is True
        assert result.user_name == "とまる"
        assert result.match_type == "exact"
        assert result.alias_hit == "マル"
        assert result.warning is None

    def test_exact_user_names_direct(self, sample_mapping):
        mapper = NameMapper(sample_mapping)
        result = mapper.map("たけし")
        assert result.matched is True
        assert result.user_name == "たけし"
        assert result.match_type == "exact"

    def test_exact_raw_to_user(self, sample_mapping):
        mapper = NameMapper(sample_mapping)
        result = mapper.map("固定の検知A")
        assert result.matched is True
        assert result.user_name == "とまる"
        assert result.match_type == "exact"

    def test_exact_takes_priority_over_approx(self, sample_mapping):
        # 「マル」は完全一致（exact）であり、近似（approx）ではない。
        mapper = NameMapper(sample_mapping)
        result = mapper.map("マル")
        assert result.match_type == "exact"


# ---------------------------------------------------------------------------
# 近似一致
# ---------------------------------------------------------------------------

class TestApproxMatch:
    def test_approx_match(self, sample_mapping):
        # 「ムル」は「マル」から1文字誤認識（編集距離1）。
        mapper = NameMapper(sample_mapping, edit_distance_threshold=2)
        result = mapper.map("ムル")
        assert result.matched is True
        assert result.user_name == "とまる"
        assert result.match_type == "approx"
        assert result.alias_hit == "マル"
        assert result.warning is not None
        assert "近似一致" in result.warning

    def test_approx_out_of_threshold(self, sample_mapping):
        # 「ムルルル」は「マル」から編集距離3（置換1＋挿入2）。閾値2では一致しない。
        mapper = NameMapper(sample_mapping, edit_distance_threshold=2)
        result = mapper.map("ムルルル")
        assert result.matched is False
        assert result.user_name is None

    def test_approx_threshold_zero_rejects(self, sample_mapping):
        # 閾値0の場合は近似一致せず、完全一致のみ。
        mapper = NameMapper(sample_mapping, edit_distance_threshold=0)
        result = mapper.map("ムル")
        assert result.matched is False

    def test_approx_multiple_candidates_prefers_alpha(self):
        # 「タ」が「あお」（alias「タケ」）と「いう」（alias「タコ」）に
        # 同距離（1）で近似。アルファベット順（Unicodeコードポイント順）で
        # 小さい方「あお」（あ=U+3042 < い=U+3044）が優先される。
        mapping = {
            "user_names": {
                "あお": {"aliases": ["タケ"]},
                "いう": {"aliases": ["タコ"]},
            }
        }
        mapper = NameMapper(mapping, edit_distance_threshold=1)
        result = mapper.map("タ")
        assert result.matched is True
        assert result.match_type == "approx"
        assert result.user_name == "あお"

    def test_warn_on_approx_disabled(self, sample_mapping):
        mapper = NameMapper(sample_mapping, warn_on_approx=False)
        result = mapper.map("ムル")
        assert result.matched is True
        assert result.warning is None


# ---------------------------------------------------------------------------
# 未マッピング時の扱い（unmapped_action）
# ---------------------------------------------------------------------------

class TestUnmappedAction:
    def _parser_and_mapper(self, sample_mapping, action):
        mapper = NameMapper(sample_mapping, unmapped_action=action)
        parser = ResultParser()
        return parser, mapper

    def test_suggest_excludes_from_count_but_records(self, sample_mapping):
        parser, mapper = self._parser_and_mapper(sample_mapping, "suggest")
        frame_results = [
            {"cards": [{"name": "未知の人物", "fans": "1,000人"}]}
        ]
        result = parser.parse_batch(frame_results, mapper)
        assert "未知の人物" not in result  # 集計には含めない
        assert mapper.unmapped_names == ["未知の人物"]  # 一覧には記録

    def test_keep_keeps_in_count(self, sample_mapping):
        parser, mapper = self._parser_and_mapper(sample_mapping, "keep")
        frame_results = [
            {"cards": [{"name": "未知の人物", "fans": "1,000人"}]}
        ]
        result = parser.parse_batch(frame_results, mapper)
        assert result["未知の人物"] == 1000  # 集計に含める
        assert mapper.unmapped_names == []  # keepは一覧に記録しない

    def test_drop_removes_from_both(self, sample_mapping):
        parser, mapper = self._parser_and_mapper(sample_mapping, "drop")
        frame_results = [
            {"cards": [{"name": "未知の人物", "fans": "1,000人"}]}
        ]
        result = parser.parse_batch(frame_results, mapper)
        assert "未知の人物" not in result  # 集計から消去
        assert mapper.unmapped_names == []  # 一覧からも消去

    def test_mixed_mapped_and_unmapped(self, sample_mapping):
        # マッピングされる検知とされない検知が混在する場合。
        mapper = NameMapper(sample_mapping, unmapped_action="suggest")
        parser = ResultParser()
        frame_results = [
            {"cards": [
                {"name": "マル", "fans": "1,000人"},   # → とまる
                {"name": "未知の人物", "fans": "2,000人"},  # 未マッピング
            ]}
        ]
        result = parser.parse_batch(frame_results, mapper)
        assert result == {"とまる": 1000}
        assert mapper.unmapped_names == ["未知の人物"]


# ---------------------------------------------------------------------------
# マッピング無効時（mapper=None）
# ---------------------------------------------------------------------------

class TestMappingDisabled:
    def test_none_mapper_uses_raw_name(self):
        # マッピング無効時（mapper=None）は検知のまま集計（現状維持）。
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "マル", "fans": "1,000人"}]}
        ]
        result = parser.parse_batch(frame_results)
        assert result == {"マル": 1000}


# ---------------------------------------------------------------------------
# NameMapperLoader
# ---------------------------------------------------------------------------

class TestNameMapperLoader:
    def test_load_mapping_from_json(self, tmp_path, sample_mapping):
        path = _write_mapping(tmp_path, sample_mapping)
        data = NameMapperLoader.load_mapping(path)
        assert data == sample_mapping

    def test_file_not_found_returns_empty(self, tmp_path):
        data = NameMapperLoader.load_mapping(str(tmp_path / "missing.json"))
        assert data == {}

    def test_json_syntax_error_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ invalid json here", encoding="utf-8")
        with pytest.raises(ValueError):
            NameMapperLoader.load_mapping(str(p))

    def test_non_object_raises(self, tmp_path):
        p = tmp_path / "not_object.json"
        p.write_text(json.dumps(["a", "b"], ensure_ascii=False), encoding="utf-8")
        with pytest.raises(ValueError):
            NameMapperLoader.load_mapping(str(p))

    def test_load_default_config_file(self):
        # 設計書 §12 ステップ1: 実ファイル config/name_mapping.json が読めるか。
        config_path = Path(__file__).parent.parent / "config" / "name_mapping.json"
        data = NameMapperLoader.load_mapping(str(config_path))
        assert "とまる" in data["user_names"]
        assert "マル" in data["user_names"]["とまる"]["aliases"]
        # 設計書 §4.2 の例キーは「固定の検知名A」（名 U+540D 付き）。
        assert data["raw_to_user"]["固定の検知名A"] == "とまる"


# ---------------------------------------------------------------------------
# parse_batch 統合
# ---------------------------------------------------------------------------

class TestParseBatchIntegration:
    def test_parse_batch_with_mapper(self, sample_mapping):
        mapper = NameMapper(sample_mapping)
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "マル", "fans": "3,000人"}]},   # → とまる
            {"cards": [{"name": "とまる", "fans": "3,000人"}]},  # → とまる
        ]
        result = parser.parse_batch(frame_results, mapper)
        assert result == {"とまる": 3000}

    def test_parse_batch_with_mapper_approx(self, sample_mapping):
        # 誤検知（近似一致）も正しく補正される。
        mapper = NameMapper(sample_mapping, edit_distance_threshold=2)
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "ムル", "fans": "3,000人"}]},  # → とまる（近似）
        ]
        result = parser.parse_batch(frame_results, mapper)
        assert result == {"とまる": 3000}

    def test_parse_batch_without_mapper_unchanged(self):
        # 設計書 §10 回帰: マッピング無効時は現状と同一の結果。
        parser = ResultParser()
        frame_results = [
            {"cards": [
                {"role": "leader", "name_raw": "ろん", "fans_raw": "3,249,444,186人"},
                {"role": "member", "name_raw": "キュルス", "fans_raw": "2,823,905,018人"},
            ]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged == {"ろん": 3249444186, "キュルス": 2823905018}

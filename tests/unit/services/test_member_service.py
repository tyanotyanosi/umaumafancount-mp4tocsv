"""ユニットテスト: src.services.member_service

  テスト対象:
    - MemberService
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.models import MemberEntry
from src.services.member_service import MemberService


class TestMemberService:
    """MemberService のテスト"""

    def test_load_empty_files(self, tmp_path):
        """ファイルが存在しない場合は空辞書を返す"""
        ms = MemberService()
        list_file = tmp_path / "memberList.txt"
        replace_file = tmp_path / "memberReplace.json"
        result = ms.load(list_file, replace_file)
        assert result == {}

    def test_load_with_list_only(self, tmp_path):
        """memberList.txt だけからメンバーを読み込む"""
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("万丈目準\nオグリローマン\n", encoding="utf-8")
        replace_file = tmp_path / "nonexistent.json"

        ms = MemberService()
        result = ms.load(list_file, replace_file)
        assert len(result) == 2
        assert "万丈目準" in result
        assert "オグリローマン" in result
        assert result["万丈目準"].replace_patterns == []

    def test_load_with_replace_data(self, tmp_path):
        """memberReplace.json の置換パターンがマージされる"""
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("テストメンバー\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        with open(replace_file, 'w', encoding='utf-8') as f:
            json.dump({"テストメンバー": ["テストメンバ"]}, f)

        ms = MemberService()
        result = ms.load(list_file, replace_file)
        assert "テストメンバー" in result
        assert result["テストメンバー"].replace_patterns == ["テストメンバ"]

    def test_load_with_invalid_replace_json(self, tmp_path):
        """壊れた JSON ファイルでもエラーにならない"""
        list_file = tmp_path / "memberList.txt"
        list_file.write_text("メンバーA\n", encoding="utf-8")
        replace_file = tmp_path / "memberReplace.json"
        replace_file.write_text("{不正なJSON}", encoding="utf-8")

        ms = MemberService()
        result = ms.load(list_file, replace_file)
        assert "メンバーA" in result
        assert result["メンバーA"].replace_patterns == []

    def test_save_roundtrip(self, tmp_path):
        """save → load のラウンドトリップ"""
        ms = MemberService()
        list_file = tmp_path / "memberList.txt"
        replace_file = tmp_path / "memberReplace.json"

        members = {
            "メンバーA": MemberEntry(name="メンバーA", replace_patterns=["パターン1"]),
            "メンバーB": MemberEntry(name="メンバーB"),
        }
        ms.save(members, list_file, replace_file)

        loaded = ms.load(list_file, replace_file)
        assert len(loaded) == 2
        assert loaded["メンバーA"].replace_patterns == ["パターン1"]
        assert loaded["メンバーB"].replace_patterns == []

    def test_save_creates_directory(self, tmp_path):
        """ディレクトリが存在しない場合は自動作成"""
        deep_dir = tmp_path / "sub" / "deep"
        list_file = deep_dir / "memberList.txt"
        replace_file = deep_dir / "memberReplace.json"

        ms = MemberService()
        members = {"テスト": MemberEntry(name="テスト")}
        ms.save(members, list_file, replace_file)
        assert list_file.exists()
        assert replace_file.exists()

    def test_save_excludes_empty_patterns(self, tmp_path):
        """置換パターンが空のメンバーは memberReplace.json に含まれない"""
        ms = MemberService()
        list_file = tmp_path / "memberList.txt"
        replace_file = tmp_path / "memberReplace.json"

        members = {
            "A": MemberEntry(name="A", replace_patterns=["x"]),
            "B": MemberEntry(name="B"),
        }
        ms.save(members, list_file, replace_file)

        with open(replace_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert "A" in data
        assert "B" not in data

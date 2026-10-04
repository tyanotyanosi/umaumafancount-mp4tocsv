# -*- coding: utf-8 -*-
"""Tests for ``conftest._ensure_extended_length_path_noop``.

Generated from: docs/00-Architecture/conftest___ensure_extended_length_path_noop.yaml

function:    conftest._ensure_extended_length_path_noop
signature:   def _ensure_extended_length_path_noop(path):
purpose:     _pytest.pathlib.ensure_extended_length_path の元実装に置き換え
             （conftest.py 38行目）、渡されたパスを何の変更も加えずそのまま返すことで、
             パスの \\?\\ 拡張長さパス化を無効化する
precondition: sys.platform.startswith("win32") が真であること（win32 でのみ
             この関数は conftest.py に定義される。実行環境は Windows のため成立）

（テスト対象外・文書化のみ）errors:
  - （errors セクションは空。エラー処理は関数内に存在しない）

unconfirmed（未決情報）:
  - この関数の呼び出し元（_pytest 側のパス処理）が戻り値を str として使うのか
    Path として使うのかは、担当ファイル外のため確認できない

missing（欠落情報）:
  - _pytest.pathlib のソース（元 ensure_extended_length_path の定義、およびこれを
    呼び出すパス処理コード）— 渡される引数の型・戻り値の使い方・呼び出しタイミングの
    仕様確定に必要
"""

import os
import sys

import pytest

# 本テストは tests/ 配下からプロジェクトルートの conftest.py を直接 import する。
# pytest の import モードに依存しないよう、プロジェクトルートを sys.path に確保する。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# conftest.py の win32 専用パッチ（_ensure_extended_length_path_noop）は
# sys.platform.startswith("win32") の下でのみ定義される。
# ubuntu-latest の CI では import 時点で ImportError になり collection 全体が
# 中断されるため、モジュールレベルで import 前にスキップする。
if not sys.platform.startswith("win32"):
    pytest.skip("win32 専用パッチのテスト", allow_module_level=True)

from conftest import _ensure_extended_length_path_noop  # noqa: E402


def test_edge_01():
    """edge_case #1
    input: '通常の絶対パス文字列（例: "C:\\\\temp\\\\foo"）'
    expected: '戻り値が入力値と同一のオブジェクトとなる'
    """
    p = "C:\\\\temp\\\\foo"
    result = _ensure_extended_length_path_noop(p)
    assert result is p, "同一オブジェクトを返す（前接辞の付加・変換はされない）"
    assert result == "C:\\\\temp\\\\foo"


def test_edge_02():
    """edge_case #2
    input: 'すでに \\\\?\\ 拡張長さ接頭辞を持つパス（例: "\\\\\\\\?\\\\C:\\\\temp\\\\foo"）'
    expected: '何の変更も加えられずそのまま返される（既存の接頭辞を除去する処理はない）'
    """
    # \\?\\ 拡張長さ接頭辞を先頭に持つパス
    p = "\\\\?\\C:\\\\temp\\\\foo"
    result = _ensure_extended_length_path_noop(p)
    assert result is p, "既存の接頭辞を除去する処理はないため同一オブジェクトが返る"
    assert result.startswith("\\\\?\\")
    assert result == p


def test_edge_03():
    """edge_case #3
    input: '任意の型オブジェクト（例: 整数 1）'
    expected: '同一オブジェクト 1 が返る（型チェック・型変換は一切行わない）'
    """
    v = 1
    result = _ensure_extended_length_path_noop(v)
    assert result is v, "型チェック・型変換は一切行わないため同一オブジェクトが返る"
    assert isinstance(result, int)

# -*- coding: utf-8 -*-
"""Tests for ``conftest._os_mkdir_ignore_mode``.

Generated from: docs/00-Architecture/conftest___os_mkdir_ignore_mode.yaml

function:    conftest._os_mkdir_ignore_mode
signature:   def _os_mkdir_ignore_mode(path, *args, **kwargs):
purpose:     win32 上の os.mkdir に置き換え（conftest.py 47行目）、path 以外の引数
             （mode 等）をすべて無視して、元の os.mkdir を path のみで呼び出す
precondition: sys.platform.startswith("win32") が真であること（実行環境は
             Windows のため成立）、モジュール変数 _os_mkdir_orig が元の
             os.mkdir を保持していること

（テスト対象外・文書化のみ）errors:
  - condition: '_os_mkdir_orig(path) の呼び出しが例外を送出する（例外種別は元の
    os.mkdir の動作に依存し、担当ファイル内に定義はない）'
    behavior: '捕捉・変換されず、呼び出し元にそのまま送出される'

unconfirmed（未決情報）:
  - キーワード引数もすべて破棄されるため、呼び出し側が exist_ok=True 等を渡した場合、
    元の os.mkdir の当該キーワード引数の効果（例: ディレクトリ存在時の非例外化）は
    効かない。実際の呼び出し側がどのキーワード引数を渡すかは担当ファイル外のため未確認
  - モジュール docstring には pytest の tmpdir 機構が mode=0o700 でディレクトリを
    作成すると書かれるが、その呼び出し元のコードは担当ファイル外のため未確認

missing（欠落情報）:
  - _pytest 内の os.mkdir を mode 付きで呼び出す呼び出し元のソース（モジュール
    docstring が示す _pytest/tmpdir.py 等）— 実際にこの関数へ渡される引数の集合確認に必要
  - 標準ライブラリ os.mkdir の仕様（シグネチャ・戻り値・送出する例外種別）—
    _os_mkdir_orig(path) の戻り値・例外動作の仕様確定に必要
"""

import os
import sys

# 本テストは tests/ 配下からプロジェクトルートの conftest.py を直接 import する。
# pytest の import モードに依存しないよう、プロジェクトルートを sys.path に確保する。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import conftest  # noqa: E402
from conftest import _os_mkdir_ignore_mode  # noqa: E402

import pytest
from unittest.mock import patch


def test_edge_01():
    """edge_case #1
    input: '_os_mkdir_orig を記録関数（例: def rec(p, *a, **k): return (p, a, k)）で
           置き換え、_os_mkdir_ignore_mode("p", 0o700, mode=0o600) を呼ぶ'
    expected: '記録関数は位置引数 ("p") のみ・キーワード引数なしで呼ばれ、この関数は
               記録関数の戻り値 ("p", (), {}) をそのまま返す'
    """
    calls = []

    def rec(p, *a, **k):
        calls.append((p, a, k))
        return (p, a, k)

    with patch.object(conftest, "_os_mkdir_orig", rec):
        result = _os_mkdir_ignore_mode("p", 0o700, mode=0o600)
    # 記録関数は位置引数 ("p") のみ・キーワード引数なしで呼ばれる
    assert calls == [("p", (), {})]
    # 記録関数の戻り値をそのまま返す
    assert result == ("p", (), {})


def test_edge_02():
    """edge_case #2
    input: '_os_mkdir_orig(path) が例外を送出する状況（例: 同名ディレクトリ存在・
           アクセス拒否）'
    expected: '例外は捕捉されず呼び出し元にそのまま送出される（関数内に try/except
               はない）'
    """

    def raise_orig(p, *a, **k):
        raise FileExistsError("同名ディレクトリが存在する（呼び出し例）")

    with patch.object(conftest, "_os_mkdir_orig", raise_orig):
        with pytest.raises(FileExistsError):
            _os_mkdir_ignore_mode("p")


def test_edge_03():
    """edge_case #3
    input: '_os_mkdir_ignore_mode("p", 0o700)（追加位置引数1つ）'
    expected: '0o700 は元の os.mkdir に渡されず、path のみで呼ばれた元の os.mkdir
               と等価の動作になる'
    """
    calls = []

    def rec(p, *a, **k):
        calls.append((p, a, k))
        return ("orig-result",)

    with patch.object(conftest, "_os_mkdir_orig", rec):
        result = _os_mkdir_ignore_mode("p", 0o700)
    # 0o700 は元の os.mkdir へ渡されない（path のみで呼ばれる）
    assert calls == [("p", (), {})]
    assert result == ("orig-result",)
